from __future__ import annotations

import json
import os
import subprocess

from src.logging.events import Event, emit_event_sync
from src.orchestrator.agents import (
    ANALYST_AGENT,
    CEO_AGENT,
    make_coder_agent,
    make_qa_agent,
)


def _query_api(system: str, prompt: str, max_tokens: int = 1024) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _query_cli(system: str, prompt: str, max_tokens: int = 1024) -> str:
    full_prompt = f"System: {system}\n\n{prompt}"
    result = subprocess.run(
        ["claude", "-p", full_prompt, "--no-input"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {result.stderr}")
    return result.stdout.strip()


DEFAULT_ENVS = ["CartPole-v1", "LunarLander-v3"]


class CEOOrchestrator:
    def __init__(self, env_names: list[str] | None = None, mode: str = "auto"):
        self.env_names = env_names or list(DEFAULT_ENVS)
        self.agents: dict[str, dict] = {}
        self.round = 0
        # Learning memory: accumulates insights across rounds
        self.memory: list[dict] = []
        # Generated code per env (latest round)
        self.generated_code: dict[str, str] = {}

        if mode == "auto":
            self.mode = "api" if os.environ.get("ANTHROPIC_API_KEY") else "cli"
        else:
            self.mode = mode

        print(f"Orchestrator mode: {self.mode}")
        print(f"Environments: {self.env_names}")

    def _query(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        if self.mode == "api":
            return _query_api(system, prompt, max_tokens)
        return _query_cli(system, prompt, max_tokens)

    def _memory_context(self) -> str:
        if not self.memory:
            return ""
        return (
            "\n\n## Learnings from previous rounds:\n"
            + json.dumps(self.memory, indent=2)
        )

    def spawn_agents(self):
        self.agents[CEO_AGENT["id"]] = CEO_AGENT
        emit_event_sync(Event(
            type="agent_spawned", agent_id=CEO_AGENT["id"], tier="ceo",
        ))

        self.agents[ANALYST_AGENT["id"]] = ANALYST_AGENT
        emit_event_sync(Event(
            type="agent_spawned", agent_id=ANALYST_AGENT["id"], tier="ceo",
        ))

        for env_name in self.env_names:
            coder = make_coder_agent(env_name)
            self.agents[coder["id"]] = coder
            emit_event_sync(Event(
                type="agent_spawned", agent_id=coder["id"], tier="coder", env=env_name,
            ))

            qa = make_qa_agent(env_name)
            self.agents[qa["id"]] = qa
            emit_event_sync(Event(
                type="agent_spawned", agent_id=qa["id"], tier="qa", env=env_name,
            ))

        print(f"Spawned agents: {list(self.agents.keys())}")

    # ── Phase 1: Generate ────────────────────────────────────────────

    def generate_all(self) -> dict[str, str]:
        """Generate code for ALL envs before any rollouts run."""
        self.round += 1
        self.generated_code = {}

        # CEO plans the whole batch
        plan = self._plan_batch()

        # Coders generate code for each env
        for env_name in self.env_names:
            code = self._generate_for_env(env_name)
            self.generated_code[env_name] = code

        return self.generated_code

    def _plan_batch(self) -> str:
        prompt = (
            f"Round {self.round}. Environments: {json.dumps(self.env_names)}. "
            f"Active agents: {json.dumps(list(self.agents.keys()))}. "
            "Plan code generation for ALL environments in this batch. "
            "For each env, specify what reward shaping / policy code to generate. "
            "Output a JSON plan with 'tasks' array, each with 'env', 'strategy', 'details'."
            + self._memory_context()
        )

        plan_text = self._query(CEO_AGENT["system_prompt"], prompt)
        print(f"CEO Batch Plan (round {self.round}):\n{plan_text}\n")
        return plan_text

    def _generate_for_env(self, env_name: str) -> str:
        coder = make_coder_agent(env_name)
        self.assign_task(CEO_AGENT["id"], coder["id"], f"Generate code for {env_name}")

        prev_code = ""
        for m in self.memory:
            if m.get("env") == env_name and m.get("code"):
                prev_code = m["code"]

        prompt = f"Environment: {env_name}. Round: {self.round}.\n"
        if prev_code:
            prompt += f"\nPrevious code that was tested:\n```python\n{prev_code}\n```\n"
        prompt += (
            "Write a reward shaping function in Python. "
            "Output only the function code."
            + self._memory_context()
        )

        code = self._query(coder["system_prompt"], prompt, max_tokens=2048)
        print(f"Coder [{env_name}] output:\n{code}\n")
        return code

    # ── Phase 2: Execute (called by runner) ──────────────────────────

    # (rollout execution happens in runner.py using the executor)

    # ── Phase 3: Learn ───────────────────────────────────────────────

    def learn_from_batch(self, batch_results: dict[str, list[dict]]) -> str:
        """Feed batch results back to CEO. Updates memory with insights."""
        # Analyst reviews the batch
        analysis = self._query(
            ANALYST_AGENT["system_prompt"],
            f"Review batch rollout results across ALL environments:\n"
            f"{json.dumps(batch_results, indent=2)}\n\n"
            "For each environment:\n"
            "1. What was the avg/min/max reward?\n"
            "2. Is performance improving vs previous rounds?\n"
            "3. What specific changes would improve the next round?\n"
            "Output structured JSON with 'per_env' analysis and 'cross_env' insights."
            + self._memory_context(),
        )
        print(f"Analyst batch review:\n{analysis}\n")

        # CEO synthesizes learnings
        learnings = self._query(
            CEO_AGENT["system_prompt"],
            f"Round {self.round} complete.\n\n"
            f"Analyst review:\n{analysis}\n\n"
            f"Batch results:\n{json.dumps(batch_results, indent=2)}\n\n"
            "Synthesize key learnings. For each environment, state:\n"
            "- What worked, what didn't\n"
            "- Concrete changes for next round's code generation\n"
            "- Any cross-environment patterns\n"
            "Output JSON with 'learnings' array, each with 'env', 'insight', 'next_action'."
            + self._memory_context(),
            max_tokens=2048,
        )
        print(f"CEO Learnings (round {self.round}):\n{learnings}\n")

        # Store in memory
        for env_name in self.env_names:
            env_results = batch_results.get(env_name, [])
            avg_reward = 0.0
            if env_results:
                avg_reward = sum(r["total_reward"] for r in env_results) / len(env_results)

            self.memory.append({
                "round": self.round,
                "env": env_name,
                "avg_reward": avg_reward,
                "num_episodes": len(env_results),
                "code": self.generated_code.get(env_name, ""),
                "analysis": analysis[:500],
                "learnings": learnings[:500],
            })

        return learnings

    # ── Helpers ───────────────────────────────────────────────────────

    def assign_task(self, from_id: str, to_id: str, task_description: str):
        emit_event_sync(Event(
            type="task_assigned",
            **{"from": from_id, "to": to_id},
            task=task_description,
        ))
        print(f"Task: {from_id} -> {to_id}: {task_description}")

    def get_agent_list(self) -> list[dict]:
        return [
            {
                "id": agent["id"],
                "tier": agent["tier"],
                "status": "active",
                "env": agent.get("env"),
                "current_task": None,
            }
            for agent in self.agents.values()
        ]
