from __future__ import annotations

import json
import os
import subprocess

import httpx

from src.logging.events import Event, emit_event_sync
from src.logging.orchestrator_log import (
    log_batch_results,
    log_error,
    log_generated_code,
    log_learnings,
    log_phase,
    log_query,
)
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
        ["claude", "-p", full_prompt, "--no-session-persistence"],
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

    def _query(self, system: str, prompt: str, max_tokens: int = 1024, agent_id: str = "unknown") -> str:
        try:
            if self.mode == "api":
                result = _query_api(system, prompt, max_tokens)
            else:
                result = _query_cli(system, prompt, max_tokens)
            log_query(self.round, agent_id, prompt, result, self.mode)
            return result
        except Exception as e:
            log_error(self.round, "query", e, agent_id=agent_id, prompt=prompt[:500])
            raise

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
            name=CEO_AGENT["name"],
        ))

        self.agents[ANALYST_AGENT["id"]] = ANALYST_AGENT
        emit_event_sync(Event(
            type="agent_spawned", agent_id=ANALYST_AGENT["id"], tier="ceo",
            name=ANALYST_AGENT["name"],
        ))

        for env_name in self.env_names:
            coder = make_coder_agent(env_name)
            self.agents[coder["id"]] = coder
            emit_event_sync(Event(
                type="agent_spawned", agent_id=coder["id"], tier="coder",
                env=env_name, name=coder["name"],
            ))

            qa = make_qa_agent(env_name)
            self.agents[qa["id"]] = qa
            emit_event_sync(Event(
                type="agent_spawned", agent_id=qa["id"], tier="qa",
                env=env_name, name=qa["name"],
            ))

        print(f"Spawned agents: {list(self.agents.keys())}")

        try:
            httpx.post(
                "http://localhost:8000/internal/agents",
                json=self.get_agent_list(),
                timeout=2.0,
            )
        except Exception:
            pass

    # ── Phase 1: Generate ────────────────────────────────────────────

    def generate_all(self, total_rounds: int = 1) -> dict[str, str]:
        """Generate code for ALL envs before any rollouts run."""
        self.round += 1
        self.generated_code = {}
        log_phase(self.round, "generate", envs=self.env_names)

        emit_event_sync(Event(
            type="phase_change", phase="generate",
            round_num=self.round, total_rounds=total_rounds,
            message=f"CEO is strategizing for {len(self.env_names)} environments...",
            agent_id=CEO_AGENT["id"],
        ))

        # CEO plans the whole batch
        plan = self._plan_batch()

        # Coders generate code for each env
        for env_name in self.env_names:
            try:
                code = self._generate_for_env(env_name)
                self.generated_code[env_name] = code
                log_generated_code(self.round, env_name, code)

                emit_event_sync(Event(
                    type="insight", source="coder",
                    agent_id=make_coder_agent(env_name)["id"],
                    env=env_name, round_num=self.round,
                    message=f"Code generated for {env_name} ({len(code)} chars)",
                ))
            except Exception as e:
                log_error(self.round, "generate", e, env=env_name)
                self.generated_code[env_name] = f"# ERROR: {e}"

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

        plan_text = self._query(CEO_AGENT["system_prompt"], prompt, agent_id=CEO_AGENT["id"])
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
            "Write a Python function `select_action(obs: list[float]) -> int` that picks "
            "the best action given an observation vector. The function must be self-contained "
            "(no global state, no classes). You may import any standard library or common packages "
            "(math, random, numpy, etc.) — put all imports at the top of the code. "
            "It receives the raw observation list and returns an integer action index. "
            "Use your knowledge of the environment dynamics to write a heuristic policy. "
            "Output ONLY the Python code (imports + function), no explanation, no markdown fences."
            + self._memory_context()
        )

        code = self._query(coder["system_prompt"], prompt, max_tokens=2048, agent_id=coder["id"])
        print(f"Coder [{env_name}] output:\n{code}\n")
        return code

    # ── Phase 2: Execute (called by runner) ──────────────────────────

    # (rollout execution happens in runner.py using the executor)

    # ── Phase 3: Learn ───────────────────────────────────────────────

    def learn_from_batch(self, batch_results: dict[str, list[dict]], total_rounds: int = 1) -> str:
        """Feed batch results back to CEO. Updates memory with insights."""
        log_phase(self.round, "learn", envs=self.env_names)
        log_batch_results(self.round, batch_results)

        # Emit batch summary insight
        for env_name, results in batch_results.items():
            if results:
                avg = sum(r["total_reward"] for r in results) / len(results)
                best = max(r["total_reward"] for r in results)
                emit_event_sync(Event(
                    type="insight", source="system",
                    env=env_name, round_num=self.round,
                    message=f"{env_name}: avg reward {avg:.1f}, best {best:.1f} ({len(results)} episodes)",
                ))

        emit_event_sync(Event(
            type="phase_change", phase="learn",
            round_num=self.round, total_rounds=total_rounds,
            message="Analyst is reviewing results...",
            agent_id=ANALYST_AGENT["id"],
        ))

        # Analyst reviews the batch
        try:
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
                agent_id=ANALYST_AGENT["id"],
            )
        except Exception as e:
            log_error(self.round, "learn", e, step="analyst_review")
            analysis = f"Analyst error: {e}"
        print(f"Analyst batch review:\n{analysis}\n")

        # Extract a short summary from the analysis for the feed
        emit_event_sync(Event(
            type="insight", source="analyst",
            agent_id=ANALYST_AGENT["id"], round_num=self.round,
            message="Batch analysis complete. Identifying patterns and improvements.",
        ))

        # CEO synthesizes learnings
        try:
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
                agent_id=CEO_AGENT["id"],
            )
        except Exception as e:
            log_error(self.round, "learn", e, step="ceo_synthesis")
            learnings = f"CEO synthesis error: {e}"
        print(f"CEO Learnings (round {self.round}):\n{learnings}\n")

        emit_event_sync(Event(
            type="insight", source="ceo",
            agent_id=CEO_AGENT["id"], round_num=self.round,
            message=f"Round {self.round} learnings synthesized. Adapting strategy for next round.",
        ))

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

        log_learnings(self.round, analysis, learnings, self.memory)
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
                "name": agent.get("name", agent["id"]),
                "status": "active",
                "env": agent.get("env"),
                "current_task": None,
            }
            for agent in self.agents.values()
        ]
