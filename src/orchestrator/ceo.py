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


class CEOOrchestrator:
    def __init__(self, env_name: str = "CartPole-v1", mode: str = "auto"):
        self.env_name = env_name
        self.agents: dict[str, dict] = {}
        self.round = 0

        if mode == "auto":
            self.mode = "api" if os.environ.get("ANTHROPIC_API_KEY") else "cli"
        else:
            self.mode = mode

        print(f"Orchestrator mode: {self.mode}")

    def _query(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        if self.mode == "api":
            return _query_api(system, prompt, max_tokens)
        return _query_cli(system, prompt, max_tokens)

    def spawn_agents(self):
        # Register CEO
        self.agents[CEO_AGENT["id"]] = CEO_AGENT
        emit_event_sync(Event(
            type="agent_spawned",
            agent_id=CEO_AGENT["id"],
            tier="ceo",
        ))

        # Register Analyst
        self.agents[ANALYST_AGENT["id"]] = ANALYST_AGENT
        emit_event_sync(Event(
            type="agent_spawned",
            agent_id=ANALYST_AGENT["id"],
            tier="ceo",
        ))

        # Spawn coder for target env
        coder = make_coder_agent(self.env_name)
        self.agents[coder["id"]] = coder
        emit_event_sync(Event(
            type="agent_spawned",
            agent_id=coder["id"],
            tier="coder",
            env=self.env_name,
        ))

        # Spawn QA worker for target env
        qa = make_qa_agent(self.env_name)
        self.agents[qa["id"]] = qa
        emit_event_sync(Event(
            type="agent_spawned",
            agent_id=qa["id"],
            tier="qa",
            env=self.env_name,
        ))

        print(f"Spawned agents: {list(self.agents.keys())}")

    def run_ceo_planning(self) -> str:
        self.round += 1
        prompt = (
            f"Round {self.round}. Environment: {self.env_name}. "
            f"Active agents: {json.dumps(list(self.agents.keys()))}. "
            "Create a plan: assign the coder to write/improve a reward shaping function, "
            "then assign QA to run rollouts. Output a JSON plan with 'tasks' array, "
            "each with 'agent_id', 'action', 'details'."
        )

        plan_text = self._query(CEO_AGENT["system_prompt"], prompt)
        print(f"CEO Plan (round {self.round}):\n{plan_text}\n")
        return plan_text

    def assign_task(self, from_id: str, to_id: str, task_description: str):
        emit_event_sync(Event(
            type="task_assigned",
            **{"from": from_id, "to": to_id},
            task=task_description,
        ))
        print(f"Task: {from_id} -> {to_id}: {task_description}")

    def run_coder_generation(self, task: str) -> str:
        coder = make_coder_agent(self.env_name)
        self.assign_task(CEO_AGENT["id"], coder["id"], task)

        code = self._query(
            coder["system_prompt"],
            f"Environment: {self.env_name}. Task: {task}. "
            "Write a reward shaping function in Python. "
            "Output only the function code.",
            max_tokens=2048,
        )
        print(f"Coder output:\n{code}\n")
        return code

    def run_analyst_review(self, rollout_results: list[dict]) -> str:
        analysis = self._query(
            ANALYST_AGENT["system_prompt"],
            f"Review these rollout results for {self.env_name}:\n"
            f"{json.dumps(rollout_results, indent=2)}\n"
            "Analyze performance and suggest improvements.",
        )
        print(f"Analyst review:\n{analysis}\n")
        return analysis

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
