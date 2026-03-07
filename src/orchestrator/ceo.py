from __future__ import annotations

import json
import os

import anthropic

from src.logging.events import Event, emit_event_sync
from src.orchestrator.agents import (
    ANALYST_AGENT,
    CEO_AGENT,
    make_coder_agent,
    make_qa_agent,
)


class CEOOrchestrator:
    def __init__(self, env_name: str = "CartPole-v1"):
        self.env_name = env_name
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.agents: dict[str, dict] = {}
        self.round = 0

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

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=CEO_AGENT["system_prompt"],
            messages=[{"role": "user", "content": prompt}],
        )

        plan_text = response.content[0].text
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

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=coder["system_prompt"],
            messages=[{"role": "user", "content": (
                f"Environment: {self.env_name}. Task: {task}. "
                "Write a reward shaping function in Python. "
                "Output only the function code."
            )}],
        )

        code = response.content[0].text
        print(f"Coder output:\n{code}\n")
        return code

    def run_analyst_review(self, rollout_results: list[dict]) -> str:
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=ANALYST_AGENT["system_prompt"],
            messages=[{"role": "user", "content": (
                f"Review these rollout results for {self.env_name}:\n"
                f"{json.dumps(rollout_results, indent=2)}\n"
                "Analyze performance and suggest improvements."
            )}],
        )

        analysis = response.content[0].text
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
