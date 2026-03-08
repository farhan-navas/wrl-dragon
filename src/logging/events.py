from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

EVENTS_LOG_PATH = Path("outputs/logs/events.jsonl")

EventType = Literal[
    "run_started",
    "agent_spawned",
    "task_assigned",
    "rollout_started",
    "rollout_completed",
    "reward_update",
    # Orchestrator narrative events
    "phase_change",
    "insight",
    # Phase 2: training events
    "training_started",
    "training_step",
    "training_checkpoint",
    "training_completed",
]


class Event(BaseModel):
    type: EventType
    ts: float = Field(default_factory=time.time)

    # agent_spawned
    agent_id: str | None = None
    tier: str | None = None
    env: str | None = None
    name: str | None = None

    # task_assigned
    from_agent: str | None = Field(None, alias="from")
    to_agent: str | None = Field(None, alias="to")
    task: str | None = None

    # rollout_started / rollout_completed
    run_id: str | None = None

    # rollout_completed
    total_reward: float | None = None
    steps: int | None = None

    # reward_update
    step: int | None = None
    reward: float | None = None
    cumulative: float | None = None

    # agent status
    status: str | None = None
    current_task: str | None = None

    # phase_change / insight
    phase: str | None = None
    round_num: int | None = None
    total_rounds: int | None = None
    message: str | None = None
    source: str | None = None  # which agent produced the insight

    # training events (Phase 2)
    iteration: int | None = None
    loss: float | None = None
    mean_rewards: dict[str, float] | None = None
    best_rewards: dict[str, float] | None = None
    syntax_rate: float | None = None
    model_id: str | None = None
    envs: list[str] | None = None
    iterations: int | None = None
    elapsed_min: float | None = None
    checkpoint_path: str | None = None
    best_code: str | None = None

    model_config = {"populate_by_name": True}

    def to_json(self) -> str:
        data = {}
        for key, value in self.model_dump(by_alias=True).items():
            if value is not None:
                data[key] = value
        return json.dumps(data)


# Global list of WebSocket broadcast callbacks
_ws_broadcasters: list[Any] = []


def register_ws_broadcaster(callback):
    _ws_broadcasters.append(callback)


def unregister_ws_broadcaster(callback):
    _ws_broadcasters.remove(callback)


async def emit_event(event: Event):
    EVENTS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVENTS_LOG_PATH, "a") as f:
        f.write(event.to_json() + "\n")

    for broadcaster in _ws_broadcasters:
        try:
            await broadcaster(event.to_json())
        except Exception:
            pass


_API_URL = "http://localhost:8000/internal/events"


def emit_event_sync(event: Event):
    EVENTS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVENTS_LOG_PATH, "a") as f:
        f.write(event.to_json() + "\n")

    try:
        import httpx
        httpx.post(_API_URL, content=event.to_json(), headers={"Content-Type": "application/json"}, timeout=2.0)
    except Exception:
        pass
