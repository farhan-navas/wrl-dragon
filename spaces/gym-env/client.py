"""
Gym environment client.

Provides both the OpenEnv EnvClient (async/sync WebSocket) and a lightweight
HTTP client for simple scripting.

Example (OpenEnv sync client):
    >>> env = GymEnv(base_url="http://localhost:8000").sync()
    >>> with env:
    ...     result = env.reset()
    ...     result = env.step({"value": 1})
    ...     print(result.observation)

Example (HTTP client):
    >>> client = GymEnvHttpClient("http://localhost:8000")
    >>> obs = client.reset()
    >>> client.close()
"""

from __future__ import annotations

from typing import Any, Dict

import httpx
from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient


class GymEnv(EnvClient[Dict[str, Any], Dict[str, Any], Dict[str, Any]]):
    """OpenEnv WebSocket client for GymEnvironment."""

    def _step_payload(self, action: Dict[str, Any]) -> Dict[str, Any]:
        return action

    def _parse_observation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data

    def _parse_state(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return data


class GymEnvHttpClient:
    """Lightweight synchronous HTTP client for the gym server."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=30.0)

    def reset(self, seed: int | None = None, episode_id: str | None = None) -> dict:
        body: dict[str, Any] = {}
        if seed is not None:
            body["seed"] = seed
        if episode_id is not None:
            body["episode_id"] = episode_id
        resp = self.client.post(f"{self.base_url}/reset", json=body)
        resp.raise_for_status()
        return resp.json()

    def step(self, action: int | list[float]) -> dict:
        resp = self.client.post(
            f"{self.base_url}/step",
            json={"action": {"value": action}},
        )
        resp.raise_for_status()
        return resp.json()

    def state(self) -> dict:
        resp = self.client.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self.client.close()
