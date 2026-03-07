from __future__ import annotations

import httpx

from src.envs.gym_wrapper.models import GymState, ResetResponse, StepResponse


class GymEnvClient:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=30.0)

    def reset(self) -> ResetResponse:
        resp = self.client.post(f"{self.base_url}/reset")
        resp.raise_for_status()
        return ResetResponse(**resp.json())

    def step(self, action: int | list[float]) -> StepResponse:
        resp = self.client.post(f"{self.base_url}/step", json={"action": action})
        resp.raise_for_status()
        return StepResponse(**resp.json())

    def state(self) -> GymState:
        resp = self.client.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return GymState(**resp.json())

    def close(self):
        self.client.close()
