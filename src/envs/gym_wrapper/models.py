from __future__ import annotations

from pydantic import BaseModel


class GymAction(BaseModel):
    action: int | list[float]


class GymObservation(BaseModel):
    obs: list[float]
    reward: float
    done: bool
    info: dict = {}


class GymState(BaseModel):
    episode_id: str
    step_count: int
    total_reward: float
    done: bool


class ResetResponse(BaseModel):
    episode_id: str
    obs: list[float]


class StepResponse(BaseModel):
    obs: list[float]
    reward: float
    done: bool
    truncated: bool
    info: dict = {}
    total_reward: float
    step_count: int
