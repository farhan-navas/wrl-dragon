from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class GymAction(Action):
    """Action for gymnasium environments. Supports discrete (int) and continuous (list[float])."""

    value: Union[int, List[float]] = Field(
        ..., description="Discrete action (int) or continuous action (list of floats)"
    )


class GymObservation(Observation):
    """Observation from a gymnasium environment step."""

    obs: List[float] = Field(default_factory=list, description="Observation vector")
    truncated: bool = Field(default=False, description="Whether the episode was truncated")
    info: Dict[str, Any] = Field(default_factory=dict, description="Gym info dict")
    total_reward: float = Field(default=0.0, description="Cumulative reward this episode")
    env_id: Optional[str] = Field(default=None, description="Gymnasium environment ID")


class GymState(State):
    """State for gymnasium environment sessions."""

    total_reward: float = Field(default=0.0, description="Cumulative reward this episode")
    env_id: str = Field(default="", description="Gymnasium environment ID")
    is_done: bool = Field(default=True, description="Whether the episode is finished")
