from __future__ import annotations

import os
from typing import Any, Optional
from uuid import uuid4

import gymnasium as gym
import numpy as np
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from src.envs.gym_wrapper.models import GymAction, GymObservation, GymState


class GymEnvironment(Environment[GymAction, GymObservation, GymState]):
    """OpenEnv environment that wraps any gymnasium.make() env."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, env_id: str | None = None, render_mode: str | None = None):
        super().__init__()
        self.env_id = env_id or os.environ.get("GYM_ENV_ID", "CartPole-v1")
        self.render_mode = render_mode
        self.env: gym.Env | None = None
        self._state = GymState(env_id=self.env_id)

    def _ensure_env(self) -> gym.Env:
        if self.env is None:
            self.env = gym.make(self.env_id, render_mode=self.render_mode)
        return self.env

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> GymObservation:
        env = self._ensure_env()
        reset_kwargs = {}
        if seed is not None:
            reset_kwargs["seed"] = seed
        obs, info = env.reset(**reset_kwargs)

        self._state = GymState(
            episode_id=episode_id or str(uuid4())[:8],
            step_count=0,
            total_reward=0.0,
            env_id=self.env_id,
            is_done=False,
        )

        return GymObservation(
            obs=_to_list(obs),
            done=False,
            reward=0.0,
            truncated=False,
            info=_filter_info(info),
            total_reward=0.0,
            env_id=self.env_id,
            metadata={"episode_id": self._state.episode_id},
        )

    def step(
        self,
        action: GymAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> GymObservation:
        if self._state.is_done:
            return GymObservation(
                done=True,
                reward=0.0,
                metadata={"error": "Episode is done. Call reset() first."},
            )

        env = self._ensure_env()

        act = action.value
        if isinstance(act, list):
            act = np.array(act, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(act)

        self._state.step_count += 1
        self._state.total_reward += float(reward)
        self._state.is_done = terminated or truncated

        return GymObservation(
            obs=_to_list(obs),
            reward=float(reward),
            done=self._state.is_done,
            truncated=truncated,
            info=_filter_info(info),
            total_reward=self._state.total_reward,
            env_id=self.env_id,
        )

    @property
    def state(self) -> GymState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name=f"GymEnvironment({self.env_id})",
            description=f"OpenEnv wrapper for gymnasium {self.env_id}",
            version="0.2.0",
        )

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def sample_action(self) -> GymAction:
        env = self._ensure_env()
        raw = env.action_space.sample()
        if isinstance(raw, np.ndarray):
            return GymAction(value=raw.tolist())
        return GymAction(value=int(raw))

    def action_space_info(self) -> dict:
        env = self._ensure_env()
        space = env.action_space
        info: dict[str, Any] = {"type": type(space).__name__}
        if hasattr(space, "n"):
            info["n"] = int(space.n)
        if hasattr(space, "shape"):
            info["shape"] = list(space.shape)
        if hasattr(space, "low"):
            info["low"] = _to_list(space.low)
        if hasattr(space, "high"):
            info["high"] = _to_list(space.high)
        return info


def _to_list(obs: Any) -> list[float]:
    if isinstance(obs, np.ndarray):
        return obs.tolist()
    return list(obs)


def _filter_info(info: dict) -> dict:
    return {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool))}
