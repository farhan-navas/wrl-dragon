from __future__ import annotations

import uuid

import gymnasium as gym
import numpy as np


class GymEnvironment:
    def __init__(self, env_id: str = "CartPole-v1", render_mode: str | None = None):
        self.env_id = env_id
        self.render_mode = render_mode
        self.env: gym.Env | None = None
        self.episode_id: str = ""
        self.step_count: int = 0
        self.total_reward: float = 0.0
        self.done: bool = True

    def _ensure_env(self) -> gym.Env:
        if self.env is None:
            self.env = gym.make(self.env_id, render_mode=self.render_mode)
        return self.env

    def reset(self) -> dict:
        env = self._ensure_env()
        obs, info = env.reset()
        self.episode_id = str(uuid.uuid4())[:8]
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        return {
            "episode_id": self.episode_id,
            "obs": self._to_list(obs),
        }

    def step(self, action: int | list[float]) -> dict:
        if self.done:
            raise RuntimeError("Episode is done. Call reset() first.")
        env = self._ensure_env()
        if isinstance(action, list):
            action = np.array(action, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        self.step_count += 1
        self.total_reward += float(reward)
        self.done = terminated or truncated
        return {
            "obs": self._to_list(obs),
            "reward": float(reward),
            "done": self.done,
            "truncated": truncated,
            "info": {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool))},
            "total_reward": self.total_reward,
            "step_count": self.step_count,
        }

    def state(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "step_count": self.step_count,
            "total_reward": self.total_reward,
            "done": self.done,
        }

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    @staticmethod
    def _to_list(obs) -> list[float]:
        if isinstance(obs, np.ndarray):
            return obs.tolist()
        return list(obs)
