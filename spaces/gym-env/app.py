"""Gym rollout server for HuggingFace Spaces.

Accepts policy code + env name, runs episodes, returns rewards.
Used by WRL-Dragon's Modal training and orchestrator as a remote gym backend.
"""

import random
import re
import traceback

import gymnasium as gym
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="WRL-Dragon Gym")


class RolloutRequest(BaseModel):
    env_name: str
    policy_code: str
    num_episodes: int = 5
    max_steps: int = 500


class RolloutResponse(BaseModel):
    rewards: list[float]
    crashed: int


@app.post("/rollout", response_model=RolloutResponse)
def rollout(req: RolloutRequest):
    """Run policy code against a gym environment and return episode rewards."""
    # Parse policy
    code = re.sub(r"^```(?:python)?\s*\n?", "", req.policy_code.strip())
    code = re.sub(r"\n?```\s*$", "", code.strip())
    if "def select_action" in code:
        code = code[code.index("def select_action"):]

    fn = None
    try:
        ns: dict = {}
        exec(code, {"__builtins__": __builtins__}, ns)
        if "select_action" in ns and callable(ns["select_action"]):
            fn = ns["select_action"]
    except Exception:
        pass

    env = gym.make(req.env_name)
    n = int(env.action_space.n) if hasattr(env.action_space, "n") else 2

    rewards: list[float] = []
    crashed = 0

    for _ in range(req.num_episodes):
        obs, _ = env.reset()
        total = 0.0
        ep_crashed = False

        for _ in range(req.max_steps):
            try:
                a = int(fn(obs.tolist())) if fn else random.randint(0, n - 1)
                a = a if 0 <= a < n else random.randint(0, n - 1)
            except Exception:
                a = random.randint(0, n - 1)
                ep_crashed = True

            obs, r, term, trunc, _ = env.step(a)
            total += r
            if term or trunc:
                break

        rewards.append(total)
        if ep_crashed:
            crashed += 1

    env.close()
    return RolloutResponse(rewards=rewards, crashed=crashed)


@app.get("/health")
def health():
    return {"status": "ok"}
