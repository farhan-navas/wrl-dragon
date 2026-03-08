---
title: WRL-Dragon Gym
emoji: "\U0001F409"
colorFrom: blue
colorTo: red
sdk: docker
app_port: 8000
---

# WRL-Dragon Gym Environment

An OpenEnv wrapper for [gymnasium](https://gymnasium.farama.org/) environments.
Supports any registered gym environment (CartPole, LunarLander, etc.) via a
standardized HTTP + WebSocket API.

## Quick Start

```python
from gym_wrapper import GymEnv

env = GymEnv(base_url="https://DESUCLUB-wrl-dragon-gym.hf.space").sync()
with env:
    result = env.reset()
    obs = result.observation["obs"]
    for _ in range(100):
        result = env.step({"value": 0})
        if result.observation["done"]:
            break
    print(f"Total reward: {result.observation['total_reward']}")
```

## Endpoints

- **POST /reset** — Start a new episode
- **POST /step** — Take an action
- **GET /state** — Current episode state
- **GET /health** — Health check
- **WebSocket /ws** — Real-time session

## Configuration

Set `GYM_ENV_ID` environment variable to change the default env (default: `CartPole-v1`).
