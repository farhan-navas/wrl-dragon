"""FastAPI application for the Gym OpenEnv wrapper on HuggingFace Spaces."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from openenv.core.env_server.http_server import create_app

try:
    from gym_wrapper.models import GymAction, GymObservation
    from gym_wrapper.server.gym_environment import GymEnvironment
except ImportError:
    from models import GymAction, GymObservation
    from server.gym_environment import GymEnvironment


def create_gym_environment():
    """Factory function that creates GymEnvironment with config."""
    env_id = os.getenv("GYM_ENV_ID", "CartPole-v1")
    return GymEnvironment(env_id=env_id)


app = create_app(
    create_gym_environment,
    GymAction,
    GymObservation,
    env_name="gym_wrapper",
    max_concurrent_envs=8,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
