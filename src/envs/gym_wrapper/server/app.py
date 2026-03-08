"""
FastAPI application for the Gym OpenEnv Wrapper.

Uses OpenEnv's create_app to get standard HTTP + WebSocket endpoints
(reset, step, state) with concurrent session support.

Usage:
    uvicorn src.envs.gym_wrapper.server.app:app --port 8001
"""

import os

# Force SDL/pygame to use offscreen rendering (no display needed for rgb_array).
# Without this, gymnasium's pygame renderer crashes on macOS when called from
# a non-main thread (e.g. WebSocket handler).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from openenv.core.env_server.http_server import create_app

from src.envs.gym_wrapper.models import GymAction, GymObservation
from src.envs.gym_wrapper.server.gym_environment import GymEnvironment

app = create_app(
    GymEnvironment,
    GymAction,
    GymObservation,
    env_name="gym_wrapper",
    max_concurrent_envs=10,
)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
