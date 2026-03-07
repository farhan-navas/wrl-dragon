from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from src.envs.gym_wrapper.models import GymAction, GymState, ResetResponse, StepResponse
from src.envs.gym_wrapper.server.gym_environment import GymEnvironment

app = FastAPI(title="Gym OpenEnv Wrapper")

ENV_ID = os.environ.get("GYM_ENV_ID", "CartPole-v1")
env = GymEnvironment(env_id=ENV_ID)


@app.post("/reset", response_model=ResetResponse)
def reset():
    result = env.reset()
    return result


@app.post("/step", response_model=StepResponse)
def step(action: GymAction):
    if env.done:
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset first.")
    result = env.step(action.action)
    return result


@app.get("/state", response_model=GymState)
def state():
    return env.state()


@app.on_event("shutdown")
def shutdown():
    env.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
