from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.logging.events import register_ws_broadcaster, unregister_ws_broadcaster

app = FastAPI(title="WRL-Dragon API")

# In-memory agent registry (populated by orchestrator or manually)
_agents: list[dict] = []
_ws_connections: set[WebSocket] = set()


def set_agents(agents: list[dict]):
    global _agents
    _agents = agents


async def _broadcast(message: str):
    global _ws_connections
    dead = set()
    for ws in list(_ws_connections):
        try:
            await ws.send_text(message)
        except Exception:
            dead.add(ws)
    _ws_connections -= dead


register_ws_broadcaster(_broadcast)


@app.post("/internal/events")
async def post_event(request: Request):
    body = await request.body()
    await _broadcast(body.decode())
    return {"ok": True}


@app.post("/internal/agents")
async def post_agents(request: Request):
    global _agents
    _agents = await request.json()
    return {"ok": True}


@app.get("/api/agents")
def get_agents():
    return _agents


@app.get("/api/rollouts/{agent_id}")
def get_rollouts(agent_id: str, env: str | None = None):
    rollouts = []
    rollout_dirs = Path("outputs/rollouts").glob("*")
    for env_dir in rollout_dirs:
        if not env_dir.is_dir():
            continue
        if env and env_dir.name != env:
            continue
        for log_file in sorted(env_dir.glob("*.jsonl"), key=lambda f: f.stat().st_mtime):
            run_id = log_file.stem
            lines = log_file.read_text().strip().split("\n")
            if not lines or not lines[0]:
                continue
            steps = len(lines)
            total_reward = sum(json.loads(line).get("reward", 0) for line in lines if line)
            video_url = f"/api/videos/{run_id}"
            rollouts.append({
                "run_id": run_id,
                "env": env_dir.name,
                "total_reward": total_reward,
                "steps": steps,
                "video_url": video_url,
            })
    return rollouts


@app.get("/api/rewards/{run_id}")
def get_rewards(run_id: str):
    for env_dir in Path("outputs/rollouts").glob("*"):
        log_file = env_dir / f"{run_id}.jsonl"
        if log_file.exists():
            lines = log_file.read_text().strip().split("\n")
            rewards = []
            cumulative = 0.0
            for line in lines:
                if not line:
                    continue
                data = json.loads(line)
                cumulative += data.get("reward", 0)
                rewards.append({
                    "step": data["step"],
                    "reward": data.get("reward", 0),
                    "cumulative": cumulative,
                })
            return rewards
    return []


@app.get("/api/videos/{run_id}")
def get_video(run_id: str):
    for video_file in Path("outputs/videos").rglob(f"{run_id}*.mp4"):
        return FileResponse(video_file, media_type="video/mp4")
    return JSONResponse({"error": "Video not found"}, status_code=404)


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    await websocket.accept()
    _ws_connections.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _ws_connections.discard(websocket)


# Serve dashboard static files
dashboard_dir = Path(__file__).parent.parent / "dashboard"
if dashboard_dir.exists():
    app.mount("/", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
