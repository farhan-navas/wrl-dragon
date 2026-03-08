# wrl-dragon

RL orchestrator that uses Claude Code to coordinate sub-agents running OpenEnv environments, with a visual observability platform.

## Overview

### Phase 1 — Claude Code Orchestrator (current)

Claude Code acts as a "CEO" orchestrator that:
- Generates task-specific code, reward functions, and environment configs
- Spawns sub-agents that each run their own OpenEnv environment
- Collects logs and renders videos of agent rollouts

**Observability platform** — a web app (ChatDev / PixelAgents style) that visualizes the hierarchy:
- Top-level view: CEO orchestrator, coder agents, worker agents
- Click on any worker to see visualized rollouts across rounds
- Real-time logs, reward curves, and video playback

```
┌─────────────────────────────────┐
│  Claude Code (CEO Orchestrator) │
│  - generates reward functions   │
│  - assigns tasks to sub-agents  │
│  - aggregates results           │
└──────────┬──────────────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Coder 1 │ │ Coder 2 │  ← generate env code, reward fns
└────┬────┘ └────┬────┘
     ▼           ▼
┌─────────┐ ┌─────────┐
│Worker 1 │ │Worker 2 │  ← run OpenEnv tasks, produce rollouts
└─────────┘ └─────────┘
     │           │
     ▼           ▼
  [logs, videos, reward curves]
         │
         ▼
  ┌──────────────┐
  │  Dashboard   │
  │  (web app)   │
  └──────────────┘
```

### Phase 2 — RL Meta-Training (current)

Replace Claude Code with GRPO (Group Relative Policy Optimization) that trains an LLM to **generate gym policies directly from environment descriptions**.

- Fine-tunes Qwen3-Coder-30B with LoRA via [Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl)
- GRPO reward = `0.3 × syntax_score + 0.7 × env_reward` (5 rollouts per policy)
- Supports CartPole-v1, LunarLander-v3, Acrobot-v1, MountainCar-v0
- Runs locally or on [Modal](https://modal.com) H200 GPUs
- Streams training events to the dashboard via webhook or local JSONL

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

cp .env.example .env
# Add your ANTHROPIC_API_KEY
```

## Running

### 1. Gym Env Server (standalone)

Starts the Gymnasium wrapper as a REST API on port 8001.

```bash
# Default: CartPole-v1
uvicorn src.envs.gym_wrapper.server.app:app --port 8001

# Custom env
GYM_ENV_ID=MountainCar-v0 uvicorn src.envs.gym_wrapper.server.app:app --port 8001
```

Test it:

```bash
curl -X POST http://localhost:8001/reset
curl -X POST http://localhost:8001/step -H "Content-Type: application/json" -d '{"action": 1}'
curl http://localhost:8001/state
```

### 2. Dashboard + API Server

Starts the FastAPI server on port 8000, serving both the REST/WebSocket API and the pixel-art dashboard.

```bash
uvicorn src.api.server:app --port 8000
```

Open [http://localhost:8000](http://localhost:8000) to see the 3-tier agent dashboard.

### 3. Rollout Executor (standalone)

Runs episodes against a running gym env server and saves JSONL logs to `outputs/rollouts/`.

```bash
# Requires gym env server running on :8001
python -m src.rollouts.executor --env CartPole-v1 --episodes 5 --max-steps 500
```

### 4. Video Recorder (standalone)

Records a rollout video using gymnasium's built-in recorder. Saves to `outputs/videos/`.

```bash
python -m src.rollouts.recorder
```

### 5. Meta-Training (Phase 2)

Train an LLM to generate gym policies using GRPO.

```bash
# Local training (requires GPU + Unsloth)
python -m src.meta.runner --envs CartPole-v1 LunarLander-v3 --iterations 20

# With dashboard integration
python -m src.meta.runner --envs CartPole-v1 --iterations 10 --webhook http://localhost:8000/api/training-event

# Skip dashboard events
python -m src.meta.runner --envs CartPole-v1 --iterations 10 --no-dashboard

# On Modal H200 GPU (works from any machine — no local GPU needed)
python -m src.meta.runner --modal --envs CartPole-v1 LunarLander-v3 --iterations 20

# Or directly via Modal CLI
modal run src/meta/modal_train.py --envs CartPole-v1,LunarLander-v3 --iterations 20
```

Outputs (adapters, summaries) are saved to `outputs/meta/` locally or `/vol/outputs/meta/` on Modal.

#### Live dashboard with Modal (ngrok tunnel)

Modal runs remotely, so it needs a public URL to reach your local dashboard. Use ngrok to tunnel port 8000:

```bash
# Terminal 1 — start dashboard
uvicorn src.api.server:app --port 8000

# Terminal 2 — expose it with ngrok
ngrok http 8000
# Copy the forwarding URL, e.g. https://abc123.ngrok-free.app

# Terminal 3 — run Modal training with the ngrok webhook
python -m src.meta.runner --modal --envs CartPole-v1 --iterations 20 \
  --webhook https://abc123.ngrok-free.app
```

Modal POSTs training events (`training_started`, `training_step`, `training_checkpoint`, `training_completed`) to `POST /api/events`, which logs them to `outputs/logs/events.jsonl` and broadcasts to WebSocket clients on the dashboard.

### 6. Full Orchestration (end-to-end)

Runs the complete CEO orchestration loop: spawns agents, plans via Claude, generates reward functions, runs rollouts, and analyzes results.

The orchestrator supports two modes:

```bash
# Auto-detect: uses API key if ANTHROPIC_API_KEY is set, otherwise uses Claude Code CLI
python -m src.orchestrator.runner --env CartPole-v1 --rounds 2 --episodes 5

# Explicit: use Claude Code CLI (subscription, no API key needed)
python -m src.orchestrator.runner --mode cli

# Explicit: use Anthropic API (requires ANTHROPIC_API_KEY in .env)
python -m src.orchestrator.runner --mode api
```

This automatically:
1. Starts the gym env server on :8001
2. Spawns CEO, Analyst, Coder, and QA agents
3. For each round: CEO plans → Coder generates reward shaping → QA runs rollouts → Analyst reviews
4. Writes structured event logs to `outputs/logs/events.jsonl`
5. Writes rollout data to `outputs/rollouts/`

To watch it live, run the dashboard server in a separate terminal first:

```bash
# Terminal 1
uvicorn src.api.server:app --port 8000

# Terminal 2
python -m src.orchestrator.runner
```

Then open [http://localhost:8000](http://localhost:8000) — agents appear in real-time, task assignments animate between tiers, and you can click QA workers to see rollout details.

## Project Structure

```
wrl-dragon/
├── src/
│   ├── meta/              # Phase 2: GRPO meta-training
│   │   ├── runner.py      # Local training entry point
│   │   ├── modal_train.py # Modal H200 deployment
│   │   ├── trainer.py     # GRPOTrainer factory + reward function
│   │   ├── config.py      # MetaConfig (model, reward weights, training params)
│   │   ├── prompt.py      # ENV_SPECS + prompt construction
│   │   ├── rewards.py     # Syntax + env reward computation (subprocess rollouts)
│   │   ├── events.py      # TrainingEventEmitter (webhook / local JSONL)
│   │   └── callbacks.py   # RewardAccumulator + DashboardCallback
│   ├── orchestrator/       # CEO agent, agent definitions, runner entry point
│   │   ├── ceo.py          # CEOOrchestrator using Claude API
│   │   ├── agents.py       # Agent role definitions (CEO, Analyst, Coder, QA)
│   │   └── runner.py       # Main entry point for full orchestration
│   ├── envs/
│   │   └── gym_wrapper/    # Gymnasium-to-OpenEnv wrapper
│   │       ├── models.py   # Pydantic models (GymAction, GymObservation, etc.)
│   │       ├── client.py   # HTTP client for gym env server
│   │       ├── openenv.yaml
│   │       └── server/
│   │           ├── app.py              # FastAPI app (port 8001)
│   │           └── gym_environment.py  # Core gym wrapper class
│   ├── rollouts/
│   │   ├── executor.py     # Episode runner, JSONL logging
│   │   └── recorder.py     # Video capture via gymnasium
│   ├── logging/
│   │   └── events.py       # Structured events + WebSocket broadcast
│   ├── api/
│   │   └── server.py       # FastAPI + WS server (port 8000)
│   └── dashboard/          # Pixel-art web dashboard
│       ├── index.html
│       ├── styles.css
│       ├── app.js          # Main entry, state management
│       ├── canvas/
│       │   ├── renderer.js # Game loop, click handling
│       │   ├── agents.js   # Pixel-art characters + animations
│       │   └── layout.js   # 3-tier office floor layout
│       ├── panels/
│       │   ├── rollout-viewer.js  # Video + log side panel
│       │   └── reward-chart.js    # Canvas reward curve chart
│       └── ws/
│           └── client.js   # WebSocket client with auto-reconnect
├── outputs/                # Runtime outputs (gitignored)
│   ├── logs/               # events.jsonl
│   ├── rollouts/           # Per-env JSONL rollout logs
│   └── videos/             # Rollout mp4 recordings
├── pyproject.toml
├── Changelog.md
└── README.md
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key for orchestrator |
| `GYM_ENV_ID` | Gymnasium env ID (default: `CartPole-v1`) |

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/agents` | List active agents with status |
| `GET /api/rollouts/{agent_id}` | Rollout logs for an agent |
| `GET /api/rewards/{run_id}` | Step-by-step reward data |
| `GET /api/videos/{run_id}` | Serve rollout mp4 video |
| `WS /ws/events` | Real-time event stream |
