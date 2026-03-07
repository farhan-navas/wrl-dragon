# Changelog

## [0.2.0] - 2026-03-07

### Added

#### Backend (Member A)

- **Gym-to-OpenEnv Wrapper** (`src/envs/gym_wrapper/`)
  - `GymEnvironment` server wrapping any `gymnasium.make(env_id)` with `reset()`, `step()`, `state()`
  - Pydantic models: `GymAction`, `GymObservation`, `GymState`, `ResetResponse`, `StepResponse`
  - `GymEnvClient` HTTP client with httpx
  - FastAPI server app on port 8001
  - OpenEnv manifest (`openenv.yaml`)

- **Orchestrator** (`src/orchestrator/`)
  - `CEOOrchestrator` — top-level agent using Anthropic Claude API to plan, assign, and analyze
  - Agent definitions: CEO, Analyst, Coder (per-env), QA Worker (per-env)
  - `runner.py` — main entry point: starts env server, runs CEO planning rounds, rollouts, and analysis

- **Rollout Engine** (`src/rollouts/`)
  - `executor.py` — runs N episodes against gym env via OpenEnv client, writes JSONL logs
  - `recorder.py` — captures rollout videos using `gymnasium.wrappers.RecordVideo`

- **Structured Events** (`src/logging/events.py`)
  - Event types: `agent_spawned`, `task_assigned`, `rollout_started`, `rollout_completed`, `reward_update`
  - JSONL file logging + WebSocket broadcast hooks

- **API Server** (`src/api/server.py`)
  - `GET /api/agents` — list active agents
  - `GET /api/rollouts/{agent_id}` — rollout logs per agent
  - `GET /api/rewards/{run_id}` — step-by-step reward data
  - `GET /api/videos/{run_id}` — serve rollout mp4 videos
  - `WS /ws/events` — real-time event stream
  - Serves dashboard static files

#### Frontend (Member B)

- **Dashboard** (`src/dashboard/`)
  - Pixel-art themed HTML/Canvas web app (dark theme, Press Start 2P font)
  - 3-tier office layout: CEO floor, Coder floor, QA Worker floor

- **Canvas Visualization** (`src/dashboard/canvas/`)
  - `renderer.js` — game loop with requestAnimationFrame, click/hover handlers, task line animations
  - `agents.js` — pixel-art characters with animation states (idle, thinking, coding, running, assigning, reporting), speech bubbles, blinking
  - `layout.js` — 3-tier floor layout with desks, monitors, connection lines

- **Rollout Viewer** (`src/dashboard/panels/`)
  - Side panel: video player, reward chart, step-by-step log, episode selector
  - `reward-chart.js` — canvas-based cumulative reward line chart with live update support

- **Real-Time Updates** (`src/dashboard/ws/client.js`)
  - WebSocket client with auto-reconnect
  - Event handlers: spawn agents, animate task assignments, rollout progress, reward updates

### Changed

- Added `httpx` dependency to `pyproject.toml`
- Added `.env.example` with `ANTHROPIC_API_KEY` and `GYM_ENV_ID`
