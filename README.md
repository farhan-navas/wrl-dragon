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

### Phase 2 — RL Meta-Training

Replace Claude Code with actual RL agents that:
- Meta-train over the orchestration process itself
- Train sub-RL-agents for OpenEnv tasks
- Optimize reward function generation and task assignment
- Aggregate and learn from cross-agent results

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

cp .env.example .env
# Add your ANTHROPIC_API_KEY
```

## Project Structure

```
wrl-dragon/
├── src/
│   ├── orchestrator/    # Claude Code CEO — task generation, agent coordination
│   ├── agents/          # Sub-agents (coders + workers)
│   ├── envs/            # OpenEnv environment definitions
│   ├── rewards/         # Generated reward functions
│   ├── logging/         # Rollout logs, metrics collection
│   └── dashboard/       # Observability web app (agent viz + rollout viewer)
├── pyproject.toml
└── README.md
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key for orchestrator |
