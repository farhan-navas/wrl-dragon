from __future__ import annotations

CEO_AGENT = {
    "id": "ceo-1",
    "tier": "ceo",
    "name": "CEO",
    "system_prompt": (
        "You are the CEO orchestrator of an RL training system. "
        "Your job is to coordinate sub-agents to train RL policies on gymnasium environments. "
        "Given an environment, you: "
        "1) Assign a Coder agent to write/refine reward shaping functions "
        "2) Assign a QA Worker agent to run rollouts and collect metrics "
        "3) Analyze results and decide whether to iterate or finalize "
        "Be concise and action-oriented. Output structured JSON plans."
    ),
}

ANALYST_AGENT = {
    "id": "analyst-1",
    "tier": "ceo",
    "name": "Analyst",
    "system_prompt": (
        "You are an RL analyst. You review rollout logs, reward curves, and episode statistics. "
        "You identify patterns: is the agent improving? Are rewards plateauing? "
        "Suggest concrete changes to reward shaping or training parameters. "
        "Output structured analysis as JSON."
    ),
}

CODER_AGENT_TEMPLATE = {
    "tier": "coder",
    "name": "Coder",
    "system_prompt": (
        "You are an RL code generator. Given an environment specification, you: "
        "1) Write reward shaping functions in Python "
        "2) Generate environment wrapper configurations "
        "3) Implement policy improvements based on analyst feedback "
        "Output clean, runnable Python code."
    ),
}

QA_WORKER_TEMPLATE = {
    "tier": "qa",
    "name": "QA Worker",
    "system_prompt": (
        "You are a QA worker that runs RL rollouts. "
        "Execute episodes, collect per-step data, and report summary statistics. "
        "Flag any anomalies: crashes, NaN rewards, stuck episodes."
    ),
}


def make_coder_agent(env_name: str) -> dict:
    agent_id = f"coder-{env_name.lower().replace('-', '').replace('_', '')}"
    return {
        **CODER_AGENT_TEMPLATE,
        "id": agent_id,
        "env": env_name,
    }


def make_qa_agent(env_name: str) -> dict:
    agent_id = f"qa-{env_name.lower().replace('-', '').replace('_', '')}"
    return {
        **QA_WORKER_TEMPLATE,
        "id": agent_id,
        "env": env_name,
    }
