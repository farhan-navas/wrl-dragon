"""Structured logging for the orchestrator.

Writes to:
  outputs/logs/orchestrator.jsonl  — structured log entries (errors, phases, results)
  outputs/logs/code/               — generated code files per env per round
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any

LOG_DIR = Path("outputs/logs")
CODE_DIR = LOG_DIR / "code"
LOG_PATH = LOG_DIR / "orchestrator.jsonl"


def _ensure_dirs():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CODE_DIR.mkdir(parents=True, exist_ok=True)


def _write_entry(entry: dict):
    _ensure_dirs()
    entry.setdefault("ts", time.time())
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def log_phase(round_num: int, phase: str, **kwargs: Any):
    _write_entry({
        "level": "INFO",
        "round": round_num,
        "phase": phase,
        **kwargs,
    })


def log_generated_code(round_num: int, env_name: str, code: str):
    _ensure_dirs()
    # Save code to file for easy inspection
    safe_name = env_name.replace("-", "_").replace("/", "_")
    code_path = CODE_DIR / f"{safe_name}_round_{round_num}.py"
    code_path.write_text(code)

    _write_entry({
        "level": "INFO",
        "round": round_num,
        "phase": "generate",
        "env": env_name,
        "event": "code_generated",
        "code_path": str(code_path),
        "code_length": len(code),
        "code": code,
    })


def log_batch_results(round_num: int, batch_results: dict[str, list[dict]]):
    summary = {}
    for env_name, results in batch_results.items():
        if results:
            rewards = [r["total_reward"] for r in results]
            summary[env_name] = {
                "episodes": len(results),
                "avg_reward": sum(rewards) / len(rewards),
                "min_reward": min(rewards),
                "max_reward": max(rewards),
            }
        else:
            summary[env_name] = {"episodes": 0, "error": "no results"}

    _write_entry({
        "level": "INFO",
        "round": round_num,
        "phase": "execute",
        "event": "batch_complete",
        "summary": summary,
        "raw_results": batch_results,
    })


def log_learnings(round_num: int, analysis: str, learnings: str, memory: list[dict]):
    _write_entry({
        "level": "INFO",
        "round": round_num,
        "phase": "learn",
        "event": "learnings_stored",
        "analysis": analysis,
        "learnings": learnings,
        "memory_size": len(memory),
    })


def log_error(round_num: int, phase: str, error: Exception, **kwargs: Any):
    _write_entry({
        "level": "ERROR",
        "round": round_num,
        "phase": phase,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        **kwargs,
    })


def log_query(round_num: int, agent_id: str, prompt: str, response: str, mode: str):
    _write_entry({
        "level": "DEBUG",
        "round": round_num,
        "phase": "query",
        "agent_id": agent_id,
        "mode": mode,
        "prompt_length": len(prompt),
        "response_length": len(response),
        "prompt": prompt[:2000],
        "response": response[:2000],
    })
