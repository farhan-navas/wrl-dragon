"""Reward functions for the RL^2 meta-training loop.

Two signals:
  R_syntax: Does the generated code compile and run?
  R_env:    How well does the policy perform in the gym env?

These are called from the GRPO reward function which receives
a list of completions and must return a list of floats.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
import textwrap
import traceback

from src.meta.config import RewardConfig

# ---------------------------------------------------------------------------
# Syntax reward (fast, no gym server needed)
# ---------------------------------------------------------------------------

def compute_syntax_reward(code: str, num_actions: int) -> float:
    """Grade generated code on a 0-1 scale.

    0.0 = doesn't parse
    0.3 = parses but no select_action
    0.7 = select_action exists but crashes on dummy input
    1.0 = runs correctly and returns valid int
    """
    clean = _strip_fences(code)

    # Parse
    try:
        ast.parse(clean)
    except SyntaxError:
        return 0.0

    # Exec
    ns: dict = {}
    try:
        exec(clean, {"__builtins__": __builtins__}, ns)
    except Exception:
        return 0.0

    if "select_action" not in ns or not callable(ns["select_action"]):
        return 0.3

    # Test call
    dummy_obs = [0.0] * 8
    try:
        result = int(ns["select_action"](dummy_obs))
        if 0 <= result < num_actions:
            return 1.0
        return 0.7
    except Exception:
        return 0.7


# ---------------------------------------------------------------------------
# Environment reward (runs rollouts — slower)
# ---------------------------------------------------------------------------

def compute_env_reward(
    code: str,
    env_name: str,
    num_episodes: int = 5,
    max_steps: int = 500,
) -> tuple[float, list[dict]]:
    """Run the policy locally in a gym env (no server needed).

    Returns (mean_total_reward, per_episode_results).
    Runs in a subprocess to isolate crashes.
    """
    # Write a self-contained eval script
    script = textwrap.dedent(f"""\
        import json, random, sys
        import gymnasium as gym

        code = {repr(code)}

        # Load policy
        namespace = {{}}
        try:
            exec(code, {{"__builtins__": __builtins__}}, namespace)
        except Exception:
            print(json.dumps([]))
            sys.exit(0)

        if "select_action" not in namespace:
            print(json.dumps([]))
            sys.exit(0)

        fn = namespace["select_action"]
        env = gym.make({repr(env_name)})
        n_actions = int(env.action_space.n) if hasattr(env.action_space, "n") else 2
        results = []

        for ep in range({num_episodes}):
            obs, info = env.reset()
            total_reward = 0.0
            steps = 0
            for _ in range({max_steps}):
                try:
                    action = int(fn(obs.tolist()))
                    if action < 0 or action >= n_actions:
                        action = random.randint(0, n_actions - 1)
                except Exception:
                    action = random.randint(0, n_actions - 1)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                if terminated or truncated:
                    break
            results.append({{"total_reward": total_reward, "steps": steps}})

        env.close()
        print(json.dumps(results))
    """)

    import json
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120,
        )
        episodes = json.loads(result.stdout) if result.stdout.strip() else []
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        episodes = []

    if not episodes:
        return 0.0, []

    mean_reward = sum(e["total_reward"] for e in episodes) / len(episodes)
    return mean_reward, episodes


def normalize_env_reward(
    raw_reward: float,
    env_name: str,
    config: RewardConfig,
) -> float:
    """Normalize raw env reward to [0, 1] using per-env baselines."""
    baseline, solved = config.env_baselines.get(env_name, (0.0, 100.0))
    denom = solved - baseline
    if denom == 0:
        return 0.0
    normalized = (raw_reward - baseline) / denom
    return max(0.0, min(1.0, normalized))


def compute_total_reward(
    r_syntax: float,
    r_env_normalized: float,
    config: RewardConfig,
) -> float:
    """Combine syntax and environment rewards."""
    return config.alpha_syntax * r_syntax + config.beta_env * r_env_normalized


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_fences(code: str) -> str:
    code = re.sub(r"^```(?:python)?\s*\n?", "", code.strip())
    code = re.sub(r"\n?```\s*$", "", code.strip())
    if "def select_action" in code:
        idx = code.index("def select_action")
        code = code[idx:]
    return code.strip()
