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

def compute_syntax_reward(code: str, num_actions: int, obs_dim: int = 4) -> float:
    """Grade generated code on a 7-tier scale for finer gradient signal.

    0.0 = doesn't parse at all
    0.1 = parses but crashes on exec (import errors, name errors, etc.)
    0.2 = executes but no select_action function defined
    0.4 = select_action exists but is not callable
    0.6 = select_action callable but crashes on dummy obs
    0.8 = runs but returns out-of-range action
    1.0 = runs correctly and returns valid int in [0, num_actions)
    """
    clean = _strip_fences(code)

    # Tier 0: Parse
    try:
        ast.parse(clean)
    except SyntaxError:
        return 0.0

    # Tier 1: Exec
    ns: dict = {}
    try:
        exec(clean, {"__builtins__": __builtins__}, ns)
    except Exception:
        return 0.1

    # Tier 2: select_action defined
    if "select_action" not in ns:
        return 0.2

    # Tier 3: callable
    if not callable(ns["select_action"]):
        return 0.4

    # Tier 4-5: Test call with correct obs dimension
    dummy_obs = [0.0] * obs_dim
    try:
        result = int(ns["select_action"](dummy_obs))
        if 0 <= result < num_actions:
            return 1.0
        return 0.8
    except Exception:
        return 0.6


# ---------------------------------------------------------------------------
# Environment reward (runs rollouts — slower)
# ---------------------------------------------------------------------------

def compute_env_reward(
    code: str,
    env_name: str,
    num_episodes: int = 5,
    max_steps: int = 500,
) -> tuple[float, list[dict], int]:
    """Run the policy locally in a gym env (no server needed).

    Returns (mean_total_reward, per_episode_results, crashed_count).
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
        return 0.0, [], 0

    successful = [e for e in episodes if e.get("total_reward") is not None]
    crashed = len(episodes) - len(successful)
    if not successful:
        return 0.0, episodes, crashed

    mean_reward = sum(e["total_reward"] for e in successful) / len(successful)
    return mean_reward, episodes, crashed


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
    progress: float = 1.0,
    crashed_episodes: int = 0,
) -> float:
    """Combine syntax and environment rewards with curriculum weighting.

    Args:
        progress: Training progress in [0, 1]. 0=start, 1=end.
            Used to interpolate between start/end weights.
        crashed_episodes: Number of episodes that crashed during env eval.
            Each crash applies a penalty to discourage broken policies.
    """
    # Curriculum: interpolate weights based on training progress
    alpha = config.alpha_syntax_start + progress * (config.alpha_syntax_end - config.alpha_syntax_start)
    beta = config.beta_env_start + progress * (config.beta_env_end - config.beta_env_start)

    total = alpha * r_syntax + beta * r_env_normalized

    # Crash penalty: negative signal for policies that error during rollouts
    if crashed_episodes > 0:
        total += config.crash_penalty * crashed_episodes

    return max(total, 0.0)


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
