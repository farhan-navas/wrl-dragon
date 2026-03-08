from __future__ import annotations

import argparse
import json
import os
import random
import re
import traceback
import uuid
from pathlib import Path
from typing import Callable

import gymnasium as gym
from openenv.core.generic_client import GenericEnvClient

from src.logging.events import Event, emit_event_sync
from src.logging.orchestrator_log import log_error

# HF Space URL for remote rollouts (set via env or passed explicitly)
GYM_SPACE_URL = os.environ.get("WRL_DRAGON_GYM_URL", "")

_action_space_cache: dict[str, int] = {}


def _get_num_actions(env_name: str) -> int:
    """Get the number of discrete actions for a gym env.

    Caches results to avoid repeated gym.make() calls (which are not
    thread-safe due to gymnasium's lazy imports).
    """
    if env_name in _action_space_cache:
        return _action_space_cache[env_name]
    tmp = gym.make(env_name)
    n = int(tmp.action_space.n) if hasattr(tmp.action_space, "n") else 2
    tmp.close()
    _action_space_cache[env_name] = n
    return n


def load_policy(code: str, num_actions: int) -> Callable[[list[float]], int]:
    """Load a select_action(obs) function from generated code string.

    Falls back to random policy if the code is invalid.
    """
    # Strip markdown fences if the LLM included them
    code = re.sub(r"^```(?:python)?\s*\n?", "", code.strip())
    code = re.sub(r"\n?```\s*$", "", code.strip())

    namespace: dict = {}
    try:
        exec(code, {"__builtins__": __builtins__}, namespace)
    except Exception as e:
        print(f"  WARNING: Failed to compile policy code: {e}")
        log_error(0, "load_policy", e, code=code[:500])
        return lambda obs: random.randint(0, num_actions - 1)

    if "select_action" not in namespace:
        print(f"  WARNING: No select_action() found in generated code, using random")
        return lambda obs: random.randint(0, num_actions - 1)

    fn = namespace["select_action"]

    # Wrap with error handling so a bad policy doesn't crash the rollout
    def safe_policy(obs: list[float]) -> int:
        try:
            action = fn(obs)
            action = int(action)
            if action < 0 or action >= num_actions:
                return random.randint(0, num_actions - 1)
            return action
        except Exception:
            return random.randint(0, num_actions - 1)

    return safe_policy


def _try_remote_rollout(
    env_name: str,
    policy_code: str,
    num_episodes: int,
    max_steps: int,
    gym_space_url: str | None = None,
) -> list[dict] | None:
    """Try running rollout via HF Space OpenEnv server. Returns results list or None on failure."""
    url = gym_space_url or GYM_SPACE_URL
    if not url:
        return None

    num_actions = _get_num_actions(env_name)
    policy = load_policy(policy_code, num_actions) if policy_code else None

    try:
        import httpx
        base = url.rstrip("/")
        client = httpx.Client(timeout=30.0)
        results = []

        for ep in range(num_episodes):
            run_id = str(uuid.uuid4())[:8]
            resp = client.post(f"{base}/reset", json={"episode_id": run_id})
            if resp.status_code != 200:
                raise RuntimeError(f"reset failed: {resp.status_code}")
            obs_data = resp.json()
            obs = obs_data.get("observation", {}).get("obs", [])
            total_reward = 0.0
            steps = 0

            for step in range(max_steps):
                action = policy(obs) if policy else random.randint(0, num_actions - 1)
                resp = client.post(f"{base}/step", json={"action": {"value": action}})
                if resp.status_code != 200:
                    break
                step_data = resp.json()
                obs_info = step_data.get("observation", {})
                reward = obs_info.get("reward", 0.0)
                done = obs_info.get("done", False)
                total_reward += reward
                obs = obs_info.get("obs", [])
                steps += 1
                if done:
                    break

            results.append({
                "run_id": run_id,
                "env": env_name,
                "total_reward": total_reward,
                "steps": steps,
            })

        client.close()
        return results
    except Exception as e:
        print(f"  HF Space rollout failed ({e}), falling back to local")
    return None


def run_episodes(
    env_url: str = "http://localhost:8001",
    env_name: str = "CartPole-v1",
    num_episodes: int = 5,
    max_steps: int = 500,
    agent_id: str = "qa-cartpole",
    policy_code: str | None = None,
    gym_space_url: str | None = None,
) -> list[dict]:
    """Run rollout episodes. Tries HF Space first, falls back to local OpenEnv."""
    # Try remote HF Space first
    if policy_code:
        remote_results = _try_remote_rollout(
            env_name, policy_code, num_episodes, max_steps, gym_space_url,
        )
        if remote_results is not None:
            print(f"  [{env_name}] {len(remote_results)} episodes via HF Space")
            for r in remote_results:
                emit_event_sync(Event(
                    type="rollout_completed",
                    agent_id=agent_id,
                    run_id=r["run_id"],
                    total_reward=r["total_reward"],
                    steps=r["steps"],
                ))
            return remote_results

    # Fallback: local OpenEnv server
    num_actions = _get_num_actions(env_name)

    # Load generated policy or fall back to random
    if policy_code:
        policy = load_policy(policy_code, num_actions)
        print(f"  [{env_name}] Using generated policy")
    else:
        policy = lambda obs: random.randint(0, num_actions - 1)
        print(f"  [{env_name}] No policy code provided, using random")

    env = GenericEnvClient(base_url=env_url)
    results = []

    with env:
        for ep in range(num_episodes):
            run_id = str(uuid.uuid4())[:8]
            rollout_dir = Path(f"outputs/rollouts/{env_name}")
            rollout_dir.mkdir(parents=True, exist_ok=True)
            log_path = rollout_dir / f"{run_id}.jsonl"

            emit_event_sync(Event(
                type="rollout_started",
                agent_id=agent_id,
                env=env_name,
                run_id=run_id,
            ))

            result = env.reset(episode_id=run_id)
            obs = result.observation.get("obs", [])
            episode_data = []
            total_reward = 0.0

            for step_num in range(max_steps):
                action = policy(obs)

                result = env.step({"value": action})
                reward = result.reward or 0.0
                done = result.done

                step_record = {
                    "step": step_num,
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    "done": done,
                }
                episode_data.append(step_record)
                total_reward += reward
                obs = result.observation.get("obs", [])

                if step_num % 10 == 0:
                    emit_event_sync(Event(
                        type="reward_update",
                        agent_id=agent_id,
                        run_id=run_id,
                        step=step_num,
                        reward=reward,
                        cumulative=total_reward,
                    ))

                if done:
                    break

            with open(log_path, "w") as f:
                for record in episode_data:
                    f.write(json.dumps(record) + "\n")

            emit_event_sync(Event(
                type="rollout_completed",
                agent_id=agent_id,
                run_id=run_id,
                total_reward=total_reward,
                steps=len(episode_data),
            ))

            results.append({
                "run_id": run_id,
                "env": env_name,
                "total_reward": total_reward,
                "steps": len(episode_data),
            })
            print(f"  [{env_name}] ep {ep+1}/{num_episodes}: reward={total_reward:.1f}, steps={len(episode_data)}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--url", default="http://localhost:8001")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--policy-file", default=None, help="Path to .py file with select_action(obs)")
    args = parser.parse_args()

    code = None
    if args.policy_file:
        code = Path(args.policy_file).read_text()

    run_episodes(
        env_url=args.url,
        env_name=args.env,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        policy_code=code,
    )
