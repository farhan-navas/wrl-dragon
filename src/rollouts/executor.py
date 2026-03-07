from __future__ import annotations

import argparse
import json
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


def run_episodes(
    env_url: str = "http://localhost:8001",
    env_name: str = "CartPole-v1",
    num_episodes: int = 5,
    max_steps: int = 500,
    agent_id: str = "qa-cartpole",
    policy_code: str | None = None,
) -> list[dict]:
    """Run rollout episodes over OpenEnv WebSocket using the generated policy."""
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

            result = env.reset()
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
