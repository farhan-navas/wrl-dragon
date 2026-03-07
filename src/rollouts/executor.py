from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path

from openenv.core.generic_client import GenericEnvClient

from src.logging.events import Event, emit_event_sync


def run_episodes(
    env_url: str = "http://localhost:8001",
    env_name: str = "CartPole-v1",
    num_episodes: int = 5,
    max_steps: int = 500,
    agent_id: str = "qa-cartpole",
) -> list[dict]:
    """Run rollout episodes over OpenEnv WebSocket (stateful sessions)."""
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
                action = random.randint(0, 1)

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
            print(f"Episode {ep+1}/{num_episodes}: reward={total_reward:.1f}, steps={len(episode_data)}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--url", default="http://localhost:8001")
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()
    run_episodes(
        env_url=args.url,
        env_name=args.env,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
    )
