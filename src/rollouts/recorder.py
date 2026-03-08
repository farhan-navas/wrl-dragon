from __future__ import annotations

import uuid
from pathlib import Path

import gymnasium as gym


def record_rollout(
    env_id: str = "CartPole-v1",
    max_steps: int = 500,
    output_dir: str | None = None,
) -> dict:
    run_id = str(uuid.uuid4())[:8]
    if output_dir is None:
        output_dir = f"outputs/videos/{env_id}"
    video_dir = Path(output_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(video_dir),
        name_prefix=run_id,
        episode_trigger=lambda _: True,
    )

    obs, info = env.reset()
    total_reward = 0.0
    steps = 0

    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    env.close()

    return {
        "run_id": run_id,
        "env": env_id,
        "total_reward": total_reward,
        "steps": steps,
        "video_dir": str(video_dir),
    }


def replay_with_video(
    env_id: str,
    run_id: str,
    actions: list[int],
) -> str | None:
    """Replay a list of actions in a local gym env and record video.

    Returns the video directory path, or None on failure.
    """
    video_dir = Path(f"outputs/videos/{env_id}")
    video_dir.mkdir(parents=True, exist_ok=True)

    try:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            name_prefix=run_id,
            episode_trigger=lambda _: True,
        )

        env.reset()
        for action in actions:
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        env.close()
        return str(video_dir)
    except Exception as e:
        print(f"  WARNING: Video recording failed for {run_id}: {e}")
        return None


if __name__ == "__main__":
    result = record_rollout()
    print(f"Recorded: {result}")
