from __future__ import annotations

import argparse
import subprocess
import sys
import time

from src.orchestrator.ceo import CEOOrchestrator
from src.rollouts.executor import run_episodes


def main(env_name: str = "CartPole-v1", rounds: int = 2, episodes_per_round: int = 5):
    print(f"=== WRL-Dragon Orchestrator ===")
    print(f"Environment: {env_name}")
    print(f"Rounds: {rounds}, Episodes/round: {episodes_per_round}\n")

    # Start gym wrapper server in background
    print("Starting gym env server...")
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.envs.gym_wrapper.server.app:app",
         "--port", "8001", "--log-level", "warning"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(2)  # Wait for server startup

    try:
        ceo = CEOOrchestrator(env_name=env_name)
        ceo.spawn_agents()

        all_results = []
        for round_num in range(1, rounds + 1):
            print(f"\n--- Round {round_num}/{rounds} ---\n")

            # CEO plans
            plan = ceo.run_ceo_planning()

            # Coder generates reward shaping
            code = ceo.run_coder_generation(
                f"Round {round_num}: Design reward shaping for {env_name}"
            )

            # QA runs rollouts
            print(f"\nRunning {episodes_per_round} episodes...")
            results = run_episodes(
                env_url="http://localhost:8001",
                env_name=env_name,
                num_episodes=episodes_per_round,
            )
            all_results.extend(results)

            # Analyst reviews
            analysis = ceo.run_analyst_review(results)

        print(f"\n=== Orchestration Complete ===")
        print(f"Total episodes: {len(all_results)}")
        avg_reward = sum(r["total_reward"] for r in all_results) / len(all_results)
        print(f"Average reward: {avg_reward:.1f}")

    finally:
        server_proc.terminate()
        server_proc.wait()
        print("Server stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    main(env_name=args.env, rounds=args.rounds, episodes_per_round=args.episodes)
