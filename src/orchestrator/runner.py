from __future__ import annotations

import argparse
import subprocess
import sys
import time

from src.orchestrator.ceo import DEFAULT_ENVS, CEOOrchestrator
from src.rollouts.executor import run_episodes

BASE_PORT = 8001


def start_env_server(env_name: str, port: int) -> subprocess.Popen:
    """Start an OpenEnv gym wrapper server for a specific env on a given port."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.envs.gym_wrapper.server.app:app",
         "--port", str(port), "--log-level", "warning"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**__import__("os").environ, "GYM_ENV_ID": env_name},
    )
    return proc


def main(
    env_names: list[str] | None = None,
    rounds: int = 2,
    episodes_per_round: int = 5,
    mode: str = "auto",
):
    envs = env_names or list(DEFAULT_ENVS)

    print(f"=== WRL-Dragon Orchestrator ===")
    print(f"Environments: {envs}")
    print(f"Rounds: {rounds}, Episodes/round: {episodes_per_round}\n")

    # Start one server per env on sequential ports
    env_ports: dict[str, int] = {}
    server_procs: list[subprocess.Popen] = []

    print("Starting env servers...")
    for i, env_name in enumerate(envs):
        port = BASE_PORT + i
        proc = start_env_server(env_name, port)
        server_procs.append(proc)
        env_ports[env_name] = port
        print(f"  {env_name} -> :{port}")
    time.sleep(2)

    try:
        ceo = CEOOrchestrator(env_names=envs, mode=mode)
        ceo.spawn_agents()

        all_results: dict[str, list[dict]] = {env: [] for env in envs}

        for round_num in range(1, rounds + 1):
            print(f"\n{'='*50}")
            print(f"  Round {round_num}/{rounds}")
            print(f"{'='*50}\n")

            # CEO plans across all envs
            plan = ceo.run_ceo_planning()

            for env_name in envs:
                port = env_ports[env_name]
                print(f"\n--- {env_name} (:{port}) ---\n")

                # Coder generates reward shaping
                code = ceo.run_coder_generation(
                    env_name,
                    f"Round {round_num}: Design reward shaping for {env_name}",
                )

                # QA runs rollouts
                print(f"Running {episodes_per_round} episodes...")
                results = run_episodes(
                    env_url=f"http://localhost:{port}",
                    env_name=env_name,
                    num_episodes=episodes_per_round,
                )
                all_results[env_name].extend(results)

            # Analyst reviews all envs together
            analysis = ceo.run_analyst_review(all_results)

        print(f"\n=== Orchestration Complete ===")
        for env_name, results in all_results.items():
            if results:
                avg = sum(r["total_reward"] for r in results) / len(results)
                print(f"  {env_name}: {len(results)} episodes, avg reward: {avg:.1f}")

    finally:
        for proc in server_procs:
            proc.terminate()
        for proc in server_procs:
            proc.wait()
        print("All servers stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", default=None,
                        help=f"Gym env IDs (default: {DEFAULT_ENVS})")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--mode", choices=["auto", "api", "cli"], default="auto",
                        help="auto: use API key if set, else Claude Code CLI. "
                             "api: require ANTHROPIC_API_KEY. cli: use 'claude -p' (subscription).")
    args = parser.parse_args()
    main(env_names=args.envs, rounds=args.rounds, episodes_per_round=args.episodes, mode=args.mode)
