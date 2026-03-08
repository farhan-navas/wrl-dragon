from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.logging.events import EVENTS_LOG_PATH, Event, emit_event_sync
from src.logging.orchestrator_log import log_error, log_phase
from src.orchestrator.ceo import DEFAULT_ENVS, CEOOrchestrator
from src.rollouts.executor import _get_num_actions, run_episodes

BASE_PORT = 8001


def start_env_server(env_name: str, port: int) -> subprocess.Popen:
    """Start an OpenEnv gym wrapper server for a specific env on a given port."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.envs.gym_wrapper.server.app:app",
         "--port", str(port), "--log-level", "warning"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "GYM_ENV_ID": env_name},
    )
    return proc


GYM_SPACE_URL = os.environ.get("WRL_DRAGON_GYM_URL", "")


def run_batch(
    env_ports: dict[str, int],
    episodes: int,
    max_steps: int = 500,
    generated_code: dict[str, str] | None = None,
    gym_space_url: str | None = None,
) -> dict[str, list[dict]]:
    """Run all envs in parallel, return batch results."""
    batch_results: dict[str, list[dict]] = {}
    generated_code = generated_code or {}
    space_url = gym_space_url or GYM_SPACE_URL

    with ThreadPoolExecutor(max_workers=len(env_ports)) as pool:
        futures = {}
        for env_name, port in env_ports.items():
            agent_id = f"qa-{env_name.lower().replace('-', '').replace('_', '')}"
            future = pool.submit(
                run_episodes,
                env_url=f"http://localhost:{port}",
                env_name=env_name,
                num_episodes=episodes,
                max_steps=max_steps,
                agent_id=agent_id,
                policy_code=generated_code.get(env_name),
                gym_space_url=space_url or None,
            )
            futures[future] = env_name

        for future in as_completed(futures):
            env_name = futures[future]
            try:
                results = future.result()
                batch_results[env_name] = results
            except Exception as e:
                print(f"  ERROR [{env_name}]: {e}")
                log_error(0, "execute", e, env=env_name)
                batch_results[env_name] = []

    return batch_results


def main(
    env_names: list[str] | None = None,
    rounds: int = 2,
    episodes_per_round: int = 5,
    mode: str = "auto",
):
    envs = env_names or list(DEFAULT_ENVS)

    print(f"=== WRL-Dragon Orchestrator ===")
    print(f"Environments: {envs}")
    print(f"Rounds: {rounds}, Episodes/round: {episodes_per_round}")
    print(f"Flow: Generate All -> Run Batch -> Learn -> Repeat\n")

    # Start one server per env
    env_ports: dict[str, int] = {}
    server_procs: list[subprocess.Popen] = []

    # Pre-warm gym action space cache on main thread (gymnasium imports aren't thread-safe)
    print("Pre-loading gym environments...")
    for env_name in envs:
        n = _get_num_actions(env_name)
        print(f"  {env_name}: {n} actions")

    print("Starting env servers...")
    for i, env_name in enumerate(envs):
        port = BASE_PORT + i
        proc = start_env_server(env_name, port)
        server_procs.append(proc)
        env_ports[env_name] = port
        print(f"  {env_name} -> :{port}")
    time.sleep(3)

    try:
        # Clear old logs and signal a fresh run — clears dashboard state
        if EVENTS_LOG_PATH.exists():
            EVENTS_LOG_PATH.write_text("")
        emit_event_sync(Event(type="run_started", message=f"New run: {rounds} rounds, {len(envs)} envs"))

        ceo = CEOOrchestrator(env_names=envs, mode=mode)
        ceo.spawn_agents()

        cumulative: dict[str, list[dict]] = {env: [] for env in envs}

        for round_num in range(1, rounds + 1):
            print(f"\n{'='*60}")
            print(f"  ROUND {round_num}/{rounds}")
            print(f"{'='*60}")

            # ── Phase 1: Generate all code ───────────────────────
            print(f"\n>> Phase 1: GENERATE (all {len(envs)} envs)")
            generated = ceo.generate_all(total_rounds=rounds)
            for env_name, code in generated.items():
                print(f"  [{env_name}] {len(code)} chars generated")

            # ── Phase 2: Run batch (all envs in parallel) ────────
            print(f"\n>> Phase 2: EXECUTE ({episodes_per_round} episodes x {len(envs)} envs, parallel)")
            log_phase(round_num, "execute", envs=envs, episodes=episodes_per_round)
            emit_event_sync(Event(
                type="phase_change", phase="execute",
                round_num=round_num, total_rounds=rounds,
                message=f"Running {episodes_per_round} episodes across {len(envs)} environments...",
            ))
            batch_results = run_batch(env_ports, episodes_per_round, generated_code=generated)

            for env_name, results in batch_results.items():
                if results:
                    avg = sum(r["total_reward"] for r in results) / len(results)
                    best = max(r["total_reward"] for r in results)
                    print(f"  [{env_name}] avg={avg:.1f}, best={best:.1f}, episodes={len(results)}")
                cumulative[env_name].extend(results)

            # ── Phase 3: Learn ───────────────────────────────────
            print(f"\n>> Phase 3: LEARN (feed results back to CEO)")
            learnings = ceo.learn_from_batch(batch_results, total_rounds=rounds)
            print(f"  Memory size: {len(ceo.memory)} entries")

        # ── Summary ──────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  ORCHESTRATION COMPLETE")
        print(f"{'='*60}")
        for env_name, results in cumulative.items():
            if results:
                avg = sum(r["total_reward"] for r in results) / len(results)
                best = max(r["total_reward"] for r in results)
                print(f"  {env_name}: {len(results)} total episodes, avg={avg:.1f}, best={best:.1f}")

        # Show learning trajectory
        if ceo.memory:
            print(f"\nLearning trajectory:")
            for env_name in envs:
                env_mem = [m for m in ceo.memory if m["env"] == env_name]
                rewards = [f"R{m['round']}={m['avg_reward']:.1f}" for m in env_mem]
                print(f"  {env_name}: {' -> '.join(rewards)}")

    finally:
        for proc in server_procs:
            proc.terminate()
        for proc in server_procs:
            proc.wait()
        print("\nAll servers stopped.")


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
