"""HuggingFace TrainerCallback for dashboard integration + reward accumulator."""
from __future__ import annotations

import threading
import time

from trl import GRPOTrainer
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from src.logging.events import Event
from src.meta.events import TrainingEventEmitter


class RewardAccumulator:
    """Thread-safe accumulator for per-completion reward stats.

    The reward_fn writes stats here; the DashboardCallback reads and flushes.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._env_rewards: dict[str, list[float]] = {}
        self._syntax_scores: list[float] = []
        self._best_code: dict[str, tuple[str, float]] = {}  # env -> (code, reward)
        self._count = 0

    def record(self, env_name: str, total_reward: float, syntax_score: float, code: str):
        with self._lock:
            self._env_rewards.setdefault(env_name, []).append(total_reward)
            self._syntax_scores.append(syntax_score)
            self._count += 1
            # Track best code per env
            best = self._best_code.get(env_name)
            if best is None or total_reward > best[1]:
                self._best_code[env_name] = (code, total_reward)

    def flush(self) -> dict:
        """Return accumulated stats and reset."""
        with self._lock:
            if self._count == 0:
                return {}

            mean_rewards = {
                env: sum(rs) / len(rs) for env, rs in self._env_rewards.items()
            }
            best_rewards = {
                env: best[1] for env, best in self._best_code.items()
            }
            syntax_valid = sum(1 for s in self._syntax_scores if s >= 1.0)
            syntax_rate = syntax_valid / len(self._syntax_scores) if self._syntax_scores else 0.0

            # Find overall best code
            best_code = None
            best_r = float("-inf")
            for env, (code, r) in self._best_code.items():
                if r > best_r:
                    best_code = code
                    best_r = r

            result = {
                "mean_rewards": mean_rewards,
                "best_rewards": best_rewards,
                "syntax_rate": syntax_rate,
                "best_code": best_code,
                "count": self._count,
            }

            # Reset
            self._env_rewards = {}
            self._syntax_scores = []
            self._best_code = {}
            self._count = 0

            return result


class DashboardCallback(TrainerCallback):
    """Emits training events to the dashboard via TrainingEventEmitter."""

    def __init__(
        self,
        emitter: TrainingEventEmitter,
        accumulator: RewardAccumulator,
        model_id: str = "",
        env_names: list[str] | None = None,
        total_iterations: int = 0,
    ):
        self.emitter = emitter
        self.accumulator = accumulator
        self.model_id = model_id
        self.env_names = env_names or []
        self.total_iterations = total_iterations
        self._start_time = 0.0

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self._start_time = time.time()
        self.emitter.emit(Event(
            type="training_started",
            model_id=self.model_id,
            envs=self.env_names,
            iterations=self.total_iterations,
        ))

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        stats = self.accumulator.flush()
        if not stats:
            return

        self.emitter.emit(Event(
            type="training_step",
            iteration=state.global_step,
            loss=logs.get("loss") if logs else None,
            mean_rewards=stats.get("mean_rewards"),
            best_rewards=stats.get("best_rewards"),
            syntax_rate=stats.get("syntax_rate"),
            best_code=stats.get("best_code"),
        ))

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.emitter.emit(Event(
            type="training_checkpoint",
            iteration=state.global_step,
            checkpoint_path=f"{args.output_dir}/checkpoint-{state.global_step}",
        ))

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        elapsed = (time.time() - self._start_time) / 60.0
        self.emitter.emit(Event(
            type="training_completed",
            iterations=state.global_step,
            loss=state.log_history[-1].get("train_loss") if state.log_history else None,
            elapsed_min=round(elapsed, 2),
        ))
