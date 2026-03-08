"""GRPO trainer for the RL^2 meta-training loop.

Uses TRL's GRPOTrainer with Unsloth for efficient LoRA training.
The reward function evaluates generated code on:
  1. Syntax validity (fast, local)
  2. Gym environment performance (runs rollouts)
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from src.meta.callbacks import DashboardCallback, RewardAccumulator
from src.meta.config import MetaConfig, RewardConfig
from src.meta.events import TrainingEventEmitter
from src.meta.prompt import ENV_SPECS, build_prompt, format_prompt
from src.meta.rewards import (
    compute_env_reward,
    compute_syntax_reward,
    compute_total_reward,
    normalize_env_reward,
)


def build_training_dataset(
    tokenizer,
    env_names: list[str],
    num_iterations: int = 200,
    best_policies: dict[str, tuple[str, float]] | None = None,
) -> Dataset:
    """Build a dataset of prompts for GRPO training.

    Each row is a prompt for one env. We repeat env prompts across iterations
    so GRPO sees many (prompt, completion) pairs.
    """
    best_policies = best_policies or {}
    rows = []

    for _ in range(num_iterations):
        for env_name in env_names:
            best_code, best_reward = best_policies.get(env_name, (None, None))
            prompt_str = format_prompt(
                tokenizer, env_name,
                best_code=best_code,
                best_reward=best_reward,
            )
            rows.append({
                "prompt": prompt_str,
                "env_name": env_name,
            })

    # Shuffle so we don't always train on the same env order
    random.shuffle(rows)
    return Dataset.from_list(rows)


def make_reward_fn(
    config: RewardConfig,
    total_steps: int = 1,
    accumulator: RewardAccumulator | None = None,
):
    """Create the GRPO reward function with curriculum scheduling.

    Args:
        config: Reward configuration (weights, baselines, penalties).
        total_steps: Total training steps for curriculum progress calculation.
        accumulator: Optional dashboard accumulator.
    """
    # Mutable counter for curriculum progress tracking
    step_counter = [0]

    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        prompts = kwargs.get("prompts", [""] * len(completions))
        rewards = []

        # Curriculum progress: 0.0 at start → 1.0 at end
        progress = min(step_counter[0] / max(total_steps, 1), 1.0)
        step_counter[0] += 1

        for i, code in enumerate(completions):
            prompt = prompts[i] if i < len(prompts) else ""

            # Detect env from prompt content
            env_name = _detect_env(prompt)
            spec = ENV_SPECS.get(env_name, {})
            num_actions = spec.get("num_actions", 2)
            obs_dim = spec.get("obs_dim", 4)

            # R_syntax (fast) — uses correct obs_dim per env
            r_syntax = compute_syntax_reward(code, num_actions, obs_dim=obs_dim)

            # R_env (slow — gate lowered to 0.4 for partial credit)
            crashed = 0
            raw_reward = 0.0
            if r_syntax >= config.env_eval_gate:
                raw_reward, episodes, crashed = compute_env_reward(
                    code, env_name,
                    num_episodes=5,
                    max_steps=500,
                )
                r_env = normalize_env_reward(raw_reward, env_name, config)
            else:
                r_env = 0.0

            total = compute_total_reward(
                r_syntax, r_env, config,
                progress=progress,
                crashed_episodes=crashed,
            )
            rewards.append(total)

            if accumulator is not None:
                accumulator.record(env_name, total, r_syntax, code)

            print(
                f"  [{env_name}] syn={r_syntax:.1f} raw={raw_reward:.1f} "
                f"env={r_env:.2f} R={total:.3f} crash={crashed} prog={progress:.2f}"
            )

        return rewards

    return reward_fn


def _detect_env(prompt: str) -> str:
    """Extract env name from prompt text."""
    for env_name in ENV_SPECS:
        if env_name in prompt:
            return env_name
    return "CartPole-v1"


def create_trainer(
    model,
    tokenizer,
    config: MetaConfig,
    best_policies: dict[str, tuple[str, float]] | None = None,
    emitter: TrainingEventEmitter | None = None,
    accumulator: RewardAccumulator | None = None,
) -> GRPOTrainer:
    """Create a fully configured GRPOTrainer."""

    dataset = build_training_dataset(
        tokenizer,
        config.train.env_names,
        config.train.num_iterations,
        best_policies,
    )

    # Total steps for curriculum scheduling
    total_steps = len(dataset) // max(config.train.per_device_train_batch_size, 1)
    reward_fn = make_reward_fn(config.reward, total_steps=total_steps, accumulator=accumulator)

    grpo_config = GRPOConfig(
        output_dir=config.output_dir,

        # GRPO-specific
        num_generations=config.train.num_generations,

        # Training
        learning_rate=config.train.learning_rate,
        per_device_train_batch_size=config.train.per_device_train_batch_size,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        max_grad_norm=config.train.grad_clip,
        num_train_epochs=1,

        # Generation
        max_completion_length=config.model.max_tokens,
        temperature=config.model.temperature,

        # Logging & saving
        logging_steps=config.train.log_every,
        save_steps=config.train.checkpoint_every,
        save_total_limit=5,

        # Misc
        bf16=True,
        report_to="none",
    )

    callbacks = []
    if emitter is not None and accumulator is not None:
        callbacks.append(DashboardCallback(
            emitter=emitter,
            accumulator=accumulator,
            model_id=config.model.model_id,
            env_names=config.train.env_names,
            total_iterations=config.train.num_iterations,
        ))

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        config=grpo_config,
        train_dataset=dataset,
        callbacks=callbacks if callbacks else None,
    )

    return trainer
