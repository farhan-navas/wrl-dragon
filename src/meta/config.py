from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Parent code-generating LLM config."""

    model_id: str = "unsloth/Qwen3-Coder-30B-A3B-Instruct"
    max_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95

    # LoRA (handled by Unsloth)
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class RewardConfig:
    """Weights and normalization for the two reward signals."""

    alpha_syntax: float = 0.3
    beta_env: float = 0.7

    # (random_baseline, solved_threshold) per env
    env_baselines: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "CartPole-v1": (20.0, 500.0),
        "LunarLander-v3": (-200.0, 200.0),
        "Acrobot-v1": (-500.0, -100.0),
        "MountainCar-v0": (-200.0, -110.0),
    })


@dataclass
class TrainConfig:
    """GRPO training hyperparameters."""

    num_iterations: int = 200
    num_generations: int = 4  # G completions per prompt (GRPO group size)
    episodes_per_eval: int = 5
    max_steps_per_episode: int = 500

    learning_rate: float = 5e-6
    grad_clip: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4

    checkpoint_every: int = 25
    log_every: int = 1

    env_names: list[str] = field(default_factory=lambda: [
        "CartPole-v1",
        "LunarLander-v3",
    ])

    # Gym server ports (one per env, started by runner)
    env_server_base_port: int = 8001


@dataclass
class MetaConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "outputs/meta"
    checkpoint_dir: str = "outputs/checkpoints"
    webhook_url: str | None = None
    enable_dashboard: bool = True
