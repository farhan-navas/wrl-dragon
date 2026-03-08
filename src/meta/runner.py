"""Phase 2 meta-training entry point.

Usage (local):
    python -m src.meta.runner

Usage (Modal):
    modal run src/meta/modal_train.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from unsloth import FastLanguageModel

from src.meta.callbacks import RewardAccumulator
from src.meta.config import MetaConfig
from src.meta.events import TrainingEventEmitter
from src.meta.trainer import create_trainer


def load_model(config: MetaConfig):
    """Load model with Unsloth + LoRA."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.model_id,
        max_seq_length=2048,
        load_in_4bit=True,
        fast_inference=False,  # we need training mode
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.model.lora_rank,
        lora_alpha=config.model.lora_alpha,
        target_modules=config.model.lora_target_modules,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    return model, tokenizer


def _start_dashboard_server():
    """Start the API server in a background thread for local dashboard access."""
    import threading
    import uvicorn
    from src.api.server import app

    def _run():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    print("  Dashboard: http://localhost:8000")


def main(
    env_names: list[str] | None = None,
    num_iterations: int = 200,
    output_dir: str = "outputs/meta",
    webhook_url: str | None = None,
    enable_dashboard: bool = True,
):
    config = MetaConfig()
    if env_names:
        config.train.env_names = env_names
    config.train.num_iterations = num_iterations
    config.output_dir = output_dir
    config.webhook_url = webhook_url
    config.enable_dashboard = enable_dashboard

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("WRL-DRAGON Phase 2: RL^2 Meta-Training")
    print(f"  Model: {config.model.model_id}")
    print(f"  Envs:  {config.train.env_names}")
    print(f"  Iterations: {config.train.num_iterations}")
    print(f"  GRPO group size: {config.train.num_generations}")
    print("=" * 60)

    # Load model
    t0 = time.time()
    model, tokenizer = load_model(config)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Count params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {trainable:,} trainable / {total:,} total ({trainable/total*100:.2f}%)")

    # Dashboard integration
    emitter = None
    accumulator = None
    if config.enable_dashboard:
        emitter = TrainingEventEmitter(webhook_url=config.webhook_url)
        accumulator = RewardAccumulator()
        if not config.webhook_url:
            _start_dashboard_server()

    # Create GRPO trainer
    trainer = create_trainer(
        model, tokenizer, config,
        emitter=emitter, accumulator=accumulator,
    )

    # Train
    print("\nStarting GRPO training loop...")
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    print(f"\nTraining complete in {elapsed/60:.1f} minutes")
    print(f"  Final loss: {result.training_loss:.4f}")

    # Save final adapter
    final_path = Path(config.output_dir) / "final_adapter"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"  Saved final adapter to {final_path}")

    # Save training summary
    summary = {
        "model": config.model.model_id,
        "envs": config.train.env_names,
        "iterations": config.train.num_iterations,
        "elapsed_minutes": elapsed / 60,
        "final_loss": result.training_loss,
        "trainable_params": trainable,
        "total_params": total,
    }
    (Path(config.output_dir) / "training_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", default=["CartPole-v1", "LunarLander-v3"])
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", default="outputs/meta")
    parser.add_argument("--webhook", default=None, help="Webhook URL for remote dashboard")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable dashboard")
    args = parser.parse_args()

    main(
        env_names=args.envs,
        num_iterations=args.iterations,
        output_dir=args.output,
        webhook_url=args.webhook,
        enable_dashboard=not args.no_dashboard,
    )
