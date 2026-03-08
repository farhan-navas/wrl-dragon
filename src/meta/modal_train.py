"""Modal deployment for RL^2 meta-training on H200 GPU.

Usage:
    modal run src/meta/modal_train.py
    modal run src/meta/modal_train.py --envs CartPole-v1 --iterations 50

The training runs on a single H200 (141GB VRAM) with:
- Qwen3-Coder-30B-A3B-Instruct (MoE: 30B total, 3B active)
- 16-bit LoRA via Unsloth (~642M trainable params)
- GRPO from TRL for policy gradient optimization
- Gym rollouts run in the same container (no separate server needed)
"""
import modal

app = modal.App("wrl-dragon-meta-train")

# Persistent volume for checkpoints + model cache
vol = modal.Volume.from_name("wrl-dragon-training", create_if_missing=True)

# Container image with all dependencies
# Use CUDA-devel base so nvcc is available for flash-attn compilation
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "swig")
    .pip_install("torch>=2.6.0", "packaging", "ninja", "numpy>=1.26.0")
    .pip_install(
        "flash-attn>=2.6.3",
        extra_options="--no-build-isolation",
    )
    .pip_install(
        "unsloth[cu124-ampere-torch260]",
        "trl>=0.17.0",
        "datasets>=3.0.0",
        "gymnasium[box2d]>=1.0.0",
        "pydantic>=2.0.0",
        "httpx>=0.27.0",
    )
    .env({"HF_HOME": "/vol/hf_cache"})
)


@app.function(
    image=image,
    gpu="H100",
    volumes={"/vol": vol},
    timeout=4 * 3600,  # 4 hour max
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train(
    envs: list[str] = ["CartPole-v1", "LunarLander-v3"],
    iterations: int = 200,
    webhook: str | None = None,
):
    """Run the full GRPO meta-training loop on an H200."""
    import json
    import time
    from pathlib import Path

    import torch
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    output_dir = "/vol/outputs/meta"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("WRL-DRAGON Phase 2: RL^2 Meta-Training (Modal H200)")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"  Envs: {envs}")
    print(f"  Iterations: {iterations}")
    print("=" * 60)

    # ---- Load model ----
    model_id = "unsloth/Qwen3-Coder-30B-A3B-Instruct"
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=2048,
        load_in_4bit=True,
        fast_inference=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {trainable:,} trainable / {total:,} total ({trainable/total*100:.2f}%)")

    # ---- Environment specs ----
    ENV_SPECS = {
        "CartPole-v1": {
            "num_actions": 2,
            "baseline": 20.0, "solved": 500.0,
            "prompt_hint": (
                "obs[0]=cart_pos, obs[1]=cart_vel, obs[2]=pole_angle, obs[3]=pole_ang_vel. "
                "Action 0=left, 1=right. Keep pole balanced."
            ),
        },
        "LunarLander-v3": {
            "num_actions": 4,
            "baseline": -200.0, "solved": 200.0,
            "prompt_hint": (
                "obs[0]=x, obs[1]=y, obs[2]=vx, obs[3]=vy, obs[4]=angle, obs[5]=ang_vel, "
                "obs[6]=left_leg, obs[7]=right_leg. "
                "Action 0=noop, 1=left, 2=main, 3=right. Land softly."
            ),
        },
        "Acrobot-v1": {
            "num_actions": 3,
            "baseline": -500.0, "solved": -100.0,
            "prompt_hint": (
                "obs=[cos(t1), sin(t1), cos(t2), sin(t2), dt1, dt2]. "
                "Action 0=neg, 1=none, 2=pos torque. Swing tip above line."
            ),
        },
        "MountainCar-v0": {
            "num_actions": 3,
            "baseline": -200.0, "solved": -110.0,
            "prompt_hint": (
                "obs[0]=position, obs[1]=velocity. "
                "Action 0=left, 1=none, 2=right. Reach flag at pos>=0.5."
            ),
        },
    }

    SYSTEM_PROMPT = (
        "You are an expert RL engineer. Write Python policy functions for gym environments. "
        "Output ONLY valid Python code, no markdown, no explanation."
    )

    # ---- Build prompts dataset ----
    import random
    rows = []
    for _ in range(iterations):
        for env_name in envs:
            spec = ENV_SPECS.get(env_name, ENV_SPECS["CartPole-v1"])
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Environment: {env_name}\n"
                    f"Details: {spec['prompt_hint']}\n\n"
                    f"Write `def select_action(obs: list[float]) -> int:` "
                    f"that returns the best action. Self-contained, may use math/random/numpy."
                )},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            rows.append({"prompt": prompt, "env_name": env_name})

    random.shuffle(rows)
    dataset = Dataset.from_list(rows)
    print(f"Dataset: {len(dataset)} prompts")

    # ---- Reward functions ----
    import ast
    import re
    import subprocess
    import sys
    import textwrap

    def syntax_reward(code: str, num_actions: int) -> float:
        clean = re.sub(r"^```(?:python)?\s*\n?", "", code.strip())
        clean = re.sub(r"\n?```\s*$", "", clean.strip())
        if "def select_action" in clean:
            clean = clean[clean.index("def select_action"):]
        try:
            ast.parse(clean)
        except SyntaxError:
            return 0.0
        ns = {}
        try:
            exec(clean, {"__builtins__": __builtins__}, ns)
        except Exception:
            return 0.0
        if "select_action" not in ns or not callable(ns["select_action"]):
            return 0.3
        try:
            result = int(ns["select_action"]([0.0] * 8))
            return 1.0 if 0 <= result < num_actions else 0.7
        except Exception:
            return 0.7

    def env_reward(code: str, env_name: str, num_episodes: int = 5) -> float:
        """Run policy in gym and return mean reward."""
        clean = re.sub(r"^```(?:python)?\s*\n?", "", code.strip())
        clean = re.sub(r"\n?```\s*$", "", clean.strip())
        if "def select_action" in clean:
            clean = clean[clean.index("def select_action"):]

        script = textwrap.dedent(f"""\
            import json, random, gymnasium as gym
            code = {repr(clean)}
            ns = {{}}
            try:
                exec(code, {{"__builtins__": __builtins__}}, ns)
            except Exception:
                print(json.dumps([]))
                exit()
            if "select_action" not in ns:
                print(json.dumps([]))
                exit()
            fn = ns["select_action"]
            env = gym.make({repr(env_name)})
            n = int(env.action_space.n) if hasattr(env.action_space, "n") else 2
            results = []
            for _ in range({num_episodes}):
                obs, _ = env.reset()
                total = 0.0
                for _ in range(500):
                    try:
                        a = int(fn(obs.tolist()))
                        a = a if 0 <= a < n else random.randint(0, n-1)
                    except Exception:
                        a = random.randint(0, n-1)
                    obs, r, term, trunc, _ = env.step(a)
                    total += r
                    if term or trunc:
                        break
                results.append(total)
            env.close()
            print(json.dumps(results))
        """)
        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, timeout=120,
            )
            rewards = json.loads(result.stdout) if result.stdout.strip() else []
        except Exception:
            rewards = []
        return sum(rewards) / len(rewards) if rewards else 0.0

    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        prompts = kwargs.get("prompts", [""] * len(completions))
        rewards = []
        for i, code in enumerate(completions):
            prompt = prompts[i] if i < len(prompts) else ""
            # Detect env
            detected_env = "CartPole-v1"
            for en in ENV_SPECS:
                if en in prompt:
                    detected_env = en
                    break
            spec = ENV_SPECS[detected_env]

            r_syn = syntax_reward(code, spec["num_actions"])
            if r_syn >= 1.0:
                raw = env_reward(code, detected_env)
                baseline, solved = spec["baseline"], spec["solved"]
                r_env = max(0.0, min(1.0, (raw - baseline) / (solved - baseline)))
            else:
                r_env = 0.0
                raw = 0.0

            total = 0.3 * r_syn + 0.7 * r_env
            rewards.append(total)
            if accumulator is not None:
                accumulator.record(detected_env, total, r_syn, code)
            print(f"  [{detected_env}] syn={r_syn:.1f} raw={raw:.1f} env={r_env:.2f} R={total:.3f}")

        return rewards

    # ---- Dashboard callback ----
    dashboard_cb = None
    accumulator = None
    if webhook:
        import httpx

        class _SimpleEmitter:
            """Lightweight emitter for Modal (avoids importing src.logging on remote)."""
            def __init__(self, url):
                self.url = url.rstrip("/") + "/api/events"
            def emit(self, event):
                try:
                    data = {}
                    for key, value in event.items():
                        if value is not None:
                            data[key] = value
                    httpx.post(self.url, json=data, timeout=5.0)
                except Exception:
                    pass

        class _Accumulator:
            """Lightweight reward accumulator for Modal."""
            def __init__(self):
                self._env_rewards = {}
                self._syntax_scores = []
                self._best_code = {}
                self._count = 0
            def record(self, env_name, total_reward, syntax_score, code):
                self._env_rewards.setdefault(env_name, []).append(total_reward)
                self._syntax_scores.append(syntax_score)
                best = self._best_code.get(env_name)
                if best is None or total_reward > best[1]:
                    self._best_code[env_name] = (code, total_reward)
                self._count += 1
            def flush(self):
                if self._count == 0:
                    return {}
                mean_rewards = {e: sum(rs)/len(rs) for e, rs in self._env_rewards.items()}
                best_rewards = {e: b[1] for e, b in self._best_code.items()}
                syntax_valid = sum(1 for s in self._syntax_scores if s >= 1.0)
                syntax_rate = syntax_valid / len(self._syntax_scores) if self._syntax_scores else 0.0
                best_code = None
                best_r = float("-inf")
                for e, (c, r) in self._best_code.items():
                    if r > best_r:
                        best_code, best_r = c, r
                result = {"mean_rewards": mean_rewards, "best_rewards": best_rewards,
                          "syntax_rate": syntax_rate, "best_code": best_code, "count": self._count}
                self._env_rewards, self._syntax_scores, self._best_code, self._count = {}, [], {}, 0
                return result

        from transformers import TrainerCallback, TrainerControl, TrainerState
        from transformers.training_args import TrainingArguments as _TA

        emitter = _SimpleEmitter(webhook)
        accumulator = _Accumulator()

        class _DashboardCB(TrainerCallback):
            def __init__(self):
                self._start = 0.0
            def on_train_begin(self, args, state, control, **kw):
                self._start = time.time()
                emitter.emit({"type": "training_started", "ts": time.time(),
                              "model_id": model_id, "envs": envs, "iterations": iterations})
            def on_log(self, args, state, control, logs=None, **kw):
                stats = accumulator.flush()
                if not stats:
                    return
                emitter.emit({"type": "training_step", "ts": time.time(),
                              "iteration": state.global_step,
                              "loss": logs.get("loss") if logs else None,
                              "mean_rewards": stats.get("mean_rewards"),
                              "best_rewards": stats.get("best_rewards"),
                              "syntax_rate": stats.get("syntax_rate"),
                              "best_code": stats.get("best_code")})
            def on_save(self, args, state, control, **kw):
                emitter.emit({"type": "training_checkpoint", "ts": time.time(),
                              "iteration": state.global_step,
                              "checkpoint_path": f"{args.output_dir}/checkpoint-{state.global_step}"})
            def on_train_end(self, args, state, control, **kw):
                elapsed = (time.time() - self._start) / 60.0
                emitter.emit({"type": "training_completed", "ts": time.time(),
                              "iterations": state.global_step,
                              "loss": state.log_history[-1].get("train_loss") if state.log_history else None,
                              "elapsed_min": round(elapsed, 2)})

        dashboard_cb = _DashboardCB()

    # ---- GRPO Training ----
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_generations=4,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        num_train_epochs=1,
        max_completion_length=512,
        temperature=0.8,
        logging_steps=1,
        save_steps=25,
        save_total_limit=5,
        bf16=True,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        config=grpo_config,
        train_dataset=dataset,
        callbacks=[dashboard_cb] if dashboard_cb else None,
    )

    print("\nStarting GRPO training...")
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed/60:.1f} min | Loss: {result.training_loss:.4f}")

    # Save final adapter
    final_path = f"{output_dir}/final_adapter"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    summary = {
        "model": model_id,
        "envs": envs,
        "iterations": iterations,
        "elapsed_min": elapsed / 60,
        "loss": result.training_loss,
        "trainable": trainable,
        "total": total,
    }
    Path(f"{output_dir}/summary.json").write_text(json.dumps(summary, indent=2))

    # Persist to volume
    vol.commit()
    print(f"Saved to /vol/outputs/meta/")
    return summary


@app.local_entrypoint()
def main(
    envs: str = "CartPole-v1,LunarLander-v3",
    iterations: int = 200,
    webhook: str = "",
):
    env_list = [e.strip() for e in envs.split(",")]
    result = train.remote(
        envs=env_list,
        iterations=iterations,
        webhook=webhook or None,
    )
    print(f"\nTraining complete: {result}")
