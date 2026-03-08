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
# Prebuilt flash-attn wheel avoids 15+ min compilation and CUDA devel image
FLASH_ATTN_WHEEL = (
    "https://github.com/lesj0610/flash-attention/releases/download/"
    "v2.8.3-cu12-torch2.10-cp312/"
    "flash_attn-2.8.3+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "swig")
    .pip_install("torch>=2.10.0", "packaging", "numpy>=1.26.0", "einops")
    .run_commands(f"pip install '{FLASH_ATTN_WHEEL}' 2>/dev/null || true")
    .pip_install(
        # unsloth + unsloth_zoo runtime deps (installed explicitly since
        # unsloth itself is installed with --no-deps to avoid torch conflicts)
        "trl>=0.18.2,<=0.24.0",
        "transformers>=4.51.3,<=5.2.0",
        "datasets>=3.4.1,<4.4.0",
        "accelerate>=0.34.1",
        "peft>=0.18.0",
        "huggingface_hub[hf_xet]>=0.34.0",
        "hf_transfer",
        "sentencepiece>=0.2.0",
        "protobuf",
        "torchvision",
        "torchao>=0.13.0",
        "triton>=3.0.0",
        "pillow",
        "psutil",
        "tyro",
        "regex",
        "msgspec",
        "cut_cross_entropy",
        "xformers",
        "bitsandbytes>=0.45.5",
        "scipy",
        "safetensors",
        "tokenizers",
        "filelock",
        "typing_extensions",
        "diffusers",
        "sentence-transformers",
        # project deps
        "gymnasium[box2d]>=1.0.0",
        "pydantic>=2.0.0",
        "httpx>=0.27.0",
    )
    .run_commands("pip install --no-deps unsloth unsloth_zoo")
    .env({"HF_HOME": "/vol/hf_cache", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
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

    # Patch torch._grouped_mm to handle mixed dtypes (float32 inputs + bf16 weights)
    # caused by gradient checkpointing recomputation dropping autocast context
    if hasattr(torch, "_grouped_mm"):
        _orig_grouped_mm = torch._grouped_mm
        def _patched_grouped_mm(inputs, weight, **kwargs):
            if inputs.dtype != weight.dtype:
                inputs = inputs.to(weight.dtype)
            return _orig_grouped_mm(inputs, weight, **kwargs)
        torch._grouped_mm = _patched_grouped_mm

    output_dir = "/vol/outputs/meta"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("WRL-DRAGON Phase 2: RL^2 Meta-Training (Modal H200)")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Envs: {envs}")
    print(f"  Iterations: {iterations}")
    print("=" * 60)

    # ---- Load model ----
    model_id = "unsloth/Qwen3-Coder-30B-A3B-Instruct"
    t0 = time.time()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=2048,
        dtype=torch.bfloat16,
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
    # Ensure pad token is set so GRPO pads all completions in a group
    # to the same length — prevents completion_mask shape mismatch
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "right"

    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"  pad_token_id={tokenizer.pad_token_id}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {trainable:,} trainable / {total:,} total ({trainable/total*100:.2f}%)")

    # ---- Environment specs (enriched with strategy hints) ----
    ENV_SPECS = {
        "CartPole-v1": {
            "obs_dim": 4, "num_actions": 2,
            "baseline": 20.0, "solved": 500.0,
            "prompt_hint": (
                "obs[0]=cart_pos [-4.8,4.8], obs[1]=cart_vel, obs[2]=pole_angle [-0.418,0.418] rad, obs[3]=pole_ang_vel. "
                "Action 0=left, 1=right. Keep pole balanced.\n"
                "Heuristic: push right if (angle + angular_velocity) > 0, else left. PD controller on obs[2],obs[3] is near-optimal.\n"
                "Failure: episode ends if angle > 0.418 rad or cart exits [-4.8, 4.8]."
            ),
        },
        "LunarLander-v3": {
            "obs_dim": 8, "num_actions": 4,
            "baseline": -200.0, "solved": 200.0,
            "prompt_hint": (
                "obs[0]=x, obs[1]=y, obs[2]=vx, obs[3]=vy, obs[4]=angle, obs[5]=ang_vel, "
                "obs[6]=left_leg_contact, obs[7]=right_leg_contact. "
                "Action 0=noop, 1=left, 2=main, 3=right. Land softly.\n"
                "Heuristic: fire main engine when vy < -0.1. Use left/right to correct angle and drift. Stop when legs touch.\n"
                "Failure: crash (high velocity) gives -100. Fuel costs reward."
            ),
        },
        "Acrobot-v1": {
            "obs_dim": 6, "num_actions": 3,
            "baseline": -500.0, "solved": -100.0,
            "prompt_hint": (
                "obs=[cos(t1), sin(t1), cos(t2), sin(t2), dt1, dt2]. dt1 in [-12.57,12.57], dt2 in [-28.27,28.27]. "
                "Action 0=neg, 1=none, 2=pos torque. Swing tip above line.\n"
                "Heuristic: apply torque in direction of angular velocity (pump like a swing). When dt1 > 0, positive torque.\n"
                "Failure: max 500 steps, reward=-1/step. Random policy ≈ -500."
            ),
        },
        "MountainCar-v0": {
            "obs_dim": 2, "num_actions": 3,
            "baseline": -200.0, "solved": -110.0,
            "prompt_hint": (
                "obs[0]=position [-1.2,0.6], obs[1]=velocity [-0.07,0.07]. "
                "Action 0=left, 1=none, 2=right. Reach flag at pos>=0.5.\n"
                "Heuristic: push in direction of velocity (action=2 if vel>0, else 0). This builds momentum via resonance.\n"
                "Failure: max 200 steps. Can't climb in one push — must rock back and forth."
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
                messages, tokenize=False, add_generation_prompt=True,
            )
            # Close Qwen3's thinking block in the prompt so completions
            # start clean. Without this, the model emits <think></think>
            # (6 tokens) inconsistently across completions, causing
            # Unsloth's completion_mask to mismatch in compute_loss.
            prompt += "</think>\n\n"
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

    def strip_thinking(text: str) -> str:
        """Remove <think>...</think> blocks from model output."""
        return re.sub(r"<think>[\s\S]*?</think>\s*", "", text).strip()

    def syntax_reward(code: str, num_actions: int, obs_dim: int = 4) -> float:
        """7-tier syntax scoring for finer gradient signal."""
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
            return 0.1
        if "select_action" not in ns:
            return 0.2
        if not callable(ns["select_action"]):
            return 0.4
        try:
            result = int(ns["select_action"]([0.0] * obs_dim))
            return 1.0 if 0 <= result < num_actions else 0.8
        except Exception:
            return 0.6

    def env_reward(code: str, env_name: str, num_episodes: int = 5) -> tuple[float, int]:
        """Run policy in gym, return (mean_reward, crashed_count)."""
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
                print(json.dumps({{"rewards": [], "crashed": {num_episodes}}}))
                exit()
            if "select_action" not in ns:
                print(json.dumps({{"rewards": [], "crashed": {num_episodes}}}))
                exit()
            fn = ns["select_action"]
            env = gym.make({repr(env_name)})
            n = int(env.action_space.n) if hasattr(env.action_space, "n") else 2
            results = []
            crashed = 0
            for _ in range({num_episodes}):
                obs, _ = env.reset()
                total = 0.0
                ep_crashed = False
                for _ in range(500):
                    try:
                        a = int(fn(obs.tolist()))
                        a = a if 0 <= a < n else random.randint(0, n-1)
                    except Exception:
                        a = random.randint(0, n-1)
                        ep_crashed = True
                    obs, r, term, trunc, _ = env.step(a)
                    total += r
                    if term or trunc:
                        break
                results.append(total)
                if ep_crashed:
                    crashed += 1
            env.close()
            print(json.dumps({{"rewards": results, "crashed": crashed}}))
        """)
        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, timeout=120,
            )
            data = json.loads(result.stdout) if result.stdout.strip() else {}
            rewards = data.get("rewards", [])
            crashed = data.get("crashed", 0)
        except Exception:
            rewards = []
            crashed = num_episodes
        mean = sum(rewards) / len(rewards) if rewards else 0.0
        return mean, crashed

    # Curriculum + reward config
    ALPHA_START, ALPHA_END = 0.5, 0.15   # syntax weight: high early, low late
    BETA_START, BETA_END = 0.5, 0.85     # env weight: low early, high late
    ENV_EVAL_GATE = 0.4                   # min syntax score to try env rollouts
    CRASH_PENALTY = -0.1                  # per crashed episode
    total_dataset_steps = len(dataset)
    step_counter = [0]

    def reward_fn(completions: list[str], **kwargs) -> list[float]:
        prompts = kwargs.get("prompts", [""] * len(completions))
        rewards = []

        progress = min(step_counter[0] / max(total_dataset_steps, 1), 1.0)
        step_counter[0] += 1

        alpha = ALPHA_START + progress * (ALPHA_END - ALPHA_START)
        beta = BETA_START + progress * (BETA_END - BETA_START)

        for i, raw_code in enumerate(completions):
            code = strip_thinking(raw_code)
            prompt = prompts[i] if i < len(prompts) else ""
            detected_env = "CartPole-v1"
            for en in ENV_SPECS:
                if en in prompt:
                    detected_env = en
                    break
            spec = ENV_SPECS[detected_env]

            r_syn = syntax_reward(code, spec["num_actions"], obs_dim=spec.get("obs_dim", 4))

            crashed = 0
            raw = 0.0
            if r_syn >= ENV_EVAL_GATE:
                raw, crashed = env_reward(code, detected_env)
                baseline, solved = spec["baseline"], spec["solved"]
                r_env = max(0.0, min(1.0, (raw - baseline) / (solved - baseline)))
            else:
                r_env = 0.0

            total = alpha * r_syn + beta * r_env
            if crashed > 0:
                total += CRASH_PENALTY * crashed
            total = max(total, 0.0)

            rewards.append(total)
            if accumulator is not None:
                accumulator.record(detected_env, total, r_syn, code)
            print(f"  [{detected_env}] syn={r_syn:.1f} raw={raw:.1f} env={r_env:.2f} R={total:.3f} crash={crashed} prog={progress:.2f}")

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
        num_generations=8,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        num_train_epochs=1,
        max_prompt_length=768,
        max_completion_length=1024,
        temperature=1.0,
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
        args=grpo_config,
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
