"""Modal deployment for RL^2 meta-training on H100 GPU.

Usage:
    modal run src/meta/modal_train.py
    modal run src/meta/modal_train.py --envs CartPole-v1 --iterations 50
    modal run src/meta/modal_train.py --gym-space-url https://USER-wrl-dragon-gym.hf.space

Runs on a single H100 (80GB VRAM) with:
- Qwen3-Coder-30B-A3B-Instruct (MoE: 30B total, 3B active) in bf16
- LoRA via PEFT (no quantization, no Unsloth)
- GRPO from TRL for policy gradient optimization
- Gym rollouts via HF Space (fallback: local subprocess)
"""
import modal

app = modal.App("wrl-dragon-meta-train")

# Persistent volume for checkpoints + model cache
vol = modal.Volume.from_name("wrl-dragon-training", create_if_missing=True)

# Prebuilt flash-attn wheel avoids 15+ min compilation
FLASH_ATTN_WHEEL = (
    "https://github.com/lesj0610/flash-attention/releases/download/"
    "v2.8.3-cu12-torch2.10-cp312/"
    "flash_attn-2.8.3+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "swig")
    .pip_install("torch>=2.10.0", "packaging", "numpy>=1.26.0")
    .run_commands(f"pip install '{FLASH_ATTN_WHEEL}' 2>/dev/null || true")
    .pip_install(
        "trl>=0.18.2,<=0.24.0",
        "transformers>=4.51.3,<=5.2.0",
        "datasets>=3.4.1,<4.4.0",
        "accelerate>=0.34.1",
        "peft>=0.18.0",
        "huggingface_hub[hf_xet]>=0.34.0",
        "hf_transfer",
        "sentencepiece>=0.2.0",
        "protobuf",
        "safetensors",
        "tokenizers",
        # project deps
        "gymnasium[box2d]>=1.0.0",
        "pydantic>=2.0.0",
        "httpx>=0.27.0",
    )
    .env({"HF_HOME": "/vol/hf_cache", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    image=image,
    gpu="H100",
    volumes={"/vol": vol},
    timeout=4 * 3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train(
    envs: list[str] = ["CartPole-v1", "LunarLander-v3"],
    iterations: int = 200,
    webhook: str | None = None,
    gym_space_url: str | None = None,
):
    """Run the full GRPO meta-training loop on an H100."""
    import json
    import time
    from pathlib import Path

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    output_dir = "/vol/outputs/meta"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("WRL-DRAGON: RL^2 Meta-Training (Modal H100)")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Envs: {envs}")
    print(f"  Iterations: {iterations}")
    print(f"  Gym Space: {gym_space_url or 'disabled (subprocess fallback)'}")
    print("=" * 60)

    # ---- Load model (bf16, no quantization) ----
    model_id = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # LoRA via PEFT
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.warnings_issued = {}
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"  pad_token_id={tokenizer.pad_token_id}")
    model.print_trainable_parameters()

    # ---- Environment specs ----
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
                "Failure: max 500 steps, reward=-1/step. Random policy ~ -500."
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

    # ---- Reward Aggregator (percentile + improvement + self-retrieval) ----
    import ast
    import random
    import re
    import subprocess
    import sys
    import textwrap
    from collections import deque

    class RewardAggregator:
        """Per-env percentile ranking with improvement tracking and best-code memory.

        Replaces fixed baseline/solved normalization. Auto-normalizes across envs
        with different reward scales by ranking new scores against a rolling buffer.
        """

        def __init__(self, buffer_size: int = 200):
            self.buffers: dict[str, deque] = {}
            self.best_reward: dict[str, float] = {}
            self.best_code: dict[str, str] = {}
            self.buffer_size = buffer_size

        def score(self, env_name: str, raw_reward: float) -> float:
            """Score a raw reward as percentile rank in the env's history.

            Returns value in [0.0, 1.2]:
              - percentile component: [0, 1] based on rank in buffer
              - improvement bonus: +0.2 if this beats the env's personal best
            """
            buf = self.buffers.setdefault(env_name, deque(maxlen=self.buffer_size))

            # Percentile rank (auto-normalizes across envs)
            if len(buf) > 0:
                percentile = sum(1 for r in buf if raw_reward > r) / len(buf)
            else:
                percentile = 0.5  # first sample gets neutral score

            # Improvement bonus (beats personal best)
            prev_best = self.best_reward.get(env_name, float("-inf"))
            improvement = 0.2 if raw_reward > prev_best else 0.0
            if raw_reward > prev_best:
                self.best_reward[env_name] = raw_reward

            buf.append(raw_reward)
            return percentile + improvement

        def update_best_code(self, env_name: str, code: str, raw_reward: float):
            """Track best-performing code per env for self-retrieval."""
            prev = self.best_reward.get(env_name, float("-inf"))
            if raw_reward >= prev:
                self.best_code[env_name] = code

        def get_best_code(self, env_name: str) -> str | None:
            return self.best_code.get(env_name)

        def summary(self) -> dict:
            return {
                env: {
                    "best": self.best_reward.get(env, 0),
                    "buffer_size": len(self.buffers.get(env, [])),
                    "median": sorted(self.buffers.get(env, [0]))[len(self.buffers.get(env, [0])) // 2],
                }
                for env in self.buffers
            }

    aggregator = RewardAggregator(buffer_size=200)

    # ---- Build prompts with self-retrieval ----
    def build_round_dataset(round_num: int, prompts_per_env: int) -> Dataset:
        """Build a dataset for one training round.

        Round 0: basic prompts (no prior code).
        Round 1+: injects best-so-far code per env as context for improvement.
        """
        rows = []
        for _ in range(prompts_per_env):
            for env_name in envs:
                spec = ENV_SPECS.get(env_name, ENV_SPECS["CartPole-v1"])
                best_code = aggregator.get_best_code(env_name) if round_num > 0 else None

                user_content = (
                    f"Environment: {env_name}\n"
                    f"Details: {spec['prompt_hint']}\n\n"
                )
                if best_code:
                    best_r = aggregator.best_reward.get(env_name, 0)
                    user_content += (
                        f"A previous policy achieved avg_reward={best_r:.1f}:\n"
                        f"```python\n{best_code}\n```\n\n"
                        f"Write an improved version. "
                    )
                user_content += (
                    f"Write `def select_action(obs: list[float]) -> int:` "
                    f"that returns the best action. Self-contained, may use math/random/numpy."
                )

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                rows.append({"prompt": prompt, "env_name": env_name})

        random.shuffle(rows)
        return Dataset.from_list(rows)

    # ---- Reward helpers ----
    def strip_thinking(text: str) -> str:
        """Remove <think>...</think> blocks from model output.
        Also handles unclosed <think> tags (model ran out of tokens mid-thought).
        """
        text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text)
        text = re.sub(r"<think>[\s\S]*$", "", text)
        return text.strip()

    def clean_code(raw: str) -> str:
        """Extract select_action function from raw model output."""
        code = strip_thinking(raw)
        code = re.sub(r"^```(?:python)?\s*\n?", "", code.strip())
        code = re.sub(r"\n?```\s*$", "", code.strip())
        if "def select_action" in code:
            code = code[code.index("def select_action"):]
        return code

    def syntax_reward(code: str, num_actions: int, obs_dim: int = 4) -> float:
        """7-tier syntax scoring for finer gradient signal."""
        try:
            ast.parse(code)
        except SyntaxError:
            return 0.0
        ns = {}
        try:
            exec(code, {"__builtins__": __builtins__}, ns)
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
        """Run policy in gym. Try HF Space first, fall back to subprocess."""
        # ---- Try HF Space (OpenEnv protocol: /reset + /step) ----
        if gym_space_url:
            try:
                import httpx
                ns = {}
                exec(code, {"__builtins__": __builtins__}, ns)
                fn = ns.get("select_action")
                if fn is None or not callable(fn):
                    raise ValueError("No select_action in code")

                spec = ENV_SPECS.get(env_name, ENV_SPECS["CartPole-v1"])
                n_actions = spec["num_actions"]
                base = gym_space_url.rstrip("/")
                client = httpx.Client(timeout=30.0)
                rewards_list = []
                crashed = 0

                for _ in range(num_episodes):
                    resp = client.post(f"{base}/reset", json={})
                    if resp.status_code != 200:
                        raise RuntimeError(f"reset failed: {resp.status_code}")
                    obs_data = resp.json()
                    obs = obs_data.get("observation", {}).get("obs", [])
                    total = 0.0
                    ep_crashed = False

                    for _ in range(500):
                        try:
                            a = int(fn(obs))
                            a = a if 0 <= a < n_actions else random.randint(0, n_actions - 1)
                        except Exception:
                            a = random.randint(0, n_actions - 1)
                            ep_crashed = True

                        resp = client.post(f"{base}/step", json={"action": {"value": a}})
                        if resp.status_code != 200:
                            break
                        step_data = resp.json()
                        obs_info = step_data.get("observation", {})
                        r = obs_info.get("reward", 0.0)
                        done = obs_info.get("done", False)
                        total += r
                        obs = obs_info.get("obs", [])
                        if done:
                            break

                    rewards_list.append(total)
                    if ep_crashed:
                        crashed += 1

                client.close()
                mean = sum(rewards_list) / len(rewards_list) if rewards_list else 0.0
                return mean, crashed
            except Exception as e:
                print(f"  HF Space rollout failed ({e}), falling back to subprocess")

        # ---- Fallback: local subprocess ----
        script = textwrap.dedent(f"""\
            import json, random, gymnasium as gym
            code = {repr(code)}
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

    # ---- Reward function (percentile + improvement) ----
    ENV_EVAL_GATE = 0.4
    CRASH_PENALTY = -0.1

    def reward_fn(completions, **kwargs) -> list[float]:
        prompts = kwargs.get("prompts", [""] * len(completions))
        rewards = []

        for i, completion in enumerate(completions):
            # Handle both string and conversation-dict formats
            if isinstance(completion, list):
                raw_text = completion[-1]["content"] if completion else ""
            elif isinstance(completion, dict):
                raw_text = completion.get("content", str(completion))
            else:
                raw_text = str(completion)

            code = clean_code(raw_text)

            prompt_raw = prompts[i] if i < len(prompts) else ""
            if isinstance(prompt_raw, list):
                prompt = " ".join(m.get("content", "") for m in prompt_raw)
            elif isinstance(prompt_raw, dict):
                prompt = prompt_raw.get("content", str(prompt_raw))
            else:
                prompt = str(prompt_raw)

            detected_env = "CartPole-v1"
            for en in ENV_SPECS:
                if en in prompt:
                    detected_env = en
                    break
            spec = ENV_SPECS[detected_env]

            r_syn = syntax_reward(code, spec["num_actions"], obs_dim=spec.get("obs_dim", 4))

            # Diagnostic: log why completions fail
            if r_syn == 0.0:
                has_think = "<think>" in raw_text
                closed = "</think>" in raw_text
                has_fn = "def select_action" in code
                print(f"    DIAG [{detected_env}] syn=0 | len={len(raw_text)} "
                      f"think={has_think} closed={closed} "
                      f"has_fn={has_fn} | code[:120]={repr(code[:120])}")

            crashed = 0
            raw = 0.0
            r_perc = 0.0
            if r_syn >= ENV_EVAL_GATE:
                raw, crashed = env_reward(code, detected_env)
                r_perc = aggregator.score(detected_env, raw)
                aggregator.update_best_code(detected_env, code, raw)

            # Reward = syntax (gates entry) + percentile (auto-normalized cross-env)
            # Syntax weight: 0.3 (just needs to pass), env percentile: 0.7 (drives learning)
            total = 0.3 * r_syn + 0.7 * r_perc
            if crashed > 0:
                total += CRASH_PENALTY * crashed
            total = max(total, 0.0)

            rewards.append(total)
            if accumulator is not None:
                accumulator.record(detected_env, total, r_syn, code)
            is_best = "NEW_BEST" if raw >= aggregator.best_reward.get(detected_env, float("-inf")) and r_syn >= ENV_EVAL_GATE else ""
            print(f"  [{detected_env}] syn={r_syn:.1f} raw={raw:.1f} pctl={r_perc:.2f} R={total:.3f} crash={crashed} {is_best}")

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

        from transformers import TrainerCallback

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

    # ---- Multi-round GRPO Training with self-retrieval ----
    num_rounds = max(1, iterations // 10)
    prompts_per_env = max(iterations // num_rounds, 5)
    print(f"\nTraining plan: {num_rounds} rounds x {prompts_per_env} prompts/env")
    print(f"  Self-retrieval: best code injected into prompts from round 2 onward\n")

    t0 = time.time()
    last_loss = 0.0

    for round_num in range(num_rounds):
        print(f"\n{'='*60}")
        print(f"  ROUND {round_num + 1}/{num_rounds}")
        if round_num > 0:
            for env_name in envs:
                best = aggregator.best_reward.get(env_name)
                if best is not None:
                    print(f"    {env_name}: best={best:.1f}, buffer={len(aggregator.buffers.get(env_name, []))}")
                    has_code = aggregator.get_best_code(env_name) is not None
                    print(f"      self-retrieval: {'injecting best code' if has_code else 'no code yet'}")
        print(f"{'='*60}")

        dataset = build_round_dataset(round_num, prompts_per_env)
        print(f"  Dataset: {len(dataset)} prompts")

        grpo_config = GRPOConfig(
            output_dir=output_dir,
            num_generations=4,
            learning_rate=5e-6,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            max_grad_norm=1.0,
            num_train_epochs=1,
            max_prompt_length=1024,
            max_completion_length=2048,
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

        result = trainer.train()
        last_loss = result.training_loss

        # Clean up trainer to free VRAM before next round
        del trainer
        torch.cuda.empty_cache()

    elapsed = time.time() - t0

    # ---- Final summary ----
    print(f"\nDone in {elapsed/60:.1f} min | Loss: {last_loss:.4f}")
    print(f"\nReward aggregator summary:")
    for env_name, stats in aggregator.summary().items():
        print(f"  {env_name}: best={stats['best']:.1f}, median={stats['median']:.1f}, samples={stats['buffer_size']}")

    # Save final adapter
    final_path = f"{output_dir}/final_adapter"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    summary = {
        "model": model_id,
        "envs": envs,
        "iterations": iterations,
        "elapsed_min": elapsed / 60,
        "loss": last_loss,
        "trainable": trainable_params,
        "total": total_params,
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
    gym_space_url: str = "",
):
    env_list = [e.strip() for e in envs.split(",")]
    result = train.remote(
        envs=env_list,
        iterations=iterations,
        webhook=webhook or None,
        gym_space_url=gym_space_url or None,
    )
    print(f"\nTraining complete: {result}")
