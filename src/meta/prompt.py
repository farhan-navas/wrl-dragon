from __future__ import annotations


# Environment-specific observation/action descriptions with strategy hints
ENV_SPECS: dict[str, dict] = {
    "CartPole-v1": {
        "obs_dim": 4,
        "num_actions": 2,
        "obs_desc": "obs[0]=cart_position, obs[1]=cart_velocity, obs[2]=pole_angle, obs[3]=pole_angular_velocity",
        "action_desc": "0=push_left, 1=push_right",
        "goal": "Keep the pole balanced upright (angle near 0) for as long as possible. Max reward = 500.",
        "obs_ranges": "obs[0] in [-4.8, 4.8], obs[1] in [-inf, inf], obs[2] in [-0.418, 0.418] rad, obs[3] in [-inf, inf]",
        "heuristics": "A simple linear policy works well: push right if (angle + angular_velocity) > 0, else push left. Weighted linear combination of obs[2] and obs[3] can solve the environment.",
        "failure_modes": "Episode ends if pole angle > 0.418 rad or cart leaves [-4.8, 4.8]. Don't ignore cart position — drifting too far ends the episode.",
        "strategy_hints": "The pole angle and angular velocity are the most important observations. A PD controller on these two values can achieve near-optimal performance.",
    },
    "LunarLander-v3": {
        "obs_dim": 8,
        "num_actions": 4,
        "obs_desc": "obs[0]=x, obs[1]=y, obs[2]=vx, obs[3]=vy, obs[4]=angle, obs[5]=angular_vel, obs[6]=left_leg_contact, obs[7]=right_leg_contact",
        "action_desc": "0=noop, 1=left_engine, 2=main_engine, 3=right_engine",
        "goal": "Land softly between the flags with low velocity. Reward ~200 for good landing, negative for crashing.",
        "obs_ranges": "obs[0] x in [-1.5, 1.5], obs[1] y in [0, 1.5], obs[2] vx in [-5, 5], obs[3] vy in [-5, 5], obs[4] angle in [-pi, pi], obs[5] angular_vel in [-5, 5], obs[6-7] leg contact 0 or 1",
        "heuristics": "Fire main engine when vy < -0.1 (falling too fast). Use left/right engines to correct angle and horizontal drift. Once both legs touch, stop firing.",
        "failure_modes": "Crashing (high velocity at ground) gives -100. Fuel usage costs reward. Drifting too far off-screen ends episode with penalty.",
        "strategy_hints": "Prioritize vertical speed control (main engine) over horizontal. Correct angle first, then position. Use leg contact flags to know when to stop engines. A multi-threshold policy on vy, angle, and x works well.",
    },
    "Acrobot-v1": {
        "obs_dim": 6,
        "num_actions": 3,
        "obs_desc": "obs=[cos(theta1), sin(theta1), cos(theta2), sin(theta2), dtheta1, dtheta2]",
        "action_desc": "0=negative_torque, 1=no_torque, 2=positive_torque",
        "goal": "Swing the tip above the target line. Reward=-1 each step, episode ends when tip reaches height.",
        "obs_ranges": "cos/sin values in [-1, 1], dtheta1 in [-12.57, 12.57], dtheta2 in [-28.27, 28.27]",
        "heuristics": "Apply torque in the direction of angular velocity to build energy (like pumping a swing). When dtheta1 > 0, apply positive torque; when dtheta1 < 0, apply negative torque.",
        "failure_modes": "Max 500 steps. Random policy averages -500. Applying constant torque doesn't work — you need to time torque applications with the swing phase.",
        "strategy_hints": "This is an energy-pumping problem. The tip height = -cos(theta1) - cos(theta1 + theta2). Track angular velocity sign to decide torque direction. Alternate torque with the natural swing frequency.",
    },
    "MountainCar-v0": {
        "obs_dim": 2,
        "num_actions": 3,
        "obs_desc": "obs[0]=position (-1.2 to 0.6), obs[1]=velocity (-0.07 to 0.07)",
        "action_desc": "0=push_left, 1=no_push, 2=push_right",
        "goal": "Reach the flag at position >= 0.5. Reward=-1 each step. Build momentum by rocking.",
        "obs_ranges": "obs[0] position in [-1.2, 0.6], obs[1] velocity in [-0.07, 0.07]",
        "heuristics": "Push in the direction of current velocity: if velocity > 0, push right; if velocity < 0, push left. This builds momentum via resonance.",
        "failure_modes": "Max 200 steps. The car can't climb the hill in one push — random actions almost never solve it. Pushing against velocity wastes energy.",
        "strategy_hints": "The key insight is velocity-matching: accelerate in the direction you're already moving. A simple policy: action = 2 if velocity > 0 else 0. This is nearly optimal.",
    },
}

SYSTEM_PROMPT = (
    "You are an expert reinforcement learning engineer. "
    "You write Python policy functions for gymnasium environments. "
    "Output ONLY valid Python code, no markdown fences, no explanation."
)


def build_prompt(
    env_name: str,
    best_code: str | None = None,
    best_reward: float | None = None,
) -> list[dict[str, str]]:
    """Build chat messages for the parent LLM.

    Returns a list of message dicts for apply_chat_template.
    """
    spec = ENV_SPECS.get(env_name, {
        "obs_dim": "?", "num_actions": "?",
        "obs_desc": "unknown", "action_desc": "unknown",
        "goal": "maximize reward",
    })

    user_parts = [
        f"Environment: {env_name}",
        f"Observation: {spec['obs_dim']} floats — {spec['obs_desc']}",
        f"Observation ranges: {spec.get('obs_ranges', 'unknown')}",
        f"Actions: {spec['num_actions']} discrete — {spec['action_desc']}",
        f"Goal: {spec['goal']}",
    ]

    if spec.get("heuristics"):
        user_parts.append(f"Known good heuristics: {spec['heuristics']}")
    if spec.get("failure_modes"):
        user_parts.append(f"Common failure modes: {spec['failure_modes']}")
    if spec.get("strategy_hints"):
        user_parts.append(f"Strategy hints: {spec['strategy_hints']}")

    if best_code and best_reward is not None:
        user_parts.append(f"\nCurrent best policy (avg reward={best_reward:.1f}):")
        user_parts.append(best_code)
        user_parts.append("\nWrite an improved version that scores higher.")
    else:
        user_parts.append("\nWrite a policy function for this environment.")

    user_parts.append(
        "\nWrite `def select_action(obs: list[float]) -> int:` "
        "that returns the best action. Self-contained, no imports except math/random/numpy."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def format_prompt(tokenizer, env_name: str, **kwargs) -> str:
    """Build and format a prompt string using the tokenizer's chat template."""
    messages = build_prompt(env_name, **kwargs)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
