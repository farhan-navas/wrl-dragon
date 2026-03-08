from __future__ import annotations


# Environment-specific observation/action descriptions
ENV_SPECS: dict[str, dict] = {
    "CartPole-v1": {
        "obs_dim": 4,
        "num_actions": 2,
        "obs_desc": "obs[0]=cart_position, obs[1]=cart_velocity, obs[2]=pole_angle, obs[3]=pole_angular_velocity",
        "action_desc": "0=push_left, 1=push_right",
        "goal": "Keep the pole balanced upright (angle near 0) for as long as possible. Max reward = 500.",
    },
    "LunarLander-v3": {
        "obs_dim": 8,
        "num_actions": 4,
        "obs_desc": "obs[0]=x, obs[1]=y, obs[2]=vx, obs[3]=vy, obs[4]=angle, obs[5]=angular_vel, obs[6]=left_leg_contact, obs[7]=right_leg_contact",
        "action_desc": "0=noop, 1=left_engine, 2=main_engine, 3=right_engine",
        "goal": "Land softly between the flags with low velocity. Reward ~200 for good landing, negative for crashing.",
    },
    "Acrobot-v1": {
        "obs_dim": 6,
        "num_actions": 3,
        "obs_desc": "obs=[cos(theta1), sin(theta1), cos(theta2), sin(theta2), dtheta1, dtheta2]",
        "action_desc": "0=negative_torque, 1=no_torque, 2=positive_torque",
        "goal": "Swing the tip above the target line. Reward=-1 each step, episode ends when tip reaches height.",
    },
    "MountainCar-v0": {
        "obs_dim": 2,
        "num_actions": 3,
        "obs_desc": "obs[0]=position (-1.2 to 0.6), obs[1]=velocity (-0.07 to 0.07)",
        "action_desc": "0=push_left, 1=no_push, 2=push_right",
        "goal": "Reach the flag at position >= 0.5. Reward=-1 each step. Build momentum by rocking.",
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
        f"Actions: {spec['num_actions']} discrete — {spec['action_desc']}",
        f"Goal: {spec['goal']}",
    ]

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
