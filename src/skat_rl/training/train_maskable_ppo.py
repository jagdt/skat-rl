import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from skat_rl.envs.skat_sb3_env import SkatSingleAgentEnv


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("models") / f"maskable_ppo_skat_player0_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.zip"
    total_timesteps = 500_000
    env_config = {
        "learning_player": 0,
        "seed": 42,
    }
    model_config = {
        "policy": "MlpPolicy",
        "policy_kwargs":dict(net_arch=dict(pi=[512, 256, 256],vf=[512, 256, 256])),
        "verbose": 1,
        "learning_rate": 3e-4,
        "n_steps": 8192,
        "batch_size": 256,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
    }

    _save_config(output_dir, timestamp, total_timesteps, env_config, model_config)

    env = SkatSingleAgentEnv(**env_config)

    env = Monitor(env)

    model = MaskablePPO(
        env=env,
        **model_config,
    )

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
    )

    model.save(model_path)
    _save_training_history(env, output_dir)
    print(f"Saved training run to {output_dir}")


def _save_config(output_dir, timestamp, total_timesteps, env_config, model_config):
    config = {
        "timestamp": timestamp,
        "total_timesteps": total_timesteps,
        "environment": env_config,
        "model": model_config,
    }
    config_path = output_dir / "config.json"

    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)

    print(f"Saved training config to {config_path}")


def _save_training_history(env, output_dir):
    rewards = env.get_episode_rewards()
    lengths = env.get_episode_lengths()
    times = env.get_episode_times()

    if not rewards:
        return

    csv_path = output_dir / "history.csv"
    plot_path = output_dir / "training.png"

    with open(csv_path, "w", encoding="utf-8") as history_file:
        history_file.write("episode,reward,length,time_seconds\n")
        for episode, (reward, length, elapsed) in enumerate(
            zip(rewards, lengths, times),
            start=1,
        ):
            history_file.write(f"{episode},{reward},{length},{elapsed}\n")

    episodes = list(range(1, len(rewards) + 1))
    rolling_rewards = _rolling_average(rewards, window=100)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        episodes,
        rolling_rewards,
        linewidth=2,
        label="Rolling reward mean (100 episodes)",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Maskable PPO Skat Rolling Reward")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Saved training history to {csv_path}")
    print(f"Saved training plot to {plot_path}")


def _rolling_average(values, window):
    averages = []
    total = 0.0

    for index, value in enumerate(values):
        total += value

        if index >= window:
            total -= values[index - window]
            divisor = window
        else:
            divisor = index + 1

        averages.append(total / divisor)

    return averages


if __name__ == "__main__":
    main()