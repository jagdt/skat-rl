import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from stable_baselines3.common.logger import configure
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
    model.set_logger(configure(str(output_dir), ["stdout", "csv"]))

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
    )

    model.save(model_path)
    _save_training_history(env, output_dir)
    _save_train_log_plots(output_dir)
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


def _save_train_log_plots(output_dir):
    log_path = output_dir / "progress.csv"
    plot_path = output_dir / "train_logs.png"

    if not log_path.exists():
        return

    train_metrics = [
        "train/approx_kl",
        "train/clip_fraction",
        "train/clip_range",
        "train/entropy_loss",
        "train/explained_variance",
        "train/learning_rate",
        "train/loss",
        "train/n_updates",
        "train/policy_gradient_loss",
        "train/value_loss",
    ]
    metric_values = {
        metric: {"x": [], "y": []}
        for metric in train_metrics
    }

    with open(log_path, "r", encoding="utf-8") as log_file:
        reader = csv.DictReader(log_file)
        for row_index, row in enumerate(reader, start=1):
            x_value = _parse_float(row.get("time/total_timesteps"), row_index)
            for metric in train_metrics:
                value = _parse_float(row.get(metric))
                if value is not None:
                    metric_values[metric]["x"].append(x_value)
                    metric_values[metric]["y"].append(value)

    metric_values = {
        metric: values
        for metric, values in metric_values.items()
        if values["y"]
    }
    if not metric_values:
        return

    n_cols = 2
    n_rows = (len(metric_values) + n_cols - 1) // n_cols
    fig, _ = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows), squeeze=False)

    for ax, (metric, values) in zip(fig.axes, metric_values.items()):
        ax.plot(values["x"], values["y"], linewidth=1.8)
        ax.set_title(metric.replace("train/", ""))
        ax.set_xlabel("Timesteps")
        ax.grid(True, alpha=0.3)

    for ax in fig.axes[len(metric_values):]:
        ax.remove()

    fig.suptitle("Maskable PPO Train Logs")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Saved train logs to {log_path}")
    print(f"Saved train log plot to {plot_path}")


def _parse_float(value, default=None):
    if value in (None, ""):
        return default

    try:
        return float(value)
    except ValueError:
        return default


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
