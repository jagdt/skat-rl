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

    env = SkatSingleAgentEnv(
        learning_player=0,
        seed=42,
    )

    env = Monitor(env)

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    model.learn(
        total_timesteps=500_000,
        progress_bar=True,
    )

    model.save(model_path)
    _save_training_history(env, output_dir)
    print(f"Saved training run to {output_dir}")


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