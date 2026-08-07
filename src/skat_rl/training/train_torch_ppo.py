import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from skat_rl.agents.ppo_agent import PPOAgent, PPOConfig, RolloutBuffer
from skat_rl.envs.skat_sb3_env import SkatSingleAgentEnv

try:
    from skat_rl.envs.skat_cpp_sb3_env import SkatCppSingleAgentEnv
except ImportError:  # pragma: no cover - C++ extension is optional during development
    SkatCppSingleAgentEnv = None


SEED_SPACING = 100_000_000


def main():
    args = _parse_args()
    if args.n_envs < 1:
        raise ValueError("--n-envs must be at least 1")
    if args.rollout_steps < 1:
        raise ValueError("--rollout-steps must be at least 1")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"torch_ppo_skat_player{args.learning_player}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    envs = [
        _make_env(args, env_index)
        for env_index in range(args.n_envs)
    ]

    try:
        observation_dim = int(envs[0].observation_space.shape[0])
        action_dim = int(envs[0].action_space.n)

        config = PPOConfig(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_sizes=tuple(args.hidden_sizes),
            activation=args.activation,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_coef=args.clip_coef,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            update_epochs=args.update_epochs,
            minibatch_size=args.minibatch_size,
            target_kl=args.target_kl,
        )

        if args.continue_model is None:
            agent = PPOAgent(config, device=args.device)
            global_step = 0
        else:
            agent = PPOAgent.load(args.continue_model, device=args.device)
            global_step = 0
            if agent.config.observation_dim != observation_dim or agent.config.action_dim != action_dim:
                raise ValueError("Checkpoint observation/action dimensions do not match the env.")

        _save_config(output_dir, args, config)
        _train(agent, envs, args, output_dir, global_step)
    finally:
        for env in envs:
            env.close()

    print(f"Saved PyTorch PPO training run to {output_dir}")


def _train(agent, envs, args, output_dir, global_step):
    observations = []
    episode_returns = [0.0 for _ in envs]
    episode_lengths = [0 for _ in envs]
    completed_episodes = []

    for env_index, env in enumerate(envs):
        observation, _ = env.reset(seed=_env_seed(args.seed, env_index))
        observations.append(observation)

    observations = np.asarray(observations, dtype=np.float32)
    dones = np.zeros(len(envs), dtype=np.float32)
    rollout = RolloutBuffer(
        args.rollout_steps,
        len(envs),
        agent.config.observation_dim,
        agent.config.action_dim,
    )
    rollout_timesteps = args.rollout_steps * len(envs)
    updates = max(math.ceil(args.total_timesteps / rollout_timesteps), 1)
    metrics_path = output_dir / "metrics.csv"

    with open(metrics_path, "w", encoding="utf-8", newline="") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=[
                "update",
                "total_timesteps",
                "mean_episode_return",
                "mean_episode_length",
                "loss",
                "policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "clip_fraction",
                "explained_variance",
            ],
        )
        writer.writeheader()

        for update in range(1, updates + 1):
            for step in range(args.rollout_steps):
                action_masks = np.asarray(
                    [env.action_masks() for env in envs],
                    dtype=bool,
                )
                actions, log_probs, values = agent.get_action_and_value(observations, action_masks)

                next_observations = []
                rewards = np.zeros(len(envs), dtype=np.float32)
                next_dones = np.zeros(len(envs), dtype=np.float32)

                for env_index, (env, action) in enumerate(zip(envs, actions)):
                    next_observation, reward, terminated, truncated, _ = env.step(int(action))
                    done = terminated or truncated
                    rewards[env_index] = float(reward)
                    next_dones[env_index] = float(done)
                    episode_returns[env_index] += float(reward)
                    episode_lengths[env_index] += 1
                    global_step += 1

                    if done:
                        completed_episodes.append(
                            {
                                "return": episode_returns[env_index],
                                "length": episode_lengths[env_index],
                                "global_step": global_step,
                            }
                        )
                        episode_returns[env_index] = 0.0
                        episode_lengths[env_index] = 0
                        next_observation, _ = env.reset()

                    next_observations.append(next_observation)

                rollout.add(
                    step,
                    observations,
                    actions,
                    log_probs,
                    rewards,
                    dones,
                    values,
                    action_masks,
                )
                observations = np.asarray(next_observations, dtype=np.float32)
                dones = next_dones

            last_values = agent.get_values(observations)
            rollout.compute_returns_and_advantages(
                last_values,
                dones,
                agent.config.gamma,
                agent.config.gae_lambda,
            )
            metrics = agent.update(rollout.flatten())

            recent_episodes = completed_episodes[-100:]
            mean_return = _mean([episode["return"] for episode in recent_episodes])
            mean_length = _mean([episode["length"] for episode in recent_episodes])
            row = {
                "update": update,
                "total_timesteps": global_step,
                "mean_episode_return": mean_return,
                "mean_episode_length": mean_length,
                **metrics,
            }
            writer.writerow(row)
            metrics_file.flush()

            if update % args.log_interval == 0 or update == 1:
                print(
                    f"update={update}/{updates} "
                    f"steps={global_step} "
                    f"mean_return={mean_return:.3f} "
                    f"loss={metrics.get('loss', 0.0):.4f}"
                )

            if update % args.save_interval == 0 or update == updates:
                agent.save(output_dir / "model.pt")

    _save_episode_history(output_dir, completed_episodes)


def _parse_args():
    parser = argparse.ArgumentParser(description="Train masked PPO from scratch with PyTorch.")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--n-envs", type=int, default=6)
    parser.add_argument("--learning-player", type=int, default=0)
    parser.add_argument("--fixed-declarer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env", choices=["python", "cpp"], default="python")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--continue-model")
    parser.add_argument("--device", default=None)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--activation", choices=["tanh", "relu", "gelu"], default="tanh")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=10)
    return parser.parse_args()


def _make_env(args, env_index):
    env_class = SkatSingleAgentEnv
    if args.env == "cpp":
        if SkatCppSingleAgentEnv is None:
            raise ImportError("C++ Skat env is unavailable. Build the extension or use --env python.")
        env_class = SkatCppSingleAgentEnv

    return env_class(
        learning_player=args.learning_player,
        fixed_declarer=args.fixed_declarer,
        seed=_env_seed(args.seed, env_index),
    )


def _env_seed(seed, env_index):
    if seed is None:
        return None
    return int(seed) + env_index * SEED_SPACING


def _save_config(output_dir, args, config):
    payload = {
        "args": vars(args),
        "ppo": {
            **config.__dict__,
            "hidden_sizes": list(config.hidden_sizes),
        },
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(payload, config_file, indent=2)


def _save_episode_history(output_dir, episodes):
    if not episodes:
        return

    with open(output_dir / "episodes.csv", "w", encoding="utf-8", newline="") as episode_file:
        writer = csv.DictWriter(
            episode_file,
            fieldnames=["episode", "global_step", "return", "length"],
        )
        writer.writeheader()
        for episode_index, episode in enumerate(episodes, start=1):
            writer.writerow({"episode": episode_index, **episode})


def _mean(values):
    if not values:
        return 0.0
    return float(sum(values) / len(values))


if __name__ == "__main__":
    main()
