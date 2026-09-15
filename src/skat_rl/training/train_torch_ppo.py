import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from skat_rl.agents.ppo_agent import PPOAgent, PPOConfig, RolloutBatch, RolloutBuffer
from skat_rl.envs.skat_sb3_env import SkatSingleAgentEnv
from skat_rl.envs.skat_cpp_batched_env import SkatCppBatchedSingleAgentEnv


def main():
    args = _parse_args()
    if args.architecture == "mlp" and args.use_belief:
        raise ValueError("--belief requires --architecture transformer.")
    if args.env == "cpp":
        if args.rollout_size < 1:
            raise ValueError("--rollout-size must be at least 1")
    else:
        if args.n_envs < 1:
            raise ValueError("--n-envs must be at least 1")
        if args.rollout_steps < 1:
            raise ValueError("--rollout-steps must be at least 1")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"torch_ppo_skat_player{args.learning_player}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.env == "cpp":
        envs = SkatCppBatchedSingleAgentEnv(
            rollout_size=args.rollout_size,
            learning_player=args.learning_player,
            fixed_declarer=args.fixed_declarer,
            seed=args.seed,
        )
    else:
        envs = [
            _make_env(args, env_index)
            for env_index in range(args.n_envs)
        ]

    try:
        if args.env == "cpp":
            observation_dim = int(envs.observation_space.shape[0])
            action_dim = int(envs.action_space.n)
        else:
            observation_dim = int(envs[0].observation_space.shape[0])
            action_dim = int(envs[0].action_space.n)

        config = PPOConfig(
            observation_dim=observation_dim,
            action_dim=action_dim,
            architecture=args.architecture,
            use_belief=args.use_belief,
            hidden_sizes=tuple(args.hidden_sizes),
            activation=args.activation,
            transformer_dim=args.transformer_dim,
            transformer_layers=args.transformer_layers,
            transformer_heads=args.transformer_heads,
            transformer_ff_dim=args.transformer_ff_dim,
            transformer_dropout=args.transformer_dropout,
            belief_coef=args.belief_coef,
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
            if args.use_belief and agent.config.architecture != "transformer":
                raise ValueError("--belief requires a transformer checkpoint.")
            if (
                args.use_belief is True
                and agent.config.architecture == "transformer"
                and not agent.config.use_belief
            ):
                raise ValueError("--belief cannot enable belief in an existing checkpoint.")

        _save_config(output_dir, args, agent.config)
        if args.env == "cpp":
            _train_cpp_batched(agent, envs, args, output_dir, global_step)
        else:
            _train_python(agent, envs, args, output_dir, global_step)
    finally:
        if args.env == "cpp":
            envs.close()
        else:
            for env in envs:
                env.close()

    print(f"Saved PyTorch PPO training run to {output_dir}")


def _train_python(agent, envs, args, output_dir, global_step):
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
        use_belief=agent.config.architecture == "transformer" and agent.config.use_belief,
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
                "belief_loss",
                "belief_accuracy",
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
                belief_targets = None
                if rollout.belief_targets is not None:
                    belief_targets = np.asarray(
                        [env.belief_targets() for env in envs],
                        dtype=np.int64,
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
                    belief_targets,
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


def _train_cpp_batched(agent, env, args, output_dir, global_step):
    completed_episodes = []
    update = 0
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
                "belief_loss",
                "belief_accuracy",
                "entropy",
                "approx_kl",
                "clip_fraction",
                "explained_variance",
            ],
        )
        writer.writeheader()

        while global_step < args.total_timesteps:
            update += 1
            rollout, episodes, global_step = _collect_cpp_batched_rollout(
                agent,
                env,
                global_step,
            )
            completed_episodes.extend(episodes)
            metrics = agent.update(rollout)

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
                    f"update={update} "
                    f"steps={global_step} "
                    f"mean_return={mean_return:.3f} "
                    f"loss={metrics.get('loss', 0.0):.4f}"
                )

            if update % args.save_interval == 0:
                agent.save(output_dir / "model.pt")

        agent.save(output_dir / "model.pt")

    _save_episode_history(output_dir, completed_episodes)


def _collect_cpp_batched_rollout(agent, env, global_step):
    state = env.reset()
    use_belief = agent.config.architecture == "transformer" and agent.config.use_belief
    trajectories = {
        int(env_index): _empty_trajectory(use_belief)
        for env_index in state["active_indices"]
    }
    completed_episodes = []

    while len(state["active_indices"]) > 0:
        active_indices = state["active_indices"]
        observations = state["observations"]
        action_masks = state["action_masks"]
        belief_targets = state["belief_targets"] if use_belief else None
        actions, log_probs, values = agent.get_action_and_value(observations, action_masks)
        step_result = env.step(actions)
        rewards = step_result["rewards"]
        terminated = step_result["terminated"]

        for batch_index, env_index in enumerate(active_indices):
            env_index = int(env_index)
            trajectory = trajectories[env_index]
            trajectory["observations"].append(observations[batch_index])
            trajectory["actions"].append(int(actions[batch_index]))
            trajectory["log_probs"].append(float(log_probs[batch_index]))
            trajectory["rewards"].append(float(rewards[batch_index]))
            trajectory["dones"].append(float(terminated[batch_index]))
            trajectory["values"].append(float(values[batch_index]))
            trajectory["action_masks"].append(action_masks[batch_index])
            if use_belief:
                trajectory["belief_targets"].append(belief_targets[batch_index])

        global_step += len(active_indices)

        for episode_return, episode_length in zip(
            step_result["completed_returns"],
            step_result["completed_lengths"],
        ):
            completed_episodes.append(
                {
                    "return": float(episode_return),
                    "length": int(episode_length),
                    "global_step": global_step,
                }
            )

        state = {
            "active_indices": step_result["active_indices"],
            "observations": step_result["observations"],
            "action_masks": step_result["action_masks"],
            "belief_targets": step_result["belief_targets"] if use_belief else None,
        }

    return (
        _trajectories_to_rollout_batch(
            trajectories.values(),
            agent.config.gamma,
            agent.config.gae_lambda,
        ),
        completed_episodes,
        global_step,
    )


def _empty_trajectory(use_belief=False):
    trajectory = {
        "observations": [],
        "actions": [],
        "log_probs": [],
        "rewards": [],
        "dones": [],
        "values": [],
        "action_masks": [],
    }
    if use_belief:
        trajectory["belief_targets"] = []
    return trajectory


def _trajectories_to_rollout_batch(trajectories, gamma, gae_lambda):
    batches = []
    for trajectory in trajectories:
        if not trajectory["rewards"]:
            continue
        advantages, returns = _compute_episode_advantages(
            trajectory["rewards"],
            trajectory["values"],
            gamma,
            gae_lambda,
        )
        batches.append(
            {
                **trajectory,
                "advantages": advantages,
                "returns": returns,
            }
        )

    return RolloutBatch(
        observations=np.asarray(
            [observation for batch in batches for observation in batch["observations"]],
            dtype=np.float32,
        ),
        actions=np.asarray(
            [action for batch in batches for action in batch["actions"]],
            dtype=np.int64,
        ),
        log_probs=np.asarray(
            [log_prob for batch in batches for log_prob in batch["log_probs"]],
            dtype=np.float32,
        ),
        rewards=np.asarray(
            [reward for batch in batches for reward in batch["rewards"]],
            dtype=np.float32,
        ),
        dones=np.asarray(
            [done for batch in batches for done in batch["dones"]],
            dtype=np.float32,
        ),
        values=np.asarray(
            [value for batch in batches for value in batch["values"]],
            dtype=np.float32,
        ),
        action_masks=np.asarray(
            [mask for batch in batches for mask in batch["action_masks"]],
            dtype=bool,
        ),
        advantages=np.asarray(
            [advantage for batch in batches for advantage in batch["advantages"]],
            dtype=np.float32,
        ),
        returns=np.asarray(
            [return_ for batch in batches for return_ in batch["returns"]],
            dtype=np.float32,
        ),
        belief_targets=np.asarray(
            [target for batch in batches for target in batch.get("belief_targets", [])],
            dtype=np.int64,
        ) if all("belief_targets" in batch for batch in batches) else None,
    )


def _compute_episode_advantages(rewards, values, gamma, gae_lambda):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_advantage = 0.0

    for index in reversed(range(len(rewards))):
        if index == len(rewards) - 1:
            next_value = 0.0
            next_non_terminal = 0.0
        else:
            next_value = values[index + 1]
            next_non_terminal = 1.0

        delta = rewards[index] + gamma * next_value * next_non_terminal - values[index]
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[index] = last_advantage

    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


def _parse_args():
    parser = argparse.ArgumentParser(description="Train masked PPO from scratch with PyTorch.")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--n-envs", type=int, default=6)
    parser.add_argument("--rollout-size", type=int, default=2000)
    parser.add_argument("--learning-player", type=int, default=0)
    parser.add_argument("--fixed-declarer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env", choices=["python", "cpp"], default="cpp")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--continue-model")
    parser.add_argument("--device", default=None)
    parser.add_argument("--architecture", choices=["mlp", "transformer"], default="transformer")
    parser.add_argument("--belief", dest="use_belief", action="store_true",
                        help="Train an auxiliary supervised hidden-card belief head.")
    parser.set_defaults(use_belief=False)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[512, 512, 512, 512])
    parser.add_argument("--activation", choices=["tanh", "relu", "gelu"], default="tanh")
    parser.add_argument("--transformer-dim", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-ff-dim", type=int, default=1024)
    parser.add_argument("--transformer-dropout", type=float, default=0.0)
    parser.add_argument("--belief-coef", type=float, default=0.05)
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
    return SkatSingleAgentEnv(
        learning_player=args.learning_player,
        fixed_declarer=args.fixed_declarer,
        seed=_env_seed(args.seed, env_index),
    )


def _env_seed(seed, env_index):
    if seed is None:
        return None
    seed_sequence = np.random.SeedSequence([int(seed), int(env_index)])
    return int(seed_sequence.generate_state(1, dtype=np.uint64)[0])


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
