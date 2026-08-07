from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical



@dataclass
class PPOConfig:
    observation_dim: int
    action_dim: int
    hidden_sizes: tuple[int, ...] = (256, 256)
    activation: str = "tanh"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 256
    target_kl: float = None
    normalize_advantages: bool = True


class MaskedActorCritic(nn.Module):
    def __init__(self, config: PPOConfig):
        super().__init__()
        activation = _activation(config.activation)

        self.policy_net = _mlp(
            config.observation_dim,
            config.hidden_sizes,
            config.action_dim,
            activation,
        )
        self.value_net = _mlp(
            config.observation_dim,
            config.hidden_sizes,
            1,
            activation,
        )

    def forward(self, observations, action_masks=None, actions=None):
        logits = self.policy_net(observations)
        distribution = masked_categorical(logits, action_masks)
        values = self.value_net(observations).squeeze(-1)

        if actions is None:
            actions = distribution.sample()

        return (
            actions,
            distribution.log_prob(actions),
            distribution.entropy(),
            values,
        )

    def value(self, observations):
        return self.value_net(observations).squeeze(-1)


class PPOAgent:
    """
    Small, dependency-light PPO implementation for discrete masked action spaces.
    """

    def __init__(self, config: PPOConfig, device: str | torch.device | None = None):
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = MaskedActorCritic(config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    @torch.no_grad()
    def act(self, observation, action_mask=None, deterministic=False):
        observations = _to_tensor(observation, self.device).float().unsqueeze(0)
        masks = None
        if action_mask is not None:
            masks = _to_tensor(action_mask, self.device).bool().unsqueeze(0)

        logits = self.model.policy_net(observations)
        distribution = masked_categorical(logits, masks)
        if deterministic:
            action = distribution.probs.argmax(dim=-1)
        else:
            action = distribution.sample()

        return int(action.item())

    @torch.no_grad()
    def get_action_and_value(self, observations, action_masks):
        observations = _to_tensor(observations, self.device).float()
        action_masks = _to_tensor(action_masks, self.device).bool()
        actions, log_probs, _, values = self.model(observations, action_masks)
        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
        )

    @torch.no_grad()
    def get_values(self, observations):
        observations = _to_tensor(observations, self.device).float()
        return self.model.value(observations).cpu().numpy()

    def update(self, rollout):
        observations = _to_tensor(rollout.observations, self.device).float()
        actions = _to_tensor(rollout.actions, self.device).long()
        old_log_probs = _to_tensor(rollout.log_probs, self.device).float()
        advantages = _to_tensor(rollout.advantages, self.device).float()
        returns = _to_tensor(rollout.returns, self.device).float()
        old_values = _to_tensor(rollout.values, self.device).float()
        action_masks = _to_tensor(rollout.action_masks, self.device).bool()

        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        batch_size = observations.shape[0]
        minibatch_size = min(self.config.minibatch_size, batch_size)
        metrics = []

        for _ in range(self.config.update_epochs):
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, minibatch_size):
                minibatch = indices[start:start + minibatch_size]

                _, new_log_probs, entropy, new_values = self.model(
                    observations[minibatch],
                    action_masks[minibatch],
                    actions[minibatch],
                )

                log_ratio = new_log_probs - old_log_probs[minibatch]
                ratio = log_ratio.exp()

                pg_loss_1 = -advantages[minibatch] * ratio
                pg_loss_2 = -advantages[minibatch] * torch.clamp(
                    ratio,
                    1.0 - self.config.clip_coef,
                    1.0 + self.config.clip_coef,
                )
                policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                value_loss_unclipped = (new_values - returns[minibatch]).pow(2)
                value_clipped = old_values[minibatch] + torch.clamp(
                    new_values - old_values[minibatch],
                    -self.config.clip_coef,
                    self.config.clip_coef,
                )
                value_loss_clipped = (value_clipped - returns[minibatch]).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (ratio - 1.0).abs() > self.config.clip_coef
                    ).float().mean()
                    explained_variance = _explained_variance(
                        returns[minibatch],
                        new_values,
                    )

                metrics.append(
                    {
                        "loss": float(loss.item()),
                        "policy_loss": float(policy_loss.item()),
                        "value_loss": float(value_loss.item()),
                        "entropy": float(entropy_loss.item()),
                        "approx_kl": float(approx_kl.item()),
                        "clip_fraction": float(clip_fraction.item()),
                        "explained_variance": float(explained_variance.item()),
                    }
                )

            if self.config.target_kl is not None and metrics[-1]["approx_kl"] > self.config.target_kl:
                break

        return _mean_metrics(metrics)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path, device: str | torch.device | None = None):
        checkpoint = torch.load(path, map_location=device or "cpu")
        config = PPOConfig(**checkpoint["config"])
        agent = cls(config, device=device)
        agent.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return agent


@dataclass
class RolloutBatch:
    observations: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    action_masks: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray


class RolloutBuffer:
    def __init__(self, rollout_steps, n_envs, observation_dim, action_dim):
        shape = (rollout_steps, n_envs)
        self.observations = np.zeros(shape + (observation_dim,), dtype=np.float32)
        self.actions = np.zeros(shape, dtype=np.int64)
        self.log_probs = np.zeros(shape, dtype=np.float32)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.dones = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        self.action_masks = np.zeros(shape + (action_dim,), dtype=bool)
        self.advantages = np.zeros(shape, dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)

    def add(self, step, observations, actions, log_probs, rewards, dones, values, action_masks):
        self.observations[step] = observations
        self.actions[step] = actions
        self.log_probs[step] = log_probs
        self.rewards[step] = rewards
        self.dones[step] = dones
        self.values[step] = values
        self.action_masks[step] = action_masks

    def compute_returns_and_advantages(self, last_values, last_dones, gamma, gae_lambda):
        last_advantage = np.zeros_like(last_values, dtype=np.float32)

        for step in reversed(range(self.rewards.shape[0])):
            if step == self.rewards.shape[0] - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]

            delta = (
                self.rewards[step]
                + gamma * next_values * next_non_terminal
                - self.values[step]
            )
            last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
            self.advantages[step] = last_advantage

        self.returns = self.advantages + self.values

    def flatten(self):
        rollout_steps, n_envs = self.actions.shape
        batch_size = rollout_steps * n_envs
        return RolloutBatch(
            observations=self.observations.reshape(batch_size, -1),
            actions=self.actions.reshape(batch_size),
            log_probs=self.log_probs.reshape(batch_size),
            rewards=self.rewards.reshape(batch_size),
            dones=self.dones.reshape(batch_size),
            values=self.values.reshape(batch_size),
            action_masks=self.action_masks.reshape(batch_size, -1),
            advantages=self.advantages.reshape(batch_size),
            returns=self.returns.reshape(batch_size),
        )


def masked_categorical(logits, action_masks=None):
    if action_masks is None:
        return Categorical(logits=logits)

    if not action_masks.any(dim=-1).all():
        raise ValueError("Every action mask row must contain at least one legal action.")

    masked_logits = logits.masked_fill(~action_masks, torch.finfo(logits.dtype).min)
    return Categorical(logits=masked_logits)


def _mlp(input_dim, hidden_sizes, output_dim, activation):
    layers = []
    previous_dim = input_dim
    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(previous_dim, hidden_size))
        layers.append(activation())
        previous_dim = hidden_size

    output = nn.Linear(previous_dim, output_dim)
    nn.init.orthogonal_(output.weight, gain=0.01)
    nn.init.constant_(output.bias, 0.0)
    layers.append(output)
    return nn.Sequential(*layers)


def _activation(name):
    activations = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
    }
    try:
        return activations[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported activation: {name}") from error


def _to_tensor(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.as_tensor(value, device=device)


def _explained_variance(y_true, y_pred):
    variance = torch.var(y_true, unbiased=False)
    if variance == 0:
        return torch.tensor(0.0, device=y_true.device)
    return 1.0 - torch.var(y_true - y_pred, unbiased=False) / variance


def _mean_metrics(metrics):
    if not metrics:
        return {}

    return {
        key: sum(metric[key] for metric in metrics) / len(metrics)
        for key in metrics[0]
    }
