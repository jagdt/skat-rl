from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F



@dataclass
class PPOConfig:
    observation_dim: int
    action_dim: int
    architecture: str = "mlp"
    hidden_sizes: tuple[int, ...] = (256, 256)
    activation: str = "tanh"
    transformer_dim: int = 256
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_ff_dim: int = 1024
    transformer_dropout: float = 0.0
    belief_decoder_layers: int = 2
    belief_coef: float = 1.0
    belief_detach_updates: int = 100
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

    def forward(self, observations, action_masks=None, actions=None, detach_beliefs=True):
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
            None,
        )

    def value(self, observations):
        return self.value_net(observations).squeeze(-1)

    def policy_logits(self, observations, detach_beliefs=True):
        return self.policy_net(observations)


class SkatObservationTokenizer(nn.Module):
    """Turns the fixed 1,149-value Skat observation into semantic tokens."""

    observation_dim = 1149
    num_cards = 32
    num_players = 3
    history_slots = 30

    def __init__(self, model_dim):
        super().__init__()
        self.card_embedding = nn.Embedding(self.num_cards, model_dim)
        self.card_status_embedding = nn.Embedding(3, model_dim)
        self.history_position_embedding = nn.Embedding(self.history_slots + 1, model_dim)
        self.played_by_embedding = nn.Embedding(self.num_players + 1, model_dim)
        self.current_trick_projection = nn.Linear(1, model_dim, bias=False)

        self.player_embedding = nn.Embedding(self.num_players, model_dim)
        self.player_projection = nn.Linear(8, model_dim)
        self.state_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.state_projection = nn.Linear(11, model_dim)
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(self, observations):
        if observations.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Skat transformer expects observation_dim={self.observation_dim}, "
                f"got {observations.shape[-1]}."
            )

        batch_size = observations.shape[0]
        own_hand = observations[:, 0:32]
        history_cards = observations[:, 32:992].reshape(batch_size, 30, 32)
        history_players = observations[:, 992:1082].reshape(batch_size, 30, 3)
        current_trick = observations[:, 1082:1114]
        current_player = observations[:, 1114:1117]
        current_leader = observations[:, 1117:1120]
        declarer = observations[:, 1120:1123]
        global_features = observations[:, 1123:1134]
        void_info = observations[:, 1134:1149].reshape(batch_size, 3, 5)

        played = history_cards.amax(dim=1) > 0.5
        history_positions = history_cards.argmax(dim=1) + 1
        history_positions = torch.where(
            played,
            history_positions,
            torch.zeros_like(history_positions),
        )
        played_by_scores = torch.einsum("bsc,bsp->bcp", history_cards, history_players)
        played_by = played_by_scores.argmax(dim=-1)
        played_by = torch.where(
            played,
            played_by,
            torch.full_like(played_by, self.num_players),
        )

        own = own_hand > 0.5
        card_status = torch.zeros_like(history_positions)
        card_status = torch.where(own, torch.ones_like(card_status), card_status)
        card_status = torch.where(played, torch.full_like(card_status, 2), card_status)

        card_ids = torch.arange(self.num_cards, device=observations.device)
        card_tokens = (
            self.card_embedding(card_ids).unsqueeze(0)
            + self.card_status_embedding(card_status)
            + self.history_position_embedding(history_positions)
            + self.played_by_embedding(played_by)
            + self.current_trick_projection(current_trick.unsqueeze(-1))
        )

        player_features = torch.cat(
            [
                current_player.unsqueeze(-1),
                current_leader.unsqueeze(-1),
                declarer.unsqueeze(-1),
                void_info,
            ],
            dim=-1,
        )
        player_ids = torch.arange(self.num_players, device=observations.device)
        player_tokens = (
            self.player_embedding(player_ids).unsqueeze(0)
            + self.player_projection(player_features)
        )

        state_token = self.state_token.expand(batch_size, -1, -1)
        state_token = state_token + self.state_projection(global_features).unsqueeze(1)
        return self.output_norm(torch.cat([state_token, player_tokens, card_tokens], dim=1))


class SkatTransformerActorCritic(nn.Module):
    belief_classes = 3

    def __init__(self, config: PPOConfig):
        super().__init__()
        if config.observation_dim != SkatObservationTokenizer.observation_dim:
            raise ValueError("The Skat transformer requires the 1,149-value Skat observation.")
        if config.action_dim != SkatObservationTokenizer.num_cards:
            raise ValueError("The Skat transformer requires one action per card (32 actions).")
        if config.transformer_heads < 1:
            raise ValueError("transformer_heads must be at least 1.")
        if config.transformer_dim % config.transformer_heads != 0:
            raise ValueError("transformer_dim must be divisible by transformer_heads.")
        if config.transformer_layers < 1 or config.belief_decoder_layers < 1:
            raise ValueError("Transformer encoder and belief decoder need at least one layer.")

        model_dim = config.transformer_dim
        self.tokenizer = SkatObservationTokenizer(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers,
            norm=nn.LayerNorm(model_dim),
            enable_nested_tensor=False,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.belief_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.belief_decoder_layers,
            norm=nn.LayerNorm(model_dim),
        )
        self.belief_queries = nn.Embedding(SkatObservationTokenizer.num_cards, model_dim)
        self.belief_head = nn.Linear(model_dim, self.belief_classes)
        self.belief_feature_projection = nn.Sequential(
            nn.Linear(SkatObservationTokenizer.num_cards * self.belief_classes, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        self.policy_head = _head(model_dim * 2, model_dim, config.action_dim, 0.01)
        self.value_head = _head(model_dim * 2, model_dim, 1, 1.0)

    def outputs(self, observations, detach_beliefs=True):
        memory = self.encoder(self.tokenizer(observations))
        batch_size = observations.shape[0]
        card_ids = torch.arange(
            SkatObservationTokenizer.num_cards,
            device=observations.device,
        )
        queries = (
            self.belief_queries(card_ids)
            + self.tokenizer.card_embedding(card_ids)
        ).unsqueeze(0).expand(batch_size, -1, -1)
        belief_states = self.belief_decoder(queries, memory)
        belief_logits = self.belief_head(belief_states)
        belief_probabilities = belief_logits.softmax(dim=-1)
        if detach_beliefs:
            belief_probabilities = belief_probabilities.detach()

        own_cards = observations[:, 0:32] > 0.5
        played_cards = observations[:, 32:992].reshape(batch_size, 30, 32).amax(dim=1) > 0.5
        hidden_cards = ~(own_cards | played_cards)
        belief_probabilities = belief_probabilities * hidden_cards.unsqueeze(-1)
        belief_features = self.belief_feature_projection(
            belief_probabilities.flatten(start_dim=1)
        )
        actor_critic_features = torch.cat([memory[:, 0], belief_features], dim=-1)
        policy_logits = self.policy_head(actor_critic_features)
        values = self.value_head(actor_critic_features).squeeze(-1)
        return policy_logits, values, belief_logits

    def forward(self, observations, action_masks=None, actions=None, detach_beliefs=True):
        logits, values, belief_logits = self.outputs(observations, detach_beliefs)
        distribution = masked_categorical(logits, action_masks)
        if actions is None:
            actions = distribution.sample()
        return (
            actions,
            distribution.log_prob(actions),
            distribution.entropy(),
            values,
            belief_logits,
        )

    def value(self, observations):
        _, values, _ = self.outputs(observations, detach_beliefs=True)
        return values

    def policy_logits(self, observations, detach_beliefs=True):
        logits, _, _ = self.outputs(observations, detach_beliefs)
        return logits


class PPOAgent:
    """
    Small, dependency-light PPO implementation for discrete masked action spaces.
    """

    def __init__(self, config: PPOConfig, device: str | torch.device | None = None):
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if config.architecture == "mlp":
            self.model = MaskedActorCritic(config).to(self.device)
        elif config.architecture == "transformer":
            self.model = SkatTransformerActorCritic(config).to(self.device)
        else:
            raise ValueError(f"Unsupported architecture: {config.architecture}")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.updates_completed = 0

    @torch.no_grad()
    def act(self, observation, action_mask=None, deterministic=False):
        observations = _to_tensor(observation, self.device).float().unsqueeze(0)
        masks = None
        if action_mask is not None:
            masks = _to_tensor(action_mask, self.device).bool().unsqueeze(0)

        logits = self.model.policy_logits(observations, detach_beliefs=True)
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
        actions, log_probs, _, values, _ = self.model(
            observations,
            action_masks,
            detach_beliefs=True,
        )
        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
        )

    @torch.no_grad()
    def get_values(self, observations):
        observations = _to_tensor(observations, self.device).float()
        return self.model.value(observations).cpu().numpy()

    @torch.no_grad()
    def get_belief_probabilities(self, observations):
        if not isinstance(self.model, SkatTransformerActorCritic):
            raise RuntimeError("Belief predictions require architecture='transformer'.")
        observations = _to_tensor(observations, self.device).float()
        _, _, belief_logits = self.model.outputs(observations, detach_beliefs=True)
        return belief_logits.softmax(dim=-1).cpu().numpy()

    def update(self, rollout):
        observations = _to_tensor(rollout.observations, self.device).float()
        actions = _to_tensor(rollout.actions, self.device).long()
        old_log_probs = _to_tensor(rollout.log_probs, self.device).float()
        advantages = _to_tensor(rollout.advantages, self.device).float()
        returns = _to_tensor(rollout.returns, self.device).float()
        old_values = _to_tensor(rollout.values, self.device).float()
        action_masks = _to_tensor(rollout.action_masks, self.device).bool()
        belief_targets = None
        if rollout.belief_targets is not None:
            belief_targets = _to_tensor(rollout.belief_targets, self.device).long()

        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        batch_size = observations.shape[0]
        minibatch_size = min(self.config.minibatch_size, batch_size)
        metrics = []

        for _ in range(self.config.update_epochs):
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, minibatch_size):
                minibatch = indices[start:start + minibatch_size]

                _, new_log_probs, entropy, new_values, belief_logits = self.model(
                    observations[minibatch],
                    action_masks[minibatch],
                    actions[minibatch],
                    detach_beliefs=self.updates_completed < self.config.belief_detach_updates,
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

                belief_loss = torch.zeros((), device=self.device)
                belief_accuracy = torch.zeros((), device=self.device)
                if belief_logits is not None and belief_targets is not None:
                    minibatch_targets = belief_targets[minibatch]
                    valid_targets = minibatch_targets >= 0
                    if valid_targets.any():
                        belief_loss = F.cross_entropy(
                            belief_logits[valid_targets],
                            minibatch_targets[valid_targets],
                        )
                        belief_accuracy = (
                            belief_logits.argmax(dim=-1)[valid_targets]
                            == minibatch_targets[valid_targets]
                        ).float().mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_loss
                    + self.config.belief_coef * belief_loss
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
                        "belief_loss": float(belief_loss.item()),
                        "belief_accuracy": float(belief_accuracy.item()),
                        "entropy": float(entropy_loss.item()),
                        "approx_kl": float(approx_kl.item()),
                        "clip_fraction": float(clip_fraction.item()),
                        "explained_variance": float(explained_variance.item()),
                    }
                )

            if self.config.target_kl is not None and metrics[-1]["approx_kl"] > self.config.target_kl:
                break

        self.updates_completed += 1
        return _mean_metrics(metrics)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "updates_completed": self.updates_completed,
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
        agent.updates_completed = int(checkpoint.get("updates_completed", 0))
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
    belief_targets: np.ndarray | None = None


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
        self.belief_targets = np.full(shape + (action_dim,), -1, dtype=np.int64)
        self.advantages = np.zeros(shape, dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)

    def add(
        self,
        step,
        observations,
        actions,
        log_probs,
        rewards,
        dones,
        values,
        action_masks,
        belief_targets=None,
    ):
        self.observations[step] = observations
        self.actions[step] = actions
        self.log_probs[step] = log_probs
        self.rewards[step] = rewards
        self.dones[step] = dones
        self.values[step] = values
        self.action_masks[step] = action_masks
        if belief_targets is not None:
            self.belief_targets[step] = belief_targets

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
            belief_targets=self.belief_targets.reshape(batch_size, -1),
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


def _head(input_dim, hidden_dim, output_dim, output_gain):
    head = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )
    nn.init.orthogonal_(head[-1].weight, gain=output_gain)
    nn.init.constant_(head[-1].bias, 0.0)
    return head


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
