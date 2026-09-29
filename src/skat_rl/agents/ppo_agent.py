from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from skat_rl.engine.cards import Rank, Suit
from skat_rl.envs.observations import (
    OBSERVATION_SPECS, NUM_CARDS, NUM_TRICKS,
    CardStatus, Contract, Phase, RelativePlayer, TrickSlot,
    StructuredSkatObservation, effective_suit_table, empty_observations, index_observations,
)


@dataclass
class PPOConfig:
    action_dim: int = NUM_CARDS
    architecture: str = "mlp"
    use_belief: bool = False
    hidden_sizes: tuple[int, ...] = (256, 256)
    activation: str = "gelu"
    transformer_dim: int = 256
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_ff_dim: int = 1024
    transformer_dropout: float = 0.0
    belief_coef: float = 0.05
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
            sum(int(np.prod(spec[0])) for spec in OBSERVATION_SPECS.values()),
            config.hidden_sizes,
            config.action_dim,
            activation,
        )
        self.value_net = _mlp(
            sum(int(np.prod(spec[0])) for spec in OBSERVATION_SPECS.values()),
            config.hidden_sizes,
            1,
            activation,
        )

    def forward(self, observations, action_masks=None, actions=None):
        features = _mlp_features(observations)
        logits = self.policy_net(features)
        distribution = masked_categorical(logits, action_masks)
        values = self.value_net(features).squeeze(-1)

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
        return self.value_net(_mlp_features(observations)).squeeze(-1)

    def policy_logits(self, observations):
        return self.policy_net(_mlp_features(observations))


class SkatObservationTokenizer(nn.Module):
    """Embed explicit public fields into one STATE and 32 persistent CARD tokens."""

    num_cards = NUM_CARDS

    def __init__(self, model_dim):
        super().__init__()
        self.rank_embedding = nn.Embedding(len(Rank), model_dim)
        self.suit_embedding = nn.Embedding(len(Suit), model_dim)
        self.effective_suit_embedding = nn.Embedding(len(Suit) + 1, model_dim)
        self.card_status_embedding = nn.Embedding(len(CardStatus), model_dim)
        self.player_embedding = nn.Embedding(len(RelativePlayer), model_dim)
        self.trick_index_projection = nn.Linear(1, model_dim, bias=False)
        self.slot_embedding = nn.Embedding(len(TrickSlot), model_dim)
        self.state_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.phase_embedding = nn.Embedding(len(Phase), model_dim)
        self.contract_embedding = nn.Embedding(len(Contract), model_dim)
        self.declarer_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.leader_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.numeric_state_projection = nn.Linear(3, model_dim, bias=False)
        self.output_norm = nn.LayerNorm(model_dim)
        card_ids = torch.arange(NUM_CARDS)
        self.register_buffer("card_ranks", card_ids % len(Rank), persistent=False)
        self.register_buffer("card_suits", card_ids // len(Rank), persistent=False)
        self.register_buffer("effective_suits", torch.as_tensor(effective_suit_table()), persistent=False)

    def tokenize_cards(self, observation):
        status = observation["card_status"]
        played = status == CardStatus.PLAYED
        card_tokens = (
            self.rank_embedding(self.card_ranks)
            + self.suit_embedding(self.card_suits)
            + self.effective_suit_embedding(self.effective_suits[observation["contract"]])
            + self.card_status_embedding(status)
        )
        # Sentinels never enter an embedding or contribute to an unplayed token.
        played_by = torch.where(played, observation["played_by"], 0)
        trick_indices = torch.where(played, observation["trick_index"], 0).float() / (NUM_TRICKS - 1)
        slots = torch.where(played, observation["trick_slot"], 0)
        metadata = (
            self.player_embedding(played_by)
            + self.trick_index_projection(trick_indices.unsqueeze(-1))
            + self.slot_embedding(slots)
        )
        return card_tokens + metadata * played.unsqueeze(-1)

    def tokenize_state(self, observation):
        numeric = torch.stack([
            observation["declarer_points"].float() / 120.0,
            observation["defender_points"].float() / 120.0,
            observation["current_trick"].float() / (NUM_TRICKS - 1),
        ], dim=-1)
        return (
            self.state_token[:, 0]
            + self.phase_embedding(observation["phase"])
            + self.contract_embedding(observation["contract"])
            + self.declarer_projection(self.player_embedding(observation["relative_declarer"]))
            + self.leader_projection(self.player_embedding(observation["relative_current_leader"]))
            + self.numeric_state_projection(numeric)
        )

    def forward(self, observations):
        # void_info is retained in the public schema. As in the previous model,
        # the transformer can infer it from the complete per-card play history.
        state = self.tokenize_state(observations)
        cards = self.tokenize_cards(observations)
        return self.output_norm(torch.cat([state.unsqueeze(1), cards], dim=1))


class SkatTransformerActorCritic(nn.Module):
    """Card-centric shared transformer with PPO value/policy and auxiliary belief heads.

    The belief head is an auxiliary supervised task only. Its logits/probabilities are
    never fed into the policy or value heads. Belief gradients can still improve the
    shared tokenizer/encoder representation through the joint training loss.
    """

    belief_classes = 3

    def __init__(self, config: PPOConfig):
        super().__init__()
        if config.action_dim != SkatObservationTokenizer.num_cards:
            raise ValueError("The Skat transformer requires one action per card (32 actions).")
        if config.transformer_heads < 1:
            raise ValueError("transformer_heads must be at least 1.")
        if config.transformer_dim % config.transformer_heads != 0:
            raise ValueError("transformer_dim must be divisible by transformer_heads.")
        if config.transformer_layers < 1:
            raise ValueError("Transformer encoder needs at least one layer.")

        model_dim = config.transformer_dim
        self.use_belief = config.use_belief
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

        # Policy and belief score each physical card from
        # [global STATE representation, contextual CARD representation].
        per_card_input_dim = model_dim * 2
        self.policy_head = _head(per_card_input_dim, model_dim, 1, 0.01)
        self.value_head = _head(model_dim, model_dim, 1, 1.0)

        if self.use_belief:
            self.belief_head = _head(
                per_card_input_dim,
                model_dim,
                self.belief_classes,
                1.0,
            )

    def outputs(self, observations, include_belief=True):
        encoded = self.encoder(self.tokenizer(observations))
        state = encoded[:, 0]
        cards = encoded[:, 1:33]

        state_per_card = state.unsqueeze(1).expand(-1, cards.shape[1], -1)
        per_card_features = torch.cat([state_per_card, cards], dim=-1)

        policy_logits = self.policy_head(per_card_features).squeeze(-1)
        values = self.value_head(state).squeeze(-1)
        belief_logits = (
            self.belief_head(per_card_features)
            if self.use_belief and include_belief
            else None
        )

        return policy_logits, values, belief_logits

    def forward(self, observations, action_masks=None, actions=None, include_belief=True):
        logits, values, belief_logits = self.outputs(observations, include_belief)
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
        _, values, _ = self.outputs(observations, include_belief=False)
        return values

    def policy_logits(self, observations):
        logits, _, _ = self.outputs(observations, include_belief=False)
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

    @torch.no_grad()
    def act(self, observation, action_mask=None, deterministic=False):
        observations = {key: value.unsqueeze(0) for key, value in
                        observation_to_tensors(observation, self.device).items()}
        masks = None
        if action_mask is not None:
            masks = _to_tensor(action_mask, self.device).bool().unsqueeze(0)

        logits = self.model.policy_logits(observations)
        distribution = masked_categorical(logits, masks)
        if deterministic:
            action = distribution.probs.argmax(dim=-1)
        else:
            action = distribution.sample()

        return int(action.item())

    @torch.no_grad()
    def get_action_and_value(self, observations, action_masks):
        observations = observation_to_tensors(observations, self.device)
        action_masks = _to_tensor(action_masks, self.device).bool()
        if isinstance(self.model, SkatTransformerActorCritic):
            results = self.model(observations, action_masks, include_belief=False)
        else:
            results = self.model(observations, action_masks)
        actions, log_probs, _, values, _ = results
        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
        )

    @torch.no_grad()
    def get_values(self, observations):
        observations = observation_to_tensors(observations, self.device)
        return self.model.value(observations).cpu().numpy()

    @torch.no_grad()
    def get_belief_probabilities(self, observations):
        if not isinstance(self.model, SkatTransformerActorCritic) or not self.model.use_belief:
            raise RuntimeError("Belief predictions require a belief-enabled transformer.")
        observations = observation_to_tensors(observations, self.device)
        _, _, belief_logits = self.model.outputs(observations)
        return belief_logits.softmax(dim=-1).cpu().numpy()

    def update(self, rollout):
        observations = observation_to_tensors(rollout.observations, self.device)
        actions = _to_tensor(rollout.actions, self.device).long()
        old_log_probs = _to_tensor(rollout.log_probs, self.device).float()
        advantages = _to_tensor(rollout.advantages, self.device).float()
        returns = _to_tensor(rollout.returns, self.device).float()
        old_values = _to_tensor(rollout.values, self.device).float()
        action_masks = _to_tensor(rollout.action_masks, self.device).bool()
        belief_targets = None
        if self.config.use_belief and rollout.belief_targets is not None:
            belief_targets = _to_tensor(rollout.belief_targets, self.device).long()

        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        batch_size = observations["contract"].shape[0]
        minibatch_size = min(self.config.minibatch_size, batch_size)
        metrics = []

        for _ in range(self.config.update_epochs):
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, minibatch_size):
                minibatch = indices[start:start + minibatch_size]

                _, new_log_probs, entropy, new_values, belief_logits = self.model(
                    index_observations(observations, minibatch),
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
        try:
            agent.model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as error:
            raise ValueError(
                "Checkpoint weights do not match the current model architecture. "
                "Older transformer checkpoints cannot be resumed after the architecture change."
            ) from error
        if "optimizer_state_dict" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return agent


@dataclass
class RolloutBatch:
    observations: StructuredSkatObservation
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
    def __init__(self, rollout_steps, n_envs, action_dim, use_belief=False):
        shape = (rollout_steps, n_envs)
        self.observations = empty_observations(shape)
        self.actions = np.zeros(shape, dtype=np.int64)
        self.log_probs = np.zeros(shape, dtype=np.float32)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.dones = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        self.action_masks = np.zeros(shape + (action_dim,), dtype=bool)
        self.belief_targets = (
            np.full(shape + (action_dim,), -1, dtype=np.int64) if use_belief else None
        )
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
        for key in OBSERVATION_SPECS:
            self.observations[key][step] = observations[key]
        self.actions[step] = actions
        self.log_probs[step] = log_probs
        self.rewards[step] = rewards
        self.dones[step] = dones
        self.values[step] = values
        self.action_masks[step] = action_masks
        if self.belief_targets is not None and belief_targets is not None:
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
            observations={key: value.reshape((batch_size,) + OBSERVATION_SPECS[key][0])
                          for key, value in self.observations.items()},
            actions=self.actions.reshape(batch_size),
            log_probs=self.log_probs.reshape(batch_size),
            rewards=self.rewards.reshape(batch_size),
            dones=self.dones.reshape(batch_size),
            values=self.values.reshape(batch_size),
            action_masks=self.action_masks.reshape(batch_size, -1),
            advantages=self.advantages.reshape(batch_size),
            returns=self.returns.reshape(batch_size),
            belief_targets=(
                self.belief_targets.reshape(batch_size, -1)
                if self.belief_targets is not None else None
            ),
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


def observation_to_tensors(observations, device):
    """Transfer named fields together, preserving integer categorical semantics."""
    if not isinstance(observations, dict) or observations.keys() != OBSERVATION_SPECS.keys():
        raise ValueError("Expected structured policy observation fields only.")
    return {
        key: (value.to(device=device, dtype=torch.long, non_blocking=True)
              if isinstance(value, torch.Tensor)
              else torch.as_tensor(value, dtype=torch.long, device=device))
        for key, value in observations.items()
    }


def _mlp_features(observations):
    """Explicit feature adapter for the optional MLP, never used by the tokenizer."""
    batch_size = observations["contract"].shape[0]
    return torch.cat([
        observations[key].float().reshape(batch_size, int(np.prod(shape))) / max(high, 1)
        for key, (shape, _, high) in OBSERVATION_SPECS.items()
    ], dim=-1)


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
