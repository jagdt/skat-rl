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
    use_belief: bool = False
    hidden_sizes: tuple[int, ...] = (256, 256)
    activation: str = "tanh"
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
            None,
        )

    def value(self, observations):
        return self.value_net(observations).squeeze(-1)

    def policy_logits(self, observations):
        return self.policy_net(observations)


class SkatObservationTokenizer(nn.Module):
    """Builds one STATE token plus one persistent token for each of the 32 cards.

    Card tokens are deliberately card-centric and fixed in number. They contain only
    information observable by the acting player:

      card = rank + printed suit + effective suit + status
             + (played-by + trick + slot, only when already played)

    The STATE token contains the contract, relative declarer, relative current
    trick leader, and a joint linear projection of numeric score/progress values.

    The tokenizer intentionally does *not* use the old current-player, current-trick,
    void-info, or explicit defender/declarer-role features as model inputs. The
    current player is used only to canonicalize absolute player IDs into a relative
    SELF/NEXT/PREVIOUS frame.

    This assumes history_cards/history_players contain every card already played,
    including cards in the current trick, in chronological play order.
    """

    observation_dim = 1149
    num_cards = 32
    num_players = 3
    history_slots = 30
    num_ranks = 8
    num_suits = 4

    # Contract IDs used internally by the tokenizer.
    clubs_contract = 0
    spades_contract = 1
    hearts_contract = 2
    diamonds_contract = 3
    grand_contract = 4
    null_contract = 5
    num_contracts = 6

    # Effective card categories: four printed suits plus TRUMP.
    trump_category = 4
    num_effective_categories = 5

    # Card status IDs.
    unknown_status = 0
    own_status = 1
    played_status = 2
    num_card_statuses = 3

    # Matches the Rank enum used by the engine: JACK = 7.
    jack_rank = 7

    def __init__(self, model_dim):
        super().__init__()

        # Card semantics. There is intentionally no separate 32-card-ID embedding:
        # rank + suit uniquely identify a physical Skat card.
        self.rank_embedding = nn.Embedding(self.num_ranks, model_dim)
        self.suit_embedding = nn.Embedding(self.num_suits, model_dim)
        self.effective_suit_embedding = nn.Embedding(
            self.num_effective_categories,
            model_dim,
        )
        self.card_status_embedding = nn.Embedding(self.num_card_statuses, model_dim)

        # Played-card metadata. No NONE categories are required; these embeddings
        # are simply added only for cards whose status is PLAYED.
        self.player_embedding = nn.Embedding(self.num_players, model_dim)
        self.trick_embedding = nn.Embedding(10, model_dim)
        self.slot_embedding = nn.Embedding(3, model_dim)

        # Global state semantics.
        self.state_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.contract_embedding = nn.Embedding(self.num_contracts, model_dim)

        # Declarer/leader use the same underlying relative-player identity space,
        # with small role-specific projections so their meaning in STATE is distinct.
        self.declarer_projection = nn.Linear(model_dim, model_dim, bias=False)
        self.leader_projection = nn.Linear(model_dim, model_dim, bias=False)

        # [declarer points, defender points, trick progress] -> model space.
        self.numeric_state_projection = nn.Linear(3, model_dim, bias=False)

        self.output_norm = nn.LayerNorm(model_dim)

        card_ids = torch.arange(self.num_cards)
        self.register_buffer("card_ranks", card_ids % self.num_ranks, persistent=False)
        self.register_buffer("card_suits", card_ids // self.num_ranks, persistent=False)

    def forward(self, observations):
        if observations.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Skat transformer expects observation_dim={self.observation_dim}, "
                f"got {observations.shape[-1]}."
            )

        batch_size = observations.shape[0]

        # Existing flat observation layout.
        own_hand = observations[:, 0:32]
        history_cards = observations[:, 32:992].reshape(batch_size, 30, 32)
        history_players = observations[:, 992:1082].reshape(batch_size, 30, 3)
        current_player = observations[:, 1114:1117]
        current_leader = observations[:, 1117:1120]
        declarer = observations[:, 1120:1123]
        global_features = observations[:, 1123:1134]

        # global_features layout used by the current environment:
        #   0:3   game kind one-hot [SUIT, GRAND, NULL]
        #   3:7   trump-suit one-hot [CLUBS, SPADES, HEARTS, DIAMONDS]
        #   7     normalized trick number
        #   8     normalized current-trick position (intentionally unused)
        #   9     normalized declarer points
        #   10    normalized defender points
        game_kind = global_features[:, 0:3].argmax(dim=-1)
        trump_suit = global_features[:, 3:7].argmax(dim=-1)
        contract = torch.where(
            game_kind == 0,
            trump_suit,
            torch.where(
                game_kind == 1,
                torch.full_like(game_kind, self.grand_contract),
                torch.full_like(game_kind, self.null_contract),
            ),
        )

        # Canonicalize all player identities relative to the acting player.
        # 0 = SELF, 1 = next seat in engine order, 2 = previous seat.
        current_player_id = current_player.argmax(dim=-1)
        declarer_id = declarer.argmax(dim=-1)
        leader_id = current_leader.argmax(dim=-1)
        relative_declarer = (declarer_id - current_player_id) % self.num_players
        relative_leader = (leader_id - current_player_id) % self.num_players

        # Reconstruct played-card metadata from the ordered 30-slot history.
        played = history_cards.amax(dim=1) > 0.5
        history_position = history_cards.argmax(dim=1)
        played_trick = history_position // 3
        played_slot = history_position % 3

        played_by_scores = torch.einsum("bsc,bsp->bcp", history_cards, history_players)
        played_by_id = played_by_scores.argmax(dim=-1)
        relative_played_by = (
            played_by_id - current_player_id.unsqueeze(1)
        ) % self.num_players

        own = own_hand > 0.5
        card_status = torch.full_like(history_position, self.unknown_status)
        card_status = torch.where(
            own,
            torch.full_like(card_status, self.own_status),
            card_status,
        )
        card_status = torch.where(
            played,
            torch.full_like(card_status, self.played_status),
            card_status,
        )

        # Determine effective suit/trump category for every card under this contract.
        ranks = self.card_ranks.unsqueeze(0).expand(batch_size, -1)
        suits = self.card_suits.unsqueeze(0).expand(batch_size, -1)
        contract_per_card = contract.unsqueeze(1)

        jack_is_trump = (ranks == self.jack_rank) & (
            contract_per_card != self.null_contract
        )
        suit_card_is_trump = (
            (contract_per_card < self.grand_contract)
            & (suits == contract_per_card)
        )
        is_trump = jack_is_trump | suit_card_is_trump
        effective_suit = torch.where(
            is_trump,
            torch.full_like(suits, self.trump_category),
            suits,
        )

        # Every game state always has exactly 32 card tokens.
        card_tokens = (
            self.rank_embedding(ranks)
            + self.suit_embedding(suits)
            + self.effective_suit_embedding(effective_suit)
            + self.card_status_embedding(card_status)
        )

        # Played metadata is conditional: unplayed cards receive exactly zero here,
        # so no artificial NONE embeddings are needed.
        played_metadata = (
            self.player_embedding(relative_played_by)
            + self.trick_embedding(played_trick)
            + self.slot_embedding(played_slot)
        )
        card_tokens = card_tokens + played_metadata * played.unsqueeze(-1)

        numeric_state = torch.stack(
            [
                global_features[:, 9],
                global_features[:, 10],
                global_features[:, 7],
            ],
            dim=-1,
        )

        state_token = self.state_token.expand(batch_size, -1, -1).squeeze(1)
        state_token = (
            state_token
            + self.contract_embedding(contract)
            + self.declarer_projection(self.player_embedding(relative_declarer))
            + self.leader_projection(self.player_embedding(relative_leader))
            + self.numeric_state_projection(numeric_state)
        )

        tokens = torch.cat([state_token.unsqueeze(1), card_tokens], dim=1)
        return self.output_norm(tokens)


class SkatTransformerActorCritic(nn.Module):
    """Card-centric shared transformer with PPO value/policy and auxiliary belief heads.

    The belief head is an auxiliary supervised task only. Its logits/probabilities are
    never fed into the policy or value heads. Belief gradients can still improve the
    shared tokenizer/encoder representation through the joint training loss.
    """

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
        observations = _to_tensor(observation, self.device).float().unsqueeze(0)
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
        observations = _to_tensor(observations, self.device).float()
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
        observations = _to_tensor(observations, self.device).float()
        return self.model.value(observations).cpu().numpy()

    @torch.no_grad()
    def get_belief_probabilities(self, observations):
        if not isinstance(self.model, SkatTransformerActorCritic) or not self.model.use_belief:
            raise RuntimeError("Belief predictions require a belief-enabled transformer.")
        observations = _to_tensor(observations, self.device).float()
        _, _, belief_logits = self.model.outputs(observations)
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
        if self.config.use_belief and rollout.belief_targets is not None:
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
        config_values = dict(checkpoint["config"])
        config_values.pop("belief_decoder_layers", None)
        config_values.pop("belief_detach_updates", None)
        if "use_belief" not in config_values:
            config_values["use_belief"] = config_values.get("architecture") == "transformer"
        config = PPOConfig(**config_values)
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
    def __init__(self, rollout_steps, n_envs, observation_dim, action_dim, use_belief=False):
        shape = (rollout_steps, n_envs)
        self.observations = np.zeros(shape + (observation_dim,), dtype=np.float32)
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
        self.observations[step] = observations
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
            observations=self.observations.reshape(batch_size, -1),
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
