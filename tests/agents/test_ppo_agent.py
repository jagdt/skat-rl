import numpy as np
import pytest

torch = pytest.importorskip("torch")

from skat_rl.agents.ppo_agent import (  # noqa: E402
    PPOAgent,
    PPOConfig,
    RolloutBuffer,
    SkatObservationTokenizer,
    SkatTransformerActorCritic,
    masked_categorical,
)


def test_masked_categorical_never_assigns_probability_to_illegal_actions():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    masks = torch.tensor([[True, False, True, False]])

    distribution = masked_categorical(logits, masks)

    assert distribution.probs[0, 1].item() == pytest.approx(0.0)
    assert distribution.probs[0, 3].item() == pytest.approx(0.0)
    assert distribution.probs[0, 0].item() > 0.0
    assert distribution.probs[0, 2].item() > 0.0
    assert distribution.probs.sum().item() == pytest.approx(1.0)


def test_masked_categorical_rejects_empty_legal_action_rows():
    logits = torch.zeros((2, 3))
    masks = torch.tensor(
        [
            [True, False, False],
            [False, False, False],
        ]
    )

    with pytest.raises(ValueError, match="at least one legal action"):
        masked_categorical(logits, masks)


def test_agent_act_respects_action_mask_when_only_one_action_is_legal():
    config = PPOConfig(observation_dim=5, action_dim=4, hidden_sizes=(8,))
    agent = PPOAgent(config, device="cpu")
    observation = np.zeros(5, dtype=np.float32)
    mask = np.array([False, False, True, False])

    assert agent.act(observation, mask, deterministic=False) == 2
    assert agent.act(observation, mask, deterministic=True) == 2


def test_get_action_and_value_returns_one_result_per_observation():
    config = PPOConfig(observation_dim=5, action_dim=4, hidden_sizes=(8,))
    agent = PPOAgent(config, device="cpu")
    observations = np.zeros((3, 5), dtype=np.float32)
    masks = np.array(
        [
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, False],
        ]
    )

    actions, log_probs, values = agent.get_action_and_value(observations, masks)

    assert actions.tolist() == [0, 1, 2]
    assert log_probs.shape == (3,)
    assert values.shape == (3,)


def test_rollout_buffer_computes_gae_returns_and_flattens_batch():
    buffer = RolloutBuffer(
        rollout_steps=3,
        n_envs=1,
        observation_dim=2,
        action_dim=2,
    )

    observations = np.zeros((1, 2), dtype=np.float32)
    action_masks = np.ones((1, 2), dtype=bool)
    for step, reward in enumerate([1.0, 1.0, 1.0]):
        buffer.add(
            step,
            observations,
            np.array([0]),
            np.array([0.0], dtype=np.float32),
            np.array([reward], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            action_masks,
        )

    buffer.compute_returns_and_advantages(
        last_values=np.array([0.0], dtype=np.float32),
        last_dones=np.array([1.0], dtype=np.float32),
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert buffer.advantages[:, 0].tolist() == pytest.approx([3.0, 2.0, 1.0])
    assert buffer.returns[:, 0].tolist() == pytest.approx([3.0, 2.0, 1.0])

    batch = buffer.flatten()

    assert batch.observations.shape == (3, 2)
    assert batch.actions.shape == (3,)
    assert batch.action_masks.shape == (3, 2)


def test_agent_update_returns_metrics_and_changes_parameters():
    torch.manual_seed(1)
    config = PPOConfig(
        observation_dim=4,
        action_dim=3,
        hidden_sizes=(8,),
        update_epochs=2,
        minibatch_size=2,
    )
    agent = PPOAgent(config, device="cpu")
    rollout = RolloutBuffer(
        rollout_steps=4,
        n_envs=1,
        observation_dim=config.observation_dim,
        action_dim=config.action_dim,
    )
    observations = np.arange(4, dtype=np.float32).reshape(1, 4)
    action_masks = np.ones((1, 3), dtype=bool)

    for step in range(4):
        rollout.add(
            step,
            observations + step,
            np.array([step % config.action_dim]),
            np.array([-1.0], dtype=np.float32),
            np.array([1.0 if step == 3 else 0.25], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            action_masks,
        )
    rollout.compute_returns_and_advantages(
        last_values=np.array([0.0], dtype=np.float32),
        last_dones=np.array([1.0], dtype=np.float32),
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter in agent.model.state_dict().items()
    }
    metrics = agent.update(rollout.flatten())

    assert {
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
    }.issubset(metrics)
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in agent.model.state_dict().items()
    )


def test_agent_save_and_load_round_trip(tmp_path):
    config = PPOConfig(observation_dim=5, action_dim=4, hidden_sizes=(8,))
    agent = PPOAgent(config, device="cpu")
    path = tmp_path / "model.pt"

    agent.save(path)
    loaded = PPOAgent.load(path, device="cpu")

    assert loaded.config == agent.config
    for name, parameter in agent.model.state_dict().items():
        assert torch.equal(parameter, loaded.model.state_dict()[name])


def _tiny_transformer_config(**overrides):
    values = {
        "observation_dim": 1149,
        "action_dim": 32,
        "architecture": "transformer",
        "use_belief": True,
        "transformer_dim": 32,
        "transformer_layers": 1,
        "transformer_heads": 4,
        "transformer_ff_dim": 64,
        "update_epochs": 1,
        "minibatch_size": 2,
    }
    values.update(overrides)
    return PPOConfig(**values)


def test_tokenizer_uses_relative_players_and_ordered_play_history():
    tokenizer = SkatObservationTokenizer(model_dim=16)
    observations = np.zeros((2, 1149), dtype=np.float32)
    observations[:, 8] = 1.0  # Own spade seven.
    observations[:, 32 + 6] = 1.0  # Club ace was played in the first slot.
    observations[:, 1123] = 1.0  # Suit game.
    observations[:, 1128] = 1.0  # Hearts trump.

    for row, current_player in enumerate([0, 1]):
        observations[row, 1114 + current_player] = 1.0
        observations[row, 1117 + (current_player + 2) % 3] = 1.0
        observations[row, 1120 + (current_player + 1) % 3] = 1.0
        observations[row, 992 + (current_player + 1) % 3] = 1.0

    tokens = tokenizer(torch.as_tensor(observations))

    assert tokens.shape == (2, 33, 16)
    assert torch.allclose(tokens[0], tokens[1])

    without_history = observations[:1].copy()
    without_history[0, 32 + 6] = 0.0
    without_history[0, 992 + 1] = 0.0
    unplayed_tokens = tokenizer(torch.as_tensor(without_history))
    assert not torch.allclose(tokens[0, 1 + 6], unplayed_tokens[0, 1 + 6])


def test_tokenizer_linearly_projects_normalized_played_trick_progress():
    tokenizer = SkatObservationTokenizer(model_dim=16)
    observations = np.zeros((2, 1149), dtype=np.float32)
    observations[:, 1114] = 1.0  # Player 0 is acting.
    observations[:, 1117] = 1.0  # Player 0 leads.
    observations[:, 1120] = 1.0  # Player 0 is declarer.
    observations[:, 1123] = 1.0  # Suit game.
    observations[:, 1126] = 1.0  # Clubs trump.

    card_id = 6
    observations[0, 32 + card_id] = 1.0  # Card in trick 0, slot 0.
    observations[0, 992] = 1.0

    final_trick_history_slot = 27
    observations[1, 32 + final_trick_history_slot * 32 + card_id] = 1.0
    observations[1, 992 + final_trick_history_slot * 3] = 1.0

    tokens = tokenizer(torch.as_tensor(observations))

    assert tokens.shape == (2, 33, 16)
    assert not torch.allclose(tokens[0, 1 + card_id], tokens[1, 1 + card_id])


def test_transformer_outputs_policy_value_and_card_beliefs():
    agent = PPOAgent(_tiny_transformer_config(), device="cpu")
    observations = np.zeros((2, 1149), dtype=np.float32)
    action_masks = np.ones((2, 32), dtype=bool)

    actions, log_probs, values = agent.get_action_and_value(observations, action_masks)
    beliefs = agent.get_belief_probabilities(observations)

    assert actions.shape == (2,)
    assert log_probs.shape == (2,)
    assert values.shape == (2,)
    assert beliefs.shape == (2, 32, 3)
    assert beliefs.sum(axis=-1) == pytest.approx(np.ones((2, 32)))


def test_auxiliary_belief_does_not_feed_policy_or_value_heads():
    model = SkatTransformerActorCritic(_tiny_transformer_config())
    observations = torch.zeros((2, 1149))

    policy_logits, values, belief_logits = model.outputs(observations)
    rollout_policy_logits, rollout_values, rollout_beliefs = model.outputs(
        observations, include_belief=False,
    )
    assert policy_logits.shape == (2, 32)
    assert values.shape == (2,)
    assert belief_logits.shape == (2, 32, 3)
    assert rollout_beliefs is None
    assert torch.allclose(policy_logits, rollout_policy_logits)
    assert torch.allclose(values, rollout_values)

    (policy_logits.sum() + values.sum()).backward()
    assert all(parameter.grad is None for parameter in model.belief_head.parameters())

    model.zero_grad()
    _, _, belief_logits = model.outputs(observations)
    belief_logits.sum().backward()
    assert all(parameter.grad is None for parameter in model.policy_head.parameters())
    assert all(parameter.grad is None for parameter in model.value_head.parameters())
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())


def test_transformer_update_trains_belief_head_with_supervised_targets():
    torch.manual_seed(3)
    config = _tiny_transformer_config()
    agent = PPOAgent(config, device="cpu")
    rollout = RolloutBuffer(2, 1, config.observation_dim, config.action_dim, use_belief=True)
    observations = np.zeros((1, config.observation_dim), dtype=np.float32)
    masks = np.ones((1, config.action_dim), dtype=bool)
    targets = np.arange(config.action_dim, dtype=np.int64)[None, :] % 3

    for step in range(2):
        actions, log_probs, values = agent.get_action_and_value(observations, masks)
        rollout.add(
            step,
            observations,
            actions,
            log_probs,
            np.array([float(step)], dtype=np.float32),
            np.array([float(step == 1)], dtype=np.float32),
            values,
            masks,
            targets,
        )
    rollout.compute_returns_and_advantages(
        last_values=np.zeros(1, dtype=np.float32),
        last_dones=np.ones(1, dtype=np.float32),
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    before = agent.model.belief_head[-1].weight.detach().clone()

    metrics = agent.update(rollout.flatten())

    assert metrics["belief_loss"] > 0.0
    assert 0.0 <= metrics["belief_accuracy"] <= 1.0
    assert not torch.equal(before, agent.model.belief_head[-1].weight)


def test_transformer_without_belief_has_only_policy_and_value_heads(tmp_path):
    config = _tiny_transformer_config(use_belief=False)
    agent = PPOAgent(config, device="cpu")
    observations = np.zeros((2, config.observation_dim), dtype=np.float32)
    masks = np.ones((2, config.action_dim), dtype=bool)

    actions, log_probs, values = agent.get_action_and_value(observations, masks)
    _, _, belief_logits = agent.model.outputs(torch.as_tensor(observations))

    assert actions.shape == log_probs.shape == values.shape == (2,)
    assert belief_logits is None
    assert not hasattr(agent.model, "belief_head")
    assert agent.model.policy_head[0].in_features == 2 * config.transformer_dim
    with pytest.raises(RuntimeError, match="belief-enabled"):
        agent.get_belief_probabilities(observations)

    path = tmp_path / "no_belief.pt"
    agent.save(path)
    loaded = PPOAgent.load(path, device="cpu")
    assert loaded.config.use_belief is False
    assert not hasattr(loaded.model, "belief_head")
    assert loaded.get_action_and_value(observations, masks)[2].shape == (2,)


def test_transformer_without_belief_updates_without_targets():
    config = _tiny_transformer_config(use_belief=False)
    agent = PPOAgent(config, device="cpu")
    rollout = RolloutBuffer(2, 1, config.observation_dim, config.action_dim, use_belief=False)
    observations = np.zeros((1, config.observation_dim), dtype=np.float32)
    masks = np.ones((1, config.action_dim), dtype=bool)

    for step in range(2):
        actions, log_probs, values = agent.get_action_and_value(observations, masks)
        rollout.add(
            step, observations, actions, log_probs,
            np.array([float(step)], dtype=np.float32),
            np.array([float(step == 1)], dtype=np.float32),
            values, masks,
        )
    rollout.compute_returns_and_advantages(
        np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32),
        config.gamma, config.gae_lambda,
    )
    batch = rollout.flatten()
    assert batch.belief_targets is None
    metrics = agent.update(batch)
    assert metrics["belief_loss"] == 0.0
    assert metrics["belief_accuracy"] == 0.0


def test_transformer_config_defaults_to_no_belief():
    config = PPOConfig(observation_dim=1149, action_dim=32, architecture="transformer",
                       transformer_dim=32, transformer_layers=1, transformer_heads=4,
                       transformer_ff_dim=64)
    agent = PPOAgent(config, device="cpu")

    assert config.use_belief is False
    assert not hasattr(agent.model, "belief_head")


def test_legacy_transformer_checkpoint_without_use_belief_stays_enabled(tmp_path):
    config = _tiny_transformer_config()
    agent = PPOAgent(config, device="cpu")
    path = tmp_path / "legacy_model.pt"
    agent.save(path)
    checkpoint = torch.load(path, map_location="cpu")
    del checkpoint["config"]["use_belief"]
    checkpoint["config"]["belief_decoder_layers"] = 2
    checkpoint["config"]["belief_detach_updates"] = 100
    torch.save(checkpoint, path)

    loaded = PPOAgent.load(path, device="cpu")

    assert loaded.config.use_belief is True
    assert hasattr(loaded.model, "belief_head")
    assert "belief_decoder_layers" not in loaded.config.__dict__


def test_incompatible_transformer_checkpoint_has_clear_error(tmp_path):
    agent = PPOAgent(_tiny_transformer_config(), device="cpu")
    path = tmp_path / "old_model.pt"
    agent.save(path)
    checkpoint = torch.load(path, map_location="cpu")
    checkpoint["model_state_dict"]["policy_head.0.weight"] = torch.zeros((1, 1))
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="Older transformer checkpoints cannot be resumed"):
        PPOAgent.load(path, device="cpu")
