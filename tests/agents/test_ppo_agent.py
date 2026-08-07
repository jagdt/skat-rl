import numpy as np
import pytest

torch = pytest.importorskip("torch")

from skat_rl.agents.ppo_agent import (  # noqa: E402
    PPOAgent,
    PPOConfig,
    RolloutBuffer,
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
