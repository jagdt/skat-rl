import argparse
import csv
import json
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from skat_rl.training import train_torch_ppo  # noqa: E402
from skat_rl.envs.observations import empty_observations, index_observations


def test_belief_cli_flag_is_explicit(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_torch_ppo"])
    assert train_torch_ppo._parse_args().use_belief is False

    monkeypatch.setattr(sys, "argv", ["train_torch_ppo", "--belief"])
    assert train_torch_ppo._parse_args().use_belief is True


def test_belief_flag_rejects_mlp_architecture(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "train_torch_ppo", "--architecture", "mlp", "--belief",
    ])
    with pytest.raises(ValueError, match="requires --architecture transformer"):
        train_torch_ppo.main()


def test_belief_flag_rejects_no_belief_checkpoint(monkeypatch, tmp_path):
    checkpoint = tmp_path / "model.pt"
    config = train_torch_ppo.PPOConfig(
        action_dim=32, architecture="transformer",
        transformer_dim=32, transformer_layers=1, transformer_heads=4,
        transformer_ff_dim=64,
    )
    train_torch_ppo.PPOAgent(config, device="cpu").save(checkpoint)
    monkeypatch.setattr(sys, "argv", [
        "train_torch_ppo", "--belief", "--continue-model", str(checkpoint),
        "--opponent-model", str(checkpoint),
        "--rollout-size", "1", "--output-dir", str(tmp_path),
    ])

    with pytest.raises(ValueError, match="cannot enable belief"):
        train_torch_ppo.main()


def test_env_seed_derives_deterministic_distinct_parallel_seeds():
    seeds = [train_torch_ppo._env_seed(42, env_index) for env_index in range(4)]

    assert seeds == [
        train_torch_ppo._env_seed(42, env_index)
        for env_index in range(4)
    ]
    assert len(set(seeds)) == 4
    assert all(0 <= seed < 2**64 for seed in seeds)
    assert train_torch_ppo._env_seed(None, 2) is None


def test_make_env_uses_python_env_by_default():
    args = argparse.Namespace(
        env="python",
        learning_player=0,
        fixed_declarer=0,
        seed=42,
    )

    env = train_torch_ppo._make_env(args, env_index=0)
    try:
        observation, _ = env.reset(seed=42)

        assert env.observation_space.contains(observation)
        assert env.action_masks().shape == (env.action_space.n,)
    finally:
        env.close()


def test_save_config_writes_cli_args_and_ppo_config(tmp_path):
    args = argparse.Namespace(
        total_timesteps=128,
        rollout_steps=8,
        n_envs=2,
        rollout_size=4,
        hidden_sizes=[16, 8],
    )
    config = train_torch_ppo.PPOConfig(
        action_dim=4,
        hidden_sizes=(16, 8),
    )

    train_torch_ppo._save_config(tmp_path, args, config)

    with open(tmp_path / "config.json", encoding="utf-8") as config_file:
        payload = json.load(config_file)

    assert payload["args"]["total_timesteps"] == 128
    assert payload["args"]["n_envs"] == 2
    assert payload["args"]["rollout_size"] == 4
    assert "observation_version" not in payload["ppo"]
    assert "observation_dim" not in payload["ppo"]
    assert payload["ppo"]["action_dim"] == 4
    assert payload["ppo"]["hidden_sizes"] == [16, 8]


def test_compute_episode_advantages_uses_complete_episode_boundary():
    advantages, returns = train_torch_ppo._compute_episode_advantages(
        rewards=[1.0, 1.0, 1.0],
        values=[0.0, 0.0, 0.0],
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert advantages.tolist() == pytest.approx([3.0, 2.0, 1.0])
    assert returns.tolist() == pytest.approx([3.0, 2.0, 1.0])


def test_trajectories_to_rollout_batch_flattens_complete_episodes():
    trajectories = [
        {
            "observations": [
                empty_observations(),
                empty_observations(),
            ],
            "actions": [0, 1],
            "log_probs": [-0.1, -0.2],
            "rewards": [0.0, 1.0],
            "dones": [0.0, 1.0],
            "values": [0.0, 0.0],
            "action_masks": [
                [True, False],
                [False, True],
            ],
        },
        {
            "observations": [
                empty_observations(),
            ],
            "actions": [1],
            "log_probs": [-0.3],
            "rewards": [2.0],
            "dones": [1.0],
            "values": [0.0],
            "action_masks": [
                [True, True],
            ],
        },
    ]

    rollout = train_torch_ppo._trajectories_to_rollout_batch(
        trajectories,
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert rollout.observations["card_status"].shape == (3, 32)
    assert rollout.action_masks.shape == (3, 2)
    assert rollout.actions.tolist() == [0, 1, 1]
    assert rollout.advantages.tolist() == pytest.approx([1.0, 1.0, 2.0])


def test_collect_cpp_batched_rollout_collects_complete_games():
    env = train_torch_ppo.SkatCppBatchedSingleAgentEnv(
        rollout_size=2,
        learning_player=0,
        fixed_declarer=0,
        seed=1,
    )
    config = train_torch_ppo.PPOConfig(
        action_dim=env.action_space.n,
        architecture="transformer",
        use_belief=True,
        transformer_dim=32,
        transformer_layers=1,
        transformer_heads=4,
        transformer_ff_dim=64,
    )
    agent = train_torch_ppo.PPOAgent(config, device="cpu")

    rollout, episodes, global_step = train_torch_ppo._collect_cpp_batched_rollout(
        agent,
        env,
        global_step=0,
        opponent=train_torch_ppo.PPOAgent(config, device="cpu"),
    )

    assert global_step == 20
    assert [episode["length"] for episode in episodes] == [10, 10]
    assert rollout.observations["card_status"].shape == (20, 32)
    assert rollout.action_masks.shape == (20, 32)
    assert rollout.belief_targets.shape == (20, 32)
    assert set(np.unique(rollout.belief_targets)) <= {-1, 0, 1, 2}


def test_collect_cpp_batched_rollout_without_belief_skips_targets():
    env = train_torch_ppo.SkatCppBatchedSingleAgentEnv(
        rollout_size=2, learning_player=0, fixed_declarer=0, seed=1,
    )
    config = train_torch_ppo.PPOConfig(
        action_dim=32, architecture="transformer",
        use_belief=False, transformer_dim=32, transformer_layers=1,
        transformer_heads=4, transformer_ff_dim=64,
    )
    agent = train_torch_ppo.PPOAgent(config, device="cpu")

    rollout, episodes, global_step = train_torch_ppo._collect_cpp_batched_rollout(
        agent, env, global_step=0, opponent=train_torch_ppo.PPOAgent(config, device="cpu"),
    )

    assert global_step == 20
    assert len(episodes) == 2
    assert rollout.belief_targets is None
    assert agent.update(rollout)["belief_loss"] == 0.0


@pytest.mark.parametrize("learning_player", [0, 1, 2])
def test_fixed_opponent_collection_batches_policies_and_credits_terminal_reward(learning_player):
    class RecordingPolicy:
        config = train_torch_ppo.PPOConfig(
            action_dim=32, architecture="transformer",
            use_belief=True, gamma=1.0, gae_lambda=1.0,
        )

        def __init__(self):
            self.observations = []

        def get_action_and_value(self, observations, masks):
            assert len(observations["contract"]) > 0
            rows = np.flatnonzero(
                (env.last_state["current_players"] == learning_player) == (self is learner))
            expected = index_observations(env.last_state["observations"], rows)
            for field in observations:
                np.testing.assert_array_equal(observations[field], expected[field])
            self.observations.append(observations.copy())
            return masks.argmax(axis=1), np.zeros(len(masks)), np.zeros(len(masks))

    class TracedEnv(train_torch_ppo.SkatCppBatchedSingleAgentEnv):
        def reset(self, *args, **kwargs):
            self.last_state = super().reset(*args, **kwargs)
            return self.last_state

        def step(self, actions):
            self.last_state = super().step(actions)
            return self.last_state

    learner, opponent = RecordingPolicy(), RecordingPolicy()
    env = TracedEnv(
        12, learning_player=learning_player, seed=17,
    )
    rollout, episodes, steps = train_torch_ppo._collect_cpp_batched_rollout(
        learner, env, 7, opponent,
    )
    assert steps == 127
    assert len(episodes) == 12
    assert all(episode["length"] == 10 for episode in episodes)
    assert sum(len(batch["contract"]) for batch in learner.observations) == 120
    assert sum(len(batch["contract"]) for batch in opponent.observations) == 240
    assert len(learner.observations) <= 30
    assert len(opponent.observations) <= 30
    assert max(len(batch["contract"]) for batch in opponent.observations) == 12
    assert "current_player" not in rollout.observations
    assert rollout.action_masks[np.arange(120), rollout.actions].all()
    assert rollout.belief_targets.shape == (120, 32)
    assert (rollout.belief_targets[np.arange(120), rollout.actions] == -1).all()
    rewards = rollout.rewards.reshape(12, 10)
    assert (rewards[:, :-1] == 0).all()
    np.testing.assert_allclose(rewards[:, -1], [ep["return"] for ep in episodes])
    assert (rollout.dones.reshape(12, 10)[:, :-1] == 0).all()
    assert (rollout.dones.reshape(12, 10)[:, -1] == 1).all()
    np.testing.assert_allclose(rollout.returns.reshape(12, 10), np.repeat(rewards[:, -1:], 10, axis=1))


def _small_transformer_config(**overrides):
    values = dict(
        action_dim=32, architecture="transformer",
        transformer_dim=16, transformer_layers=1, transformer_heads=2,
        transformer_ff_dim=32, update_epochs=1, minibatch_size=20,
    )
    return train_torch_ppo.PPOConfig(**(values | overrides))


@pytest.mark.parametrize("loader", ["load", "initialize", "opponent"])
def test_checkpoint_loaders_work_without_observation_version(tmp_path, loader):
    path = tmp_path / "model.pt"
    config = _small_transformer_config()
    source = train_torch_ppo.PPOAgent(config, device="cpu")
    source.save(path)
    checkpoint = torch.load(path, weights_only=True)
    assert "observation_version" not in checkpoint["config"]
    if loader == "load":
        loaded = train_torch_ppo.PPOAgent.load(path, "cpu")
    elif loader == "initialize":
        loaded = train_torch_ppo.initialize_agent(config, path, "cpu")
    else:
        loaded = train_torch_ppo.load_frozen_opponent(path, config, "cpu")
    for name, value in loaded.model.state_dict().items():
        assert torch.equal(value, source.model.state_dict()[name]), name


def test_pretrained_weights_transfer_with_fresh_ppo_hyperparameters(tmp_path):
    from skat_rl.training.train_supervised import save_pretrained

    source = train_torch_ppo.PPOAgent(_small_transformer_config(gamma=1.0), device="cpu")
    path = tmp_path / "pretrained.pt"
    save_pretrained(source, path, epoch=1)
    learner = train_torch_ppo.initialize_agent(_small_transformer_config(gamma=.99), path, "cpu")
    assert learner.config.gamma == .99
    assert not learner.optimizer.state
    for name, value in learner.model.state_dict().items():
        assert torch.equal(value, source.model.state_dict()[name]), name


@pytest.mark.parametrize("supervised_checkpoint", [False, True])
def test_frozen_opponent_is_unchanged_by_ppo_training(tmp_path, supervised_checkpoint):
    from skat_rl.training.train_supervised import save_pretrained

    torch.set_num_threads(1)
    source = train_torch_ppo.PPOAgent(
        _small_transformer_config(transformer_dropout=0.5, use_belief=True), device="cpu",
    )
    path = tmp_path / "opponent.pt"
    if supervised_checkpoint:
        save_pretrained(source, path, epoch=1)
    else:
        source.save(path)
    learner = train_torch_ppo.PPOAgent(
        _small_transformer_config(transformer_dim=32), device="cpu",
    )
    opponent = train_torch_ppo.load_frozen_opponent(path, learner.config, "cpu")
    assert opponent.config.transformer_dim == 16
    assert not opponent.model.training
    assert all(not p.requires_grad for p in opponent.model.parameters())
    before = {name: value.clone() for name, value in opponent.model.state_dict().items()}
    learner_before = {name: value.clone() for name, value in learner.model.state_dict().items()}
    env = train_torch_ppo.SkatCppBatchedSingleAgentEnv(4)
    rollout, episodes, steps = train_torch_ppo._collect_cpp_batched_rollout(
        learner, env, 0, opponent,
    )
    assert steps == 40
    assert len(episodes) == 4
    assert rollout.belief_targets is None
    metrics = learner.update(rollout)
    assert np.isfinite(metrics["loss"])
    assert all(torch.equal(before[n], p) for n, p in opponent.model.state_dict().items())
    assert all(p.grad is None for p in opponent.model.parameters())
    assert not opponent.model.training
    assert any(not torch.equal(learner_before[n], p) for n, p in learner.model.state_dict().items())


@pytest.mark.parametrize("arguments,expected", [
    ([], None),
    (["--env", "python"], None),
    (["--fixed-declarer", "0"], 0),
    (["--fixed-declarer", "1"], 1),
    (["--opponent-model", "frozen.pt"], None),
    (["--opponent-model", "frozen.pt", "--fixed-declarer", "2"], 2),
    (["--fixed-declarer", "-1"], -1),
])
def test_declarer_cli_defaults_and_overrides(monkeypatch, arguments, expected):
    monkeypatch.setattr(sys, "argv", ["train_torch_ppo", *arguments])
    assert train_torch_ppo._parse_args().fixed_declarer == expected


def test_python_env_rejects_neural_opponents(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "train_torch_ppo", "--env", "python", "--opponent-model", "frozen.pt",
    ])
    with pytest.raises(ValueError, match="requires --env cpp"):
        train_torch_ppo.main()


def test_cpp_training_requires_neural_opponent_before_creating_run(monkeypatch, tmp_path):
    output = tmp_path / "runs"
    monkeypatch.setattr(sys, "argv", ["train_torch_ppo", "--env", "cpp", "--output-dir", str(output)])
    with pytest.raises(ValueError, match="--env cpp requires --opponent-model"):
        train_torch_ppo.main()
    assert not output.exists()


def test_opponent_checkpoint_rejects_incompatible_dimensions(tmp_path):
    path = tmp_path / "bad.pt"
    train_torch_ppo.PPOAgent(train_torch_ppo.PPOConfig(
        action_dim=2, hidden_sizes=(8,),
    ), device="cpu").save(path)
    with pytest.raises(ValueError, match="action dimension does not match"):
        train_torch_ppo.load_frozen_opponent(path, _small_transformer_config(), "cpu")


def test_fixed_opponent_cli_trains_and_saves_run(tmp_path, monkeypatch):
    from skat_rl.training.train_supervised import save_pretrained

    torch.set_num_threads(1)
    pretrained = train_torch_ppo.PPOAgent(_small_transformer_config(), device="cpu")
    path = tmp_path / "pretrained.pt"
    save_pretrained(pretrained, path, epoch=1)
    original = path.read_bytes()
    monkeypatch.setattr(sys, "argv", [
        "train_torch_ppo", "--init-model", str(path), "--opponent-model", str(path),
        "--env", "cpp", "--rollout-size", "2", "--total-timesteps", "40",
        "--update-epochs", "1", "--minibatch-size", "20", "--device", "cpu",
        "--output-dir", str(tmp_path / "runs"),
    ])
    train_torch_ppo.main()
    run = next((tmp_path / "runs").iterdir())
    assert (run / "model.pt").is_file()
    assert path.read_bytes() == original
    config = json.loads((run / "config.json").read_text())
    assert config["args"]["fixed_declarer"] is None
    assert config["args"]["opponent_model"] == str(path)
    with (run / "metrics.csv").open() as handle:
        metrics = list(csv.DictReader(handle))
    assert [int(row["total_timesteps"]) for row in metrics] == [20, 40]
    assert all(np.isfinite(float(row["loss"])) for row in metrics)
    with (run / "episodes.csv").open() as handle:
        episodes = list(csv.DictReader(handle))
    assert len(episodes) == 4
    assert all(int(row["length"]) == 10 for row in episodes)


@pytest.mark.parametrize("architecture", ["mlp", "transformer"])
def test_python_ppo_cli_trains_with_structured_observations(tmp_path, monkeypatch, architecture):
    torch.set_num_threads(1)
    monkeypatch.setattr(sys, "argv", [
        "train_torch_ppo", "--env", "python", "--n-envs", "2", "--rollout-steps", "12",
        "--total-timesteps", "48", "--architecture", architecture, "--belief",
        "--transformer-dim", "16", "--transformer-layers", "1", "--transformer-heads", "2",
        "--transformer-ff-dim", "32", "--hidden-sizes", "16",
        "--update-epochs", "1", "--minibatch-size", "8", "--device", "cpu",
        "--output-dir", str(tmp_path),
    ])
    if architecture == "mlp":
        sys.argv.remove("--belief")
    train_torch_ppo.main()
    run = next(tmp_path.iterdir())
    agent = train_torch_ppo.PPOAgent.load(run / "model.pt", device="cpu")
    assert agent.config.architecture == architecture
    with (run / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(np.isfinite(float(row["loss"])) for row in rows)
    assert (run / "episodes.csv").exists()
