import argparse
import json

import numpy as np
import pytest

pytest.importorskip("torch")

from skat_rl.training import train_torch_ppo  # noqa: E402


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

        assert observation.shape == env.observation_space.shape
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
        observation_dim=5,
        action_dim=4,
        hidden_sizes=(16, 8),
    )

    train_torch_ppo._save_config(tmp_path, args, config)

    with open(tmp_path / "config.json", encoding="utf-8") as config_file:
        payload = json.load(config_file)

    assert payload["args"]["total_timesteps"] == 128
    assert payload["args"]["n_envs"] == 2
    assert payload["args"]["rollout_size"] == 4
    assert payload["ppo"]["observation_dim"] == 5
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
                [1.0, 0.0],
                [0.5, 0.5],
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
                [0.0, 1.0],
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

    assert rollout.observations.shape == (3, 2)
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
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_sizes=(8,),
    )
    agent = train_torch_ppo.PPOAgent(config, device="cpu")

    rollout, episodes, global_step = train_torch_ppo._collect_cpp_batched_rollout(
        agent,
        env,
        global_step=0,
    )

    assert global_step == 20
    assert [episode["length"] for episode in episodes] == [10, 10]
    assert rollout.observations.shape == (20, 1149)
    assert rollout.action_masks.shape == (20, 32)
    assert rollout.belief_targets.shape == (20, 32)
    assert set(np.unique(rollout.belief_targets)) <= {-1, 0, 1, 2}
