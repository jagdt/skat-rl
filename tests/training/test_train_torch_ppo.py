import argparse
import json

import pytest

pytest.importorskip("torch")

from skat_rl.training import train_torch_ppo  # noqa: E402


def test_env_seed_spaces_parallel_environments_apart():
    assert train_torch_ppo._env_seed(42, 0) == 42
    assert train_torch_ppo._env_seed(42, 2) == 200_000_042
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
    assert payload["ppo"]["observation_dim"] == 5
    assert payload["ppo"]["action_dim"] == 4
    assert payload["ppo"]["hidden_sizes"] == [16, 8]
