import numpy as np

from skat_rl.envs.skat_cpp_sb3_env import SkatCppSingleAgentEnv


def test_cpp_env_reset_mask_and_step():
    env = SkatCppSingleAgentEnv(
        learning_player=0,
        fixed_declarer=0,
        seed=1,
    )
    observation, _ = env.reset(seed=1)
    mask = env.action_masks()

    assert env.game.declarer() == 0
    assert observation.shape == env.observation_space.shape
    assert observation.shape == (1149,)
    assert observation.dtype == np.float32
    assert mask.dtype == bool
    assert mask.shape == (32,)
    assert mask.any()

    action = int(np.flatnonzero(mask)[0])
    observation, reward, terminated, truncated, info = env.step(action)

    assert observation.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert truncated is False
    assert isinstance(info, dict)


def test_cpp_env_nonzero_learning_player_is_supported():
    env = SkatCppSingleAgentEnv(
        learning_player=2,
        fixed_declarer=2,
        seed=1,
    )
    observation, _ = env.reset(seed=1)

    assert env.game.declarer() == 2
    assert observation.shape == (1149,)
