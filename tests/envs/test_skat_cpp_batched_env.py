import numpy as np

from skat_rl.envs.skat_cpp_batched_env import SkatCppBatchedSingleAgentEnv


def test_cpp_batched_env_reset_shapes_and_masks():
    env = SkatCppBatchedSingleAgentEnv(
        rollout_size=4,
        learning_player=0,
        fixed_declarer=0,
        seed=1,
    )

    state = env.reset(seed=1)

    assert state["active_indices"].tolist() == [0, 1, 2, 3]
    assert state["observations"].shape == (4, 1149)
    assert state["observations"].dtype == np.float32
    assert state["action_masks"].shape == (4, 32)
    assert state["action_masks"].dtype == bool
    assert state["action_masks"].any(axis=1).all()
    assert state["belief_targets"].shape == (4, 32)
    assert state["belief_targets"].dtype == np.int64
    assert np.isin(state["belief_targets"], [-1, 0, 1, 2]).all()


def test_cpp_batched_env_derives_deterministic_distinct_child_seeds():
    env = SkatCppBatchedSingleAgentEnv(
        rollout_size=4,
        learning_player=0,
        fixed_declarer=0,
        seed=1,
    )
    seeds = env._env_seeds(42)

    assert seeds == env._env_seeds(42)
    assert len(set(seeds)) == 4
    assert seeds != [42, 43, 44, 45]
    assert all(0 <= seed < 2**64 for seed in seeds)


def test_cpp_batched_env_steps_all_active_games_until_done():
    env = SkatCppBatchedSingleAgentEnv(
        rollout_size=3,
        learning_player=0,
        fixed_declarer=0,
        seed=1,
    )
    state = env.reset(seed=1)
    total_steps = 0
    completed_returns = []
    completed_lengths = []

    while len(state["active_indices"]) > 0:
        actions = np.argmax(state["action_masks"], axis=1)
        result = env.step(actions)
        total_steps += len(result["env_indices"])
        completed_returns.extend(result["completed_returns"].tolist())
        completed_lengths.extend(result["completed_lengths"].tolist())
        state = {
            "active_indices": result["active_indices"],
            "observations": result["observations"],
            "action_masks": result["action_masks"],
            "belief_targets": result["belief_targets"],
        }

    assert total_steps == 30
    assert completed_lengths == [10, 10, 10]
    assert len(completed_returns) == 3
