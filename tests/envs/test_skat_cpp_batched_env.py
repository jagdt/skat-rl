import numpy as np
import pytest

from skat_rl._skat_cpp import FastSkatGame
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


@pytest.mark.parametrize("learning_player", [0, 1, 2])
def test_external_turns_match_individual_games_for_every_player(learning_player):
    env = SkatCppBatchedSingleAgentEnv(
        rollout_size=12, learning_player=learning_player, seed=17,
        autoplay_opponents=False,
    )
    games = []
    for seed in env._env_seeds(17):
        game = FastSkatGame()
        game.reset(seed)
        games.append(game)
    state = env.reset(seed=17)
    assert state["current_players"].tolist() == [0] * 12
    assert set(state["declarers"]) == {0, 1, 2}
    seen_mixed_turns = False
    terminal_actors = set()

    for turn in range(30):
        seen_mixed_turns |= len(set(state["current_players"])) > 1
        actions = []
        expected_rewards = []
        for row, game in enumerate(games):
            player = game.current_player()
            assert state["current_players"][row] == player
            assert state["declarers"][row] == game.declarer()
            np.testing.assert_allclose(state["observations"][row], game.observation(player))
            np.testing.assert_array_equal(state["action_masks"][row], game.legal_mask_array())
            np.testing.assert_array_equal(state["belief_targets"][row], game.belief_targets(player))
            # The hand plane must never contain the other seats' private cards.
            assert set(np.flatnonzero(state["observations"][row, :32])) == set(game.hand(player))
            action = game.legal_actions()[0]
            actions.append(action)
            info = game.step(action)
            reward = 0.0
            if info["terminated"]:
                terminal_actors.add(player)
                reward = (1.0 if info["declarer_won"] else -1.0)
                reward += 0.2 * (info["declarer_points"] - 60) / 60
                if learning_player != game.declarer():
                    reward /= -2
            expected_rewards.append(reward)
        state = env.step(np.asarray(actions))
        np.testing.assert_allclose(state["rewards"], expected_rewards, atol=1e-6)
        assert state["terminated"].tolist() == [turn == 29] * 12
        if turn < 29:
            assert state["active_indices"].tolist() == list(range(12))
            assert len(state["completed_env_indices"]) == 0

    assert seen_mixed_turns
    assert terminal_actors - {learning_player}
    assert state["completed_env_indices"].tolist() == list(range(12))
    assert state["completed_lengths"].tolist() == [10] * 12
    np.testing.assert_allclose(state["completed_returns"], expected_rewards, atol=1e-6)
    assert env.active_count() == 0
    assert state["observations"].shape == (0, 1149)
    assert state["action_masks"].shape == (0, 32)
    assert state["belief_targets"].shape == (0, 32)
    assert state["current_players"].shape == (0,)
    assert state["declarers"].shape == (0,)
    assert env.step([])["env_indices"].shape == (0,)
    reset = env.reset(seed=17)
    assert reset["current_players"].tolist() == [0] * 12
    np.testing.assert_allclose(reset["observations"], env.reset(seed=17)["observations"])


@pytest.mark.parametrize("autoplay", [True, False])
def test_invalid_batch_does_not_partially_step_games(autoplay):
    env = SkatCppBatchedSingleAgentEnv(3, autoplay_opponents=autoplay)
    state = env.reset(seed=42)
    actions = state["action_masks"].argmax(axis=1)
    actions[-1] = np.flatnonzero(~state["action_masks"][-1])[0]
    with pytest.raises(ValueError, match="illegal action"):
        env.step(actions)
    np.testing.assert_array_equal(env._state()["observations"], state["observations"])
    for invalid in ([0], [[0, 1, 2]], [0.5, 1.5, 2.5], [-1, 0, 0], [32, 0, 0]):
        with pytest.raises(ValueError):
            env.step(invalid)
        np.testing.assert_array_equal(env._state()["observations"], state["observations"])


@pytest.mark.parametrize("size", [0, -1])
def test_batch_size_must_be_positive(size):
    with pytest.raises(ValueError, match="at least 1"):
        SkatCppBatchedSingleAgentEnv(size)
