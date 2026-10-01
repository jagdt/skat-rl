from copy import deepcopy
import random

import numpy as np
import pytest

from skat_rl._skat_cpp import FastSkatGame, bid_values
from skat_rl.engine.actions import BID_VALUES, Contract, DISCARD_PAIRS, discard_action, discard_pair
from skat_rl.engine.game import SkatGame
from skat_rl.engine.state import AuctionRole, BiddingStatus, Phase
from skat_rl.envs.observations import CardStatus, build_observation, encode_belief_targets, observation_space
from skat_rl.envs.skat_cpp_batched_env import SkatCppBatchedSingleAgentEnv
from skat_rl.envs.skat_python_env import SkatSingleAgentEnv


def games(seed=17, forehand=0):
    python = SkatGame()
    state = python.reset(seed=seed, full_game=True, forehand=forehand)
    cpp = FastSkatGame()
    cpp.reset_full_from_deal([sorted(h) for h in state.hands], state.skat, forehand)
    return python, cpp


@pytest.mark.parametrize("full_game", [False, True])
def test_reset_after_termination_restores_live_phase_in_both_engines(full_game):
    python, cpp = games()
    for _ in range(3):
        python.step(0)
        cpp.step(0)
    assert python.state.terminated and cpp.is_terminal()
    assert python.legal_actions() == cpp.legal_actions() == []
    assert not any(cpp.legal_mask_array())
    assert cpp.legal_mask_bits() == 0
    for game in (python, cpp):
        with pytest.raises(RuntimeError, match="terminated"):
            game.step(0)
    python.reset(seed=17, full_game=full_game)
    if full_game:
        cpp.reset_full(17)
    else:
        cpp.reset(17)
    phase = Phase.BIDDING if full_game else Phase.CARD_PLAY
    assert python.state.phase == cpp.phase() == phase
    assert not python.state.terminated and not cpp.is_terminal()
    assert python.legal_actions() and cpp.legal_actions()


def assert_equal(python, cpp):
    state = python.state
    assert cpp.phase() == state.phase
    assert cpp.current_player() == state.current_player
    assert cpp.declarer() == state.declarer
    assert cpp.is_terminal() == state.terminated
    assert cpp.winning_bid() == state.winning_bid
    assert cpp.legal_actions() == sorted(python.legal_actions())
    assert cpp.skat() == state.skat
    for player in range(3):
        assert set(cpp.hand(player)) == state.hands[player]
        expected = build_observation(state, player)
        assert "winning_bid" not in expected
        assert observation_space().contains(expected)
        actual = cpp.observation(player)
        assert actual.keys() == expected.keys()
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        np.testing.assert_array_equal(cpp.belief_targets(player), encode_belief_targets(state, player))


def step_both(python, cpp, action):
    result = python.step(action)
    native = cpp.step(action)
    assert_equal(python, cpp)
    if result.terminated and not native["passed_out"]:
        for key in ("declarer_won", "declarer_points", "defender_points", "game_value", "overbid"):
            assert native[key] == result.info["result"][key], key
    return result


@pytest.mark.parametrize("contract", list(Contract))
@pytest.mark.parametrize("hand", [False, True])
@pytest.mark.parametrize("forehand", [0, 1, 2])
def test_full_game_engine_parity_every_phase_and_contract(contract, hand, forehand):
    python, cpp = games(seed=17 + int(contract), forehand=forehand)
    assert_equal(python, cpp)
    # Middlehand calls 18; forehand and then rearhand pass.
    for action in (1, 0, 0, int(hand)):
        step_both(python, cpp, action)
    if not hand:
        step_both(python, cpp, 65)
    step_both(python, cpp, int(contract))
    rng = random.Random(31)
    while not python.state.terminated:
        step_both(python, cpp, rng.choice(python.legal_actions()))


def test_auction_tracks_calls_holds_passes_and_persists_after_bidding():
    python, cpp = games()
    for action in (1, 1, 1, 0, 1, 0):
        step_both(python, cpp, action)
    state = python.state
    assert state.phase == Phase.PICKUP_DECISION
    assert state.declarer == 2 and state.winning_bid == 22
    assert state.highest_called == [-1, 20, 22]
    assert state.highest_held == [18, -1, -1]
    assert state.pass_threshold == [20, 22, -1]
    assert state.pass_role == [AuctionRole.HOLDER, AuctionRole.HOLDER, -1]
    assert state.bid_status == [BiddingStatus.PASSED, BiddingStatus.PASSED, BiddingStatus.ACTIVE]
    before = build_observation(state, 2)
    for action in (0, 7, int(Contract.HEARTS)):
        step_both(python, cpp, action)
    after = build_observation(state, 2)
    for key in ("bid_status", "highest_called", "highest_held", "pass_threshold", "pass_role"):
        np.testing.assert_array_equal(after[key], before[key])


@pytest.mark.parametrize("accept", [False, True])
def test_forehand_final_offer_and_passed_out_deal(accept):
    python, cpp = games()
    for action in (0, 0, int(accept)):
        result = step_both(python, cpp, action)
    assert python.state.terminated != accept
    assert python.state.highest_called == [-1] * 3
    assert result.reward == [0.0] * 3
    if accept:
        assert python.state.declarer == 0
        assert python.state.winning_bid == 18
    else:
        assert result.info["passed_out"]
        assert python.state.declarer == -1


def test_bid_ladder_ceiling_finishes_without_inventing_passes_and_overbid_loses():
    assert list(BID_VALUES) == bid_values()
    python, cpp = games()
    while python.state.phase == Phase.BIDDING:
        step_both(python, cpp, 1)
    assert python.state.winning_bid == 264
    assert python.state.declarer == 0
    assert python.state.pass_threshold == [-1] * 3
    assert python.state.bid_status[2] == BiddingStatus.NOT_ENTERED
    step_both(python, cpp, 1)
    assert int(Contract.NULL) not in python.legal_actions()
    step_both(python, cpp, int(Contract.DIAMONDS))
    while not python.state.terminated:
        result = step_both(python, cpp, python.legal_actions()[0])
    assert result.info["result"]["overbid"]
    assert result.info["result"]["game_value"] == 270
    assert result.reward == pytest.approx([-5.9, .4, .4])


def test_discard_pairs_are_canonical_and_known_only_to_declarer():
    python, cpp = games()
    for action in (1, 0, 0, 0):
        step_both(python, cpp, action)
    state = python.state
    own = sorted(state.hands[state.declarer])
    assert len(own) == 12 and state.skat == []
    assert len(DISCARD_PAIRS) == 66
    pairs = [discard_pair(own, action) for action in range(66)]
    assert len(set(pairs)) == 66 and all(a < b for a, b in pairs)
    assert [discard_action(reversed(own), pair) for pair in pairs] == list(range(66))
    for action, pair in enumerate(pairs):
        clone = deepcopy(python)
        clone.step(action)
        assert set(clone.state.hands[state.declarer]) == set(own) - set(pair)
        assert clone.state.skat == list(pair)
    step_both(python, cpp, 19)
    for player in range(3):
        obs = build_observation(state, player)
        targets = encode_belief_targets(state, player)
        known = player == state.declarer
        assert np.all(obs["card_status"][state.skat] == (CardStatus.KNOWN_DISCARD if known else CardStatus.UNKNOWN))
        assert np.all(targets[state.skat] == (-1 if known else 2))
        assert obs["contract"] == -1


def test_hidden_information_does_not_change_preplay_observations():
    python, _ = games()
    for actions in ((), (1, 0, 0), (0,)):
        for action in actions:
            python.step(action)
        state = python.state
        player = state.current_player
        changed = deepcopy(state)
        left, right = (player + 1) % 3, (player + 2) % 3
        a, b = min(changed.hands[left]), min(changed.hands[right])
        changed.hands[left].remove(a)
        changed.hands[right].remove(b)
        changed.hands[left].add(b)
        changed.hands[right].add(a)
        for key, value in build_observation(state, player).items():
            np.testing.assert_array_equal(value, build_observation(changed, player)[key])
        assert not np.array_equal(encode_belief_targets(state, player), encode_belief_targets(changed, player))


def test_batched_full_games_match_individual_games_with_mixed_phases():
    env = SkatCppBatchedSingleAgentEnv(12, learning_player=2, full_game=True)
    state = env.reset(seed=17)
    individual = [FastSkatGame() for _ in range(12)]
    for game, seed in zip(individual, env._env_seeds(17)):
        game.reset_full(seed)
    rng = np.random.default_rng(31)
    mixed_phases = False
    completed = set()
    for _ in range(400):
        indices = state["active_indices"]
        if not len(indices):
            break
        mixed_phases |= len(set(state["observations"]["phase"].tolist())) > 1
        actions = []
        for row, index in enumerate(indices):
            game = individual[index]
            assert state["current_players"][row] == game.current_player()
            for key, value in game.observation(game.current_player()).items():
                np.testing.assert_array_equal(state["observations"][key][row], value)
            np.testing.assert_array_equal(state["action_masks"][row], game.legal_mask_array())
            actions.append(int(rng.choice(game.legal_actions())))
        for index, action in zip(indices, actions):
            individual[index].step(action)
        state = env.step(actions)
        completed.update(state["completed_env_indices"].tolist())
    assert env.active_count() == 0
    assert mixed_phases and completed == set(range(12))
    assert state["action_masks"].shape == (0, 66)


def test_full_batch_validation_is_atomic_and_passed_out_rewards_are_zero():
    env = SkatCppBatchedSingleAgentEnv(4, full_game=True)
    state = env.reset(seed=1)
    before = {key: value.copy() for key, value in state["observations"].items()}
    with pytest.raises(ValueError, match="illegal action"):
        env.step([0, 0, 2, 0])
    for key, expected in before.items():
        np.testing.assert_array_equal(env.game.observations()[key], expected)
    for _ in range(3):
        state = env.step([0] * 4)
    assert env.active_count() == 0
    np.testing.assert_array_equal(state["completed_returns"], np.zeros(4))
    np.testing.assert_array_equal(state["completed_lengths"], np.ones(4))


@pytest.mark.parametrize("hand,maximum", [(False, 23), (True, 35)])
def test_null_contract_mask_respects_hand_and_winning_bid(hand, maximum):
    from skat_rl.engine.actions import legal_contracts
    assert int(Contract.NULL) in legal_contracts(maximum, hand)
    assert int(Contract.NULL) not in legal_contracts(maximum + 1, hand)


def test_native_full_game_does_not_use_card_only_setup():
    cpp = FastSkatGame()
    cpp.reset_full(42)
    assert cpp.phase() == Phase.BIDDING
    assert cpp.declarer() == -1 and cpp.game_kind() == -1
    assert cpp.legal_actions() == [0, 1]
    with pytest.raises(ValueError, match="bidding"):
        SkatCppBatchedSingleAgentEnv(4, fixed_declarer=0, full_game=True)


@pytest.mark.parametrize("player", [0, 1, 2])
def test_python_full_games_against_heuristic_are_gym_compatible(player):
    env = SkatSingleAgentEnv(learning_player=player, full_game=True)
    for seed in range(5):
        obs, _ = env.reset(seed=seed)
        for _ in range(300):
            assert env.observation_space.contains(obs)
            assert not env.game.state.terminated
            obs, reward, done, _, info = env.step(int(np.flatnonzero(env.action_masks())[0]))
            if done:
                assert env.observation_space.contains(obs)
                assert info.get("passed_out") or "result" in info
                assert np.isfinite(reward)
                break
        else:
            pytest.fail("Full game did not finish")
