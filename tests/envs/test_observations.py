from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from skat_rl.agents.ppo_agent import SkatObservationTokenizer, observation_to_tensors
from skat_rl.engine.cards import Suit
from skat_rl.engine.game import SkatGame
from skat_rl.engine.rules import effective_suit, points_won_by_player
from skat_rl.engine.state import GameKind, GameType
from skat_rl.envs.observations import (
    CardStatus, Contract, Phase, OBSERVATION_SPECS, UNPLAYED,
    build_observation, effective_suit_table, encode_belief_targets,
    index_observations, observation_space, stack_observations,
)
from skat_rl.envs.skat_python_env import SkatSingleAgentEnv


CONTRACTS = [GameType(GameKind.SUIT, suit) for suit in Suit] + [
    GameType(GameKind.GRAND), GameType(GameKind.NULL),
]


def assert_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for field in actual:
        np.testing.assert_array_equal(actual[field], expected[field], err_msg=field)


def reconstruct_history(observation):
    cards = np.flatnonzero(observation["card_status"] == CardStatus.PLAYED)
    cards = cards[np.lexsort((observation["trick_slot"][cards], observation["trick_index"][cards]))]
    return [(int(observation["played_by"][card]), int(card)) for card in cards]


def reconstruct_voids(observation):
    history = reconstruct_history(observation)
    suits = effective_suit_table()[observation["contract"]]
    voids = np.zeros((3, 5), dtype=np.int8)
    for start in range(0, len(history), 3):
        trick = history[start:start + 3]
        required = suits[trick[0][1]]
        for player, card in trick[1:]:
            if suits[card] != required:
                voids[player, required] = 1
    return voids


@pytest.mark.parametrize("game_type", CONTRACTS)
def test_complete_public_history_and_current_trick_are_preserved(game_type):
    game = SkatGame()
    game.reset(seed=19, declarer=2, game_type=game_type)
    rng = random.Random(19)
    space = observation_space()
    for move in range(31):
        state = game.state
        for player in range(3):
            obs = build_observation(state, player)
            assert space.contains(obs)
            assert set(obs) == set(OBSERVATION_SPECS)
            assert obs["phase"] == (Phase.TERMINAL if state.terminated else Phase.CARD_PLAY)
            assert obs["current_trick"] == max(0, move // 3 - int(state.terminated))
            played = obs["card_status"] == CardStatus.PLAYED
            own = obs["card_status"] == CardStatus.OWN
            assert int(played.sum()) == move
            assert set(np.flatnonzero(own)) == state.hands[player]
            positions = 3 * obs["trick_index"][played] + obs["trick_slot"][played]
            np.testing.assert_array_equal(np.sort(positions), np.arange(move))
            np.testing.assert_array_equal(np.sort(obs["trick_index"][played]), np.arange(move) // 3)
            for field in ("played_by", "trick_index", "trick_slot"):
                assert np.all(obs[field][~played] == UNPLAYED)
            tricks = state.completed_tricks + ([] if state.terminated else [state.current_trick])
            expected = [((actor - player) % 3, card) for trick in tricks for actor, card in trick.cards]
            assert reconstruct_history(obs) == expected
            current = played & (obs["trick_index"] == obs["current_trick"])
            assert set(np.flatnonzero(current)) == {card for _, card in state.current_trick.cards}
            assert obs["relative_declarer"] == (state.declarer - player) % 3
            assert obs["relative_current_leader"] == (state.current_trick.leader - player) % 3
            assert obs["declarer_points"] == points_won_by_player(state.won_cards, state.declarer)
            assert obs["defender_points"] == sum(points_won_by_player(state.won_cards, p)
                                                  for p in range(3) if p != state.declarer)
            np.testing.assert_array_equal(obs["void_info"], reconstruct_voids(obs))
        if state.terminated:
            break
        game.step(rng.choice(game.legal_actions()))


def rotate_seats(state, shift):
    rotated = deepcopy(state)
    for field in ("hands", "won_cards", "bid_status", "highest_called", "highest_held", "pass_threshold", "pass_role"):
        original = getattr(state, field)
        setattr(rotated, field, [deepcopy(original[(p - shift) % 3]) for p in range(3)])
    rotated.declarer = (state.declarer + shift) % 3 if state.declarer >= 0 else -1
    rotated.forehand = (state.forehand + shift) % 3
    rotated.auction_caller = (state.auction_caller + shift) % 3
    rotated.auction_holder = (state.auction_holder + shift) % 3
    rotated.current_player = (state.current_player + shift) % 3
    rotated.trick_winners = [(p + shift) % 3 for p in state.trick_winners]
    # A terminal current trick aliases the last completed trick in the engine.
    seen = set()
    for trick in rotated.completed_tricks + [rotated.current_trick]:
        if id(trick) not in seen:
            trick.leader = (trick.leader + shift) % 3
            trick.cards = [((p + shift) % 3, card) for p, card in trick.cards]
            seen.add(id(trick))
    return rotated


@pytest.mark.parametrize("shift", [1, 2])
def test_rotated_absolute_seats_produce_identical_observations_and_tokens(shift):
    game = SkatGame()
    game.reset(seed=31, declarer=1, game_type=GameType(GameKind.GRAND))
    for _ in range(17):
        game.step(game.legal_actions()[0])
    player = game.state.current_player
    rotated = rotate_seats(game.state, shift)
    original = build_observation(game.state, player)
    other = build_observation(rotated, (player + shift) % 3)
    assert_equal(original, other)
    tokenizer = SkatObservationTokenizer(16)
    tokens = tokenizer(observation_to_tensors(stack_observations([original, other]), "cpu"))
    torch.testing.assert_close(tokens[0], tokens[1])


@pytest.mark.parametrize("steps", [0, 2, 4, 5, 6, 7])
def test_full_game_observations_are_rotation_invariant(steps):
    game = SkatGame()
    game.reset(seed=31, full_game=True, forehand=0)
    for action in [1, 1, 0, 0, 0, 0, 0][:steps]:
        game.step(action)
    player = game.state.current_player
    original = build_observation(game.state, player)
    for shift in (1, 2):
        other = build_observation(rotate_seats(game.state, shift), (player + shift) % 3)
        assert_equal(original, other)


@pytest.mark.parametrize("player", [0, 1, 2])
def test_hidden_hands_and_skat_change_labels_but_not_policy_observation(player):
    game = SkatGame()
    game.reset(seed=11, declarer=player)
    for _ in range(8):
        game.step(game.legal_actions()[0])
    state = game.state
    changed = deepcopy(state)
    left, right = (player + 1) % 3, (player + 2) % 3
    left_card, right_card = min(changed.hands[left]), min(changed.hands[right])
    skat_card = changed.skat[0]
    changed.hands[left].remove(left_card)
    changed.hands[left].add(skat_card)
    changed.hands[right].remove(right_card)
    changed.hands[right].add(left_card)
    changed.skat[0] = right_card
    original = build_observation(state, player)
    assert_equal(original, build_observation(changed, player))
    assert not np.array_equal(encode_belief_targets(state, player), encode_belief_targets(changed, player))
    targets = encode_belief_targets(state, player)
    np.testing.assert_array_equal(original["card_status"] == CardStatus.UNKNOWN, targets >= 0)
    assert "belief_targets" not in original
    assert "current_player" not in original


@pytest.mark.parametrize("player", [0, 1, 2])
def test_cached_voids_match_history_for_every_python_env_step(player):
    env = SkatSingleAgentEnv(learning_player=player, seed=42)
    obs, _ = env.reset(seed=42)
    for _ in range(10):
        assert_equal(obs, build_observation(env.game.state, player))
        np.testing.assert_array_equal(obs["void_info"], reconstruct_voids(obs))
        obs, _, _, _, _ = env.step(np.flatnonzero(env.action_masks())[0])
    assert_equal(obs, build_observation(env.game.state, player))
    assert obs["phase"] == Phase.TERMINAL
    assert env.observation_space.contains(obs)


def test_effective_suit_lookup_uses_engine_rules_for_every_card_and_contract():
    table = effective_suit_table()
    assert table.shape == (len(Contract), 32)
    for contract, game_type in enumerate(CONTRACTS):
        for card in range(32):
            suit = effective_suit(card, game_type)
            assert table[contract, card] == (4 if suit == "TRUMP" else int(suit))


def test_empty_batches_keep_field_shapes_and_compact_integer_storage():
    batch = stack_observations([])
    for field, (shape, _, _) in OBSERVATION_SPECS.items():
        assert batch[field].shape == (0,) + shape
        assert batch[field].dtype == np.int16
    game = SkatGame()
    obs = build_observation(game.reset(seed=1), 0)
    assert sum(value.nbytes for value in obs.values()) == 340
    batch = stack_observations([obs, obs])
    assert_equal(index_observations(batch, 1), obs)
    assert_equal(index_observations(batch, np.array([1, 0])), batch)
