import numpy as np
import pytest

from skat_rl.engine.cards import Rank, Suit, make_card
from skat_rl.engine.state import GameKind, GameType, Trick
from skat_rl.envs.observations import CardStatus, Contract, build_observation
from skat_rl.envs.skat_python_env import SkatSingleAgentEnv


def test_observation_matches_structured_space():
    env = SkatSingleAgentEnv(learning_player=0, fixed_declarer=0, seed=1)
    observation, _ = env.reset(seed=1)

    assert env.game.state.declarer == 0
    assert env.game._choose_declarer(env.game.state.hands) == 0
    assert env.observation_space.contains(observation)
    assert observation["card_status"].shape == (32,)
    assert all(value.dtype == np.int16 for value in observation.values())


def test_nonzero_learning_player_is_fixed_declarer():
    env = SkatSingleAgentEnv(learning_player=2, fixed_declarer=2, seed=1)
    env.reset(seed=1)

    targets = env.belief_targets()
    next_opponent = (env.learning_player + 1) % 3
    previous_opponent = (env.learning_player + 2) % 3
    assert all(targets[card] == 0 for card in env.game.state.hands[next_opponent])
    assert all(targets[card] == 1 for card in env.game.state.hands[previous_opponent])
    assert all(targets[card] == 2 for card in env.game.state.skat)

    assert env.game.state.declarer == 2
    assert env.game._choose_declarer(env.game.state.hands) == 2


def test_observation_encodes_ordered_history_current_trick_and_void_info():
    env = SkatSingleAgentEnv(learning_player=0, seed=1)
    env.game.reset(game_type=GameType(GameKind.GRAND), declarer=0, seed=1)

    club_ace = make_card(Suit.CLUBS, Rank.ACE)
    spade_ace = make_card(Suit.SPADES, Rank.ACE)
    heart_king = make_card(Suit.HEARTS, Rank.KING)
    diamond_ten = make_card(Suit.DIAMONDS, Rank.TEN)

    own_card = make_card(Suit.CLUBS, Rank.SEVEN)
    env.game.state.hands = [{own_card}, set(), set()]
    env.game.state.won_cards = [[club_ace], [spade_ace], []]
    env.game.state.completed_tricks = [
        Trick(
            leader=0,
            cards=[
                (0, club_ace),
                (1, spade_ace),
                (2, heart_king),
            ],
        )
    ]
    env.game.state.current_trick = Trick(leader=1, cards=[(1, diamond_ten)])
    env.game.state.current_player = 0
    env._void_info_cache[1, int(Suit.CLUBS)] = 1.0
    env._void_info_cache[2, int(Suit.CLUBS)] = 1.0

    observation = env._get_observation()

    assert observation["card_status"][own_card] == CardStatus.OWN
    for position, (player, card) in enumerate([
        (0, club_ace), (1, spade_ace), (2, heart_king), (1, diamond_ten),
    ]):
        assert observation["card_status"][card] == CardStatus.PLAYED
        assert observation["played_by"][card] == player
        assert observation["trick_index"][card] == position // 3
        assert observation["trick_slot"][card] == position % 3
    assert observation["relative_current_leader"] == 1
    assert observation["relative_declarer"] == 0
    assert observation["contract"] == Contract.GRAND
    assert observation["current_trick"] == 1
    assert observation["declarer_points"] == 11
    assert observation["defender_points"] == 11
    assert observation["void_info"][1, int(Suit.CLUBS)] == 1
    assert observation["void_info"][2, int(Suit.CLUBS)] == 1
    derived = build_observation(env.game.state, env.learning_player)
    np.testing.assert_array_equal(observation["void_info"], derived["void_info"])


def test_void_info_cache_updates_when_player_cannot_follow_lead_suit():
    env = SkatSingleAgentEnv(learning_player=0, seed=1)
    env.game.reset(game_type=GameType(GameKind.GRAND), declarer=0, seed=1)

    club_ace = make_card(Suit.CLUBS, Rank.ACE)
    spade_ace = make_card(Suit.SPADES, Rank.ACE)
    club_king = make_card(Suit.CLUBS, Rank.KING)

    env.game.state.current_trick = Trick(leader=0, cards=[(0, club_ace)])

    env._update_void_info_for_action(player=1, action=spade_ace)
    env._update_void_info_for_action(player=2, action=club_king)

    assert env._void_info_cache[1, int(Suit.CLUBS)] == 1.0
    assert env._void_info_cache[2, int(Suit.CLUBS)] == 0.0


def test_void_info_cache_resets_on_env_reset():
    env = SkatSingleAgentEnv(learning_player=0, fixed_declarer=0, seed=1)
    env._void_info_cache[1, int(Suit.CLUBS)] = 1.0

    env.reset(seed=1)

    assert not env._void_info_cache.any()
