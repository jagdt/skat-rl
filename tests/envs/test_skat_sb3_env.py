import numpy as np
import pytest

from skat_rl.engine.cards import Rank, Suit, make_card
from skat_rl.engine.state import GameKind, GameType, Trick
from skat_rl.envs.skat_sb3_env import SkatSingleAgentEnv


def test_observation_matches_new_vector_shape():
    env = SkatSingleAgentEnv(learning_player=0, fixed_declarer=0, seed=1)
    observation, _ = env.reset(seed=1)

    assert env.game.state.declarer == 0
    assert env.game._choose_declarer(env.game.state.hands) == 0
    assert observation.shape == env.observation_space.shape
    assert observation.shape == (1149,)
    assert observation.dtype == np.float32


def test_nonzero_learning_player_is_fixed_declarer():
    env = SkatSingleAgentEnv(learning_player=2, fixed_declarer=2, seed=1)
    env.reset(seed=1)

    assert env.game.state.declarer == 2
    assert env.game._choose_declarer(env.game.state.hands) == 2


def test_observation_encodes_ordered_history_current_trick_and_void_info():
    env = SkatSingleAgentEnv(learning_player=0, seed=1)
    env.game.reset(game_type=GameType(GameKind.GRAND), declarer=0, seed=1)

    club_ace = make_card(Suit.CLUBS, Rank.ACE)
    spade_ace = make_card(Suit.SPADES, Rank.ACE)
    heart_king = make_card(Suit.HEARTS, Rank.KING)
    diamond_ten = make_card(Suit.DIAMONDS, Rank.TEN)

    env.game.state.hands = [
        {club_ace, diamond_ten},
        {spade_ace},
        {heart_king},
    ]
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

    observation = env._get_observation()

    own_hand_start = 0
    history_cards_start = own_hand_start + 32
    history_players_start = history_cards_start + 10 * 3 * 32
    current_trick_start = history_players_start + 10 * 3 * 3
    current_player_start = current_trick_start + 32
    current_leader_start = current_player_start + 3
    declarer_start = current_leader_start + 3
    game_kind_start = declarer_start + 3
    trump_suit_start = game_kind_start + 3
    trick_number_start = trump_suit_start + 4
    trick_position_start = trick_number_start + 1
    points_start = trick_position_start + 1
    void_info_start = points_start + 2

    assert observation[own_hand_start + club_ace] == 1.0
    assert observation[own_hand_start + diamond_ten] == 1.0

    first_history_card = history_cards_start + club_ace
    second_history_player = history_players_start + 3 + 1
    current_trick_card = history_cards_start + (1 * 3 * 32) + diamond_ten
    assert observation[first_history_card] == 1.0
    assert observation[second_history_player] == 1.0
    assert observation[current_trick_card] == 1.0
    assert observation[current_trick_start + diamond_ten] == 1.0

    assert observation[current_player_start:current_player_start + 3].tolist() == [1.0, 0.0, 0.0]
    assert observation[current_leader_start:current_leader_start + 3].tolist() == [0.0, 1.0, 0.0]
    assert observation[declarer_start:declarer_start + 3].tolist() == [1.0, 0.0, 0.0]
    assert observation[game_kind_start:game_kind_start + 3].tolist() == [0.0, 1.0, 0.0]
    assert observation[trump_suit_start:trump_suit_start + 4].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert observation[trick_number_start] == pytest.approx(0.1)
    assert observation[trick_position_start] == pytest.approx(1.0 / 3.0)
    assert observation[points_start:points_start + 2].tolist() == pytest.approx(
        [11.0 / 120.0, 11.0 / 120.0]
    )

    void_info = observation[void_info_start:void_info_start + 15].reshape(3, 5)
    assert void_info[1, int(Suit.CLUBS)] == 1.0
    assert void_info[2, int(Suit.CLUBS)] == 1.0
