import pytest

from skat_rl.engine.cards import Rank, Suit, make_card
from skat_rl.engine.rules import (
    card_strength_in_trick,
    effective_suit,
    game_result,
    is_trump,
    legal_moves,
    points_won_by_player,
    trick_points,
    trick_winner,
)
from skat_rl.engine.state import GameKind, GameType, Trick


def test_is_trump_for_suit_game_grand_and_null():
    club_jack = make_card(Suit.CLUBS, Rank.JACK)
    heart_ace = make_card(Suit.HEARTS, Rank.ACE)
    spade_ace = make_card(Suit.SPADES, Rank.ACE)

    suit_game = GameType(GameKind.SUIT, trump_suit=Suit.HEARTS)
    grand_game = GameType(GameKind.GRAND)
    null_game = GameType(GameKind.NULL)

    assert is_trump(club_jack, suit_game)
    assert is_trump(heart_ace, suit_game)
    assert not is_trump(spade_ace, suit_game)
    assert is_trump(club_jack, grand_game)
    assert not is_trump(heart_ace, grand_game)
    assert not is_trump(club_jack, null_game)


def test_suit_game_requires_trump_suit():
    with pytest.raises(ValueError, match="trump_suit"):
        is_trump(make_card(Suit.CLUBS, Rank.ACE), GameType(GameKind.SUIT))


def test_effective_suit_and_legal_moves_follow_trump():
    game_type = GameType(GameKind.SUIT, trump_suit=Suit.HEARTS)
    lead = make_card(Suit.CLUBS, Rank.JACK)
    hand = {
        make_card(Suit.HEARTS, Rank.ACE),
        make_card(Suit.SPADES, Rank.ACE),
        make_card(Suit.DIAMONDS, Rank.SEVEN),
    }
    trick = Trick(leader=0, cards=[(0, lead)])

    assert effective_suit(lead, game_type) == "TRUMP"
    assert legal_moves(hand, trick, game_type) == [make_card(Suit.HEARTS, Rank.ACE)]


def test_legal_moves_follow_lead_suit_when_possible():
    game_type = GameType(GameKind.GRAND)
    lead = make_card(Suit.SPADES, Rank.SEVEN)
    spade_ace = make_card(Suit.SPADES, Rank.ACE)
    club_ace = make_card(Suit.CLUBS, Rank.ACE)
    trick = Trick(leader=0, cards=[(0, lead)])

    assert legal_moves({spade_ace, club_ace}, trick, game_type) == [spade_ace]


def test_card_strength_and_trick_winner_in_suit_game():
    game_type = GameType(GameKind.SUIT, trump_suit=Suit.HEARTS)
    trick = Trick(
        leader=0,
        cards=[
            (0, make_card(Suit.SPADES, Rank.ACE)),
            (1, make_card(Suit.SPADES, Rank.TEN)),
            (2, make_card(Suit.HEARTS, Rank.SEVEN)),
        ],
    )

    assert card_strength_in_trick(trick.cards[2][1], trick.lead_card(), game_type) > 50
    assert trick_winner(trick, game_type) == 2


def test_trick_winner_requires_complete_trick():
    with pytest.raises(ValueError, match="incomplete"):
        trick_winner(Trick(leader=0, cards=[(0, make_card(Suit.CLUBS, Rank.ACE))]), GameType(GameKind.GRAND))


def test_trick_points_and_game_result_for_suit_game():
    won_cards = [
        [
            make_card(Suit.CLUBS, Rank.ACE),
            make_card(Suit.SPADES, Rank.ACE),
            make_card(Suit.HEARTS, Rank.ACE),
            make_card(Suit.DIAMONDS, Rank.ACE),
            make_card(Suit.CLUBS, Rank.TEN),
            make_card(Suit.SPADES, Rank.TEN),
            make_card(Suit.HEARTS, Rank.TEN),
        ],
        [],
        [],
    ]

    result = game_result(
        won_cards=won_cards,
        trick_winners=[0],
        declarer=0,
        game_type=GameType(GameKind.SUIT, trump_suit=Suit.CLUBS),
    )

    assert points_won_by_player(won_cards, 0) == 74
    assert result["declarer_won"]
    assert result["declarer_points"] == 74
    assert result["defender_points"] == 46


def test_null_game_result_declarer_wins_only_without_tricks():
    result = game_result(
        won_cards=[[], [], []],
        trick_winners=[1, 2],
        declarer=0,
        game_type=GameType(GameKind.NULL),
    )

    assert result["declarer_won"]
    assert result["schwarz"]


def test_trick_points():
    trick = Trick(
        leader=0,
        cards=[
            (0, make_card(Suit.CLUBS, Rank.ACE)),
            (1, make_card(Suit.SPADES, Rank.TEN)),
            (2, make_card(Suit.HEARTS, Rank.QUEEN)),
        ],
    )

    assert trick_points(trick) == 24
