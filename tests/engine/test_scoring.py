import pytest

from skat_rl.engine.scoring import game_value, tournament_rewards
from skat_rl.engine.state import GameKind, GameType


@pytest.mark.parametrize("won,expected", [(True, [0.0, .98, 0.0]),
                                         (False, [.4, -1.46, .4])])
def test_three_player_tournament_score(won, expected):
    assert tournament_rewards(1, won, 48) == pytest.approx(expected)


@pytest.mark.parametrize("hand,schneider,schwarz,value", [
    (False, False, False, 24), (True, False, False, 36),
    (False, True, False, 36), (False, True, True, 48),
    (True, True, True, 60),
])
def test_suit_game_levels(hand, schneider, schwarz, value):
    assert game_value(GameType(GameKind.SUIT, 0, hand), [7], schneider, schwarz) == value


def test_matadors_include_skat_and_continue_into_trump_suit():
    clubs = GameType(GameKind.SUIT, 0)
    assert game_value(clubs, [15]) == 24  # Without one.
    assert game_value(clubs, [7, 15]) == 36  # Club jack can be in the Skat.
    assert game_value(clubs, [31]) == 48  # Without three.
    assert game_value(clubs, [7, 15, 23, 31, 6, 5]) == 84  # With six.
    assert game_value(GameType(GameKind.GRAND), [7, 15, 23, 31, 6, 5]) == 120


@pytest.mark.parametrize("hand,expected", [(False, 23), (True, 35)])
def test_null_has_fixed_game_value(hand, expected):
    assert game_value(GameType(GameKind.NULL, hand=hand), [], True, True) == expected
