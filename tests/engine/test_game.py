import pytest

from skat_rl.engine.cards import Rank, Suit, full_deck, make_card
from skat_rl.engine.game import SkatGame
from skat_rl.engine.state import GameKind, GameType


def test_reset_deals_all_cards_and_initializes_state():
    game = SkatGame(seed=1)
    state = game.reset(seed=1)
    all_dealt_cards = set().union(*state.hands, set(state.skat))

    assert len(state.hands) == 3
    assert all(len(hand) == 10 for hand in state.hands)
    assert len(state.skat) == 2
    assert all_dealt_cards == set(full_deck())
    assert state.current_player == 0
    assert not state.terminated


def test_observe_and_legal_actions_require_reset():
    game = SkatGame()

    with pytest.raises(RuntimeError, match="not been reset"):
        game.observe(0)

    with pytest.raises(RuntimeError, match="not been reset"):
        game.legal_actions()


def test_legal_actions_empty_for_non_current_player():
    game = SkatGame(seed=1)
    game.reset(seed=1)

    assert game.legal_actions(player=1) == []


def test_step_rejects_illegal_action():
    game = SkatGame(seed=1)
    game.reset(seed=1)
    illegal_card = next(card for card in full_deck() if card not in game.state.hands[0])

    with pytest.raises(ValueError, match="Illegal action"):
        game.step(illegal_card)


def test_step_completes_trick_and_assigns_points_to_winner():
    game = SkatGame(game_type=GameType(GameKind.GRAND), declarer=0)
    game.reset(game_type=GameType(GameKind.GRAND), declarer=0, seed=1)

    ace = make_card(Suit.CLUBS, Rank.ACE)
    ten = make_card(Suit.CLUBS, Rank.TEN)
    seven = make_card(Suit.CLUBS, Rank.SEVEN)
    game.state.hands = [
        {ace},
        {ten},
        {seven},
    ]
    game.state.current_player = 0

    assert game.step(ace).info == {}
    assert game.state.current_player == 1
    assert game.step(ten).info == {}
    result = game.step(seven)

    assert result.info["trick_winner"] == 0
    assert result.info["trick_points"] == 21
    assert game.state.trick_winners == [0]
    assert game.state.won_cards[0] == [ace, ten, seven]
    assert len(game.state.completed_tricks) == 1
    assert game.state.current_player == 0


def test_terminal_reward_marks_declarer_win_and_loss():
    game = SkatGame()

    assert game._terminal_reward({"declarer": 1, "declarer_won": True}) == [
        -0.5,
        1.0,
        -0.5,
    ]
    assert game._terminal_reward({"declarer": 1, "declarer_won": False}) == [
        0.5,
        -1.0,
        0.5,
    ]
