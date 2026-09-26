import random

import pytest

from skat_rl._skat_cpp import FastSkatGame
from skat_rl.engine.cards import card_points
from skat_rl.engine.game import SkatGame
from skat_rl.engine.rules import points_won_by_player
from skat_rl.engine.scoring import tournament_rewards
from skat_rl.engine.state import GameKind, GameState, GameType, Trick


def _game_kind_to_int(game_kind):
    if game_kind == GameKind.SUIT:
        return 0
    if game_kind == GameKind.GRAND:
        return 1
    if game_kind == GameKind.NULL:
        return 2
    raise ValueError(game_kind)


def _python_game_from_deal(hands, skat, declarer, game_type, current_player=0):
    game = SkatGame()
    game.state = GameState(
        hands=[set(hand) for hand in hands],
        skat=list(skat),
        declarer=declarer,
        game_type=game_type,
        current_player=current_player,
        current_trick=Trick(leader=current_player),
    )
    return game


def _cpp_game_from_deal(hands, skat, declarer, game_type, current_player=0):
    game = FastSkatGame()
    trump_suit = -1 if game_type.trump_suit is None else int(game_type.trump_suit)
    game.reset_from_deal(
        [sorted(hand) for hand in hands],
        list(skat),
        declarer,
        _game_kind_to_int(game_type.kind),
        trump_suit,
        current_player,
        game_type.hand,
    )
    return game


def _deal_from_python_seed(seed):
    game = SkatGame(seed=seed)
    state = game.reset(seed=seed)
    return (
        [set(hand) for hand in state.hands],
        list(state.skat),
        state.declarer,
        state.game_type,
        state.current_player,
    )


@pytest.mark.parametrize("game_kind", [GameKind.SUIT, GameKind.GRAND, GameKind.NULL])
@pytest.mark.parametrize("hand_game", [False, True])
def test_cpp_matches_python_engine_for_seeded_deals(game_kind, hand_game):
    for seed in range(100):
        rng = random.Random(seed)
        hands, skat, declarer, game_type, current_player = _deal_from_python_seed(seed)
        if game_kind != GameKind.SUIT:
            game_type = GameType(game_kind)
        game_type.hand = hand_game
        py_game = _python_game_from_deal(hands, skat, declarer, game_type, current_player)
        cpp_game = _cpp_game_from_deal(hands, skat, declarer, game_type, current_player)

        while not py_game.state.terminated:
            py_legal = py_game.legal_actions()
            cpp_legal = cpp_game.legal_actions()
            assert cpp_legal == py_legal

            action = rng.choice(py_legal)
            py_result = py_game.step(action)
            cpp_result = cpp_game.step(action)

            assert cpp_result["terminated"] == py_result.terminated
            assert cpp_game.current_player() == py_game.state.current_player
            assert cpp_game.trick_index() == len(py_game.state.completed_tricks)
            assert cpp_game.declarer_points() == points_won_by_player(
                py_game.state.won_cards,
                declarer,
            )
            assert cpp_game.defender_points() == sum(
                points_won_by_player(py_game.state.won_cards, player)
                for player in range(3)
                if player != declarer
            )
            if not py_result.terminated:
                assert cpp_result["declarer_points"] == cpp_game.declarer_points()
                assert cpp_game.observation(cpp_game.current_player())[1132] == pytest.approx(
                    points_won_by_player(py_game.state.won_cards, declarer) / 120.0
                )

        terminal_rewards = py_result.reward
        py_result = py_result.info["result"]
        cpp_terminal = cpp_result
        assert cpp_terminal["declarer_won"] == py_result["declarer_won"]
        assert cpp_terminal["game_value"] == py_result["game_value"]
        assert terminal_rewards == pytest.approx(tournament_rewards(
            declarer, py_result["declarer_won"], cpp_terminal["game_value"],
        ))
        if game_type.kind != GameKind.NULL:
            assert cpp_terminal["declarer_points"] == py_result["declarer_points"]
            assert cpp_terminal["defender_points"] == py_result["defender_points"]
            expected_points = points_won_by_player(py_game.state.won_cards, declarer) + sum(
                card_points(card) for card in skat
            )
            assert py_result["declarer_points"] == expected_points
            assert py_result["defender_points"] == sum(
                points_won_by_player(py_game.state.won_cards, player)
                for player in range(3)
                if player != declarer
            )
            assert py_result["declarer_won"] == (expected_points > 60)
        else:
            assert cpp_terminal["declarer_points"] == py_result["declarer_points"] == 0
            assert cpp_terminal["defender_points"] == py_result["defender_points"] == 0
