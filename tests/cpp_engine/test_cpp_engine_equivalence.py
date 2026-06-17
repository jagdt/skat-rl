import random

from skat_rl._skat_cpp import FastSkatGame
from skat_rl.engine.game import SkatGame
from skat_rl.engine.rules import points_won_by_player
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


def test_cpp_matches_python_engine_for_seeded_deals():
    for seed in range(100):
        rng = random.Random(seed)
        hands, skat, declarer, game_type, current_player = _deal_from_python_seed(seed)
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

        py_result = py_result.info["result"]
        cpp_terminal = cpp_result
        assert cpp_terminal["declarer_won"] == py_result["declarer_won"]
        if game_type.kind != GameKind.NULL:
            assert cpp_terminal["declarer_points"] == py_result["declarer_points"]
            assert cpp_terminal["defender_points"] == py_result["defender_points"]
