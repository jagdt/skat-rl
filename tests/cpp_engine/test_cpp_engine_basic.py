import random

from skat_rl._skat_cpp import FastSkatGame


def test_cpp_extension_imports_and_resets():
    game = FastSkatGame()
    game.reset(1)

    summary = game.state_summary()
    assert len(summary["hands"][0]) == 10
    assert len(summary["hands"][1]) == 10
    assert len(summary["hands"][2]) == 10
    assert len(summary["skat"]) == 2
    assert not summary["terminated"]


def test_legal_mask_contains_only_current_player_hand_cards():
    game = FastSkatGame()
    game.reset(1)

    hand = set(game.hand(game.current_player()))
    legal_actions = set(game.legal_actions())
    mask = game.legal_mask_array()

    assert legal_actions
    assert legal_actions <= hand
    assert len(mask) == 32
    assert {card for card, is_legal in enumerate(mask) if is_legal} == legal_actions


def test_playing_legal_card_removes_it_from_hand():
    game = FastSkatGame()
    game.reset(1)

    player = game.current_player()
    action = game.legal_actions()[0]
    game.step(action)

    assert action not in game.hand(player)


def test_random_cpp_game_terminates_after_30_card_plays():
    rng = random.Random(1)
    game = FastSkatGame()
    game.reset(1)

    plays = 0
    result = None
    while not game.is_terminal():
        action = rng.choice(game.legal_actions())
        result = game.step(action)
        plays += 1

    assert plays == 30
    assert game.trick_index() == 10
    if game.state_summary()["game_kind"] != 2:
        assert result["declarer_points"] + result["defender_points"] == 120
