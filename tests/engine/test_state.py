import pytest

from skat_rl.engine.cards import Rank, Suit, make_card
from skat_rl.engine.state import GameKind, GameState, GameType, Phase, Trick


def test_trick_lead_card_and_completion():
    first_card = make_card(Suit.CLUBS, Rank.ACE)
    trick = Trick(leader=1, cards=[(1, first_card)])

    assert not trick.is_complete()
    assert trick.lead_card() == first_card

    trick.cards.extend(
        [
            (2, make_card(Suit.SPADES, Rank.SEVEN)),
            (0, make_card(Suit.HEARTS, Rank.EIGHT)),
        ]
    )

    assert trick.is_complete()


def test_empty_trick_has_no_lead_card():
    with pytest.raises(ValueError, match="no lead card"):
        Trick(leader=0).lead_card()


@pytest.mark.parametrize("phase", list(Phase))
def test_termination_is_derived_from_phase_and_read_only(phase):
    state = GameState(
        hands=[set(), set(), set()], skat=[], declarer=-1, game_type=None,
        current_player=0, current_trick=Trick(leader=0), phase=phase,
    )
    assert state.terminated is (phase == Phase.TERMINAL)
    assert "terminated" not in vars(state)
    assert state.clone_public_for_player(0)["terminated"] is state.terminated
    with pytest.raises(AttributeError):
        state.terminated = not state.terminated
    state.phase = Phase.TERMINAL
    assert state.terminated
    state.phase = Phase.BIDDING
    assert not state.terminated


def test_clone_public_for_player_returns_public_view_and_own_hand():
    own_card = make_card(Suit.CLUBS, Rank.ACE)
    other_card = make_card(Suit.SPADES, Rank.TEN)
    completed_trick = Trick(leader=0, cards=[(0, own_card), (1, other_card), (2, make_card(Suit.HEARTS, Rank.SEVEN))])
    state = GameState(
        hands=[{own_card}, {other_card}, set()],
        skat=[make_card(Suit.DIAMONDS, Rank.QUEEN), make_card(Suit.DIAMONDS, Rank.KING)],
        declarer=0,
        game_type=GameType(GameKind.GRAND),
        current_player=1,
        current_trick=Trick(leader=1, cards=[(1, other_card)]),
        completed_tricks=[completed_trick],
        trick_winners=[0],
    )

    observation = state.clone_public_for_player(0)

    assert observation["player_id"] == 0
    assert observation["own_hand"] == [own_card]
    assert observation["declarer"] == 0
    assert observation["current_player"] == 1
    assert observation["current_trick"] == [(1, other_card)]
    assert observation["completed_tricks"] == [
        {"leader": 0, "cards": list(completed_trick.cards)}
    ]
    assert observation["trick_winners"] == [0]
    assert observation["terminated"] is False
