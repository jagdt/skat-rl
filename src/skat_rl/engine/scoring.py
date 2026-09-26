"""Three-player Seeger-Fabian scoring for the supported Skat contracts."""

from .cards import Rank, Suit, make_card
from .state import GameKind


REWARD_SCHEME = "tournament_3p_v1"
REWARD_SCALE = 100.0


def game_value(game_type, declarer_cards, schneider=False, schwarz=False):
    """Unsigned game value; declarer_cards includes the Skat, even for Hand.

    Announced Schneider/Schwarz, ouvert and overbids are not supported by the
    current engine/importer. A loss is doubled by scoring, not by this function.
    """
    if game_type.kind == GameKind.NULL:
        return 35 if game_type.hand else 23
    trumps = [make_card(suit, Rank.JACK) for suit in Suit]
    if game_type.kind == GameKind.SUIT:
        if game_type.trump_suit is None:
            raise ValueError("Suit game requires trump_suit.")
        suit = Suit(game_type.trump_suit)
        trumps.extend(make_card(suit, rank) for rank in
                      (Rank.ACE, Rank.TEN, Rank.KING, Rank.QUEEN, Rank.NINE, Rank.EIGHT, Rank.SEVEN))
        base = (12, 11, 10, 9)[suit]
    elif game_type.kind == GameKind.GRAND:
        base = 24
    else:
        raise ValueError(f"Unsupported game type: {game_type}")
    cards = set(declarer_cards)
    with_top = trumps[0] in cards
    matadors = 0
    for card in trumps:
        if (card in cards) != with_top:
            break
        matadors += 1
    multiplier = matadors + 1 + int(game_type.hand) + int(schneider) + int(schwarz)
    return base * multiplier


def tournament_rewards(declarer, won, value):
    """Actual score-sheet points / 100; these rewards are not zero-sum."""
    if declarer not in range(3) or value <= 0:
        raise ValueError("Scoring needs a valid declarer and positive game value.")
    rewards = [0.0 if won else 40.0 / REWARD_SCALE] * 3
    rewards[declarer] = (value + 50 if won else -2 * value - 50) / REWARD_SCALE
    return rewards
