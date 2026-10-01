"""Phase-local action IDs shared by observations, models, and game replay."""

from enum import IntEnum
from itertools import combinations

from .cards import Suit
from .state import GameKind, GameType


class BidAction(IntEnum):
    PASS = 0
    CONTINUE = 1


class PickupAction(IntEnum):
    PICKUP = 0
    HAND = 1


class Contract(IntEnum):
    CLUBS = 0
    SPADES = 1
    HEARTS = 2
    DIAMONDS = 3
    GRAND = 4
    NULL = 5


# Standard bid ladder, including values attainable by announced/open games.
# Those announcements are not actions in the six-contract model.
BID_VALUES = tuple(sorted(
    {base * multiplier for base in (9, 10, 11, 12) for multiplier in range(2, 19)}
    | {24 * multiplier for multiplier in range(2, 12)} | {23, 35, 46, 59}
))
DISCARD_PAIRS = tuple(combinations(range(12), 2))
NUM_ACTIONS = len(DISCARD_PAIRS)


def contract_type(contract, hand=False):
    contract = Contract(contract)
    if contract < Contract.GRAND:
        return GameType(GameKind.SUIT, Suit(contract), hand=hand)
    return GameType(GameKind.GRAND if contract == Contract.GRAND else GameKind.NULL, hand=hand)


def legal_contracts(winning_bid, hand):
    # Never inspect hidden Skat cards to mask Hand contracts. Resolve possible
    # suit/Grand overbids using actual game value at the end of play.
    contracts = list(range(int(Contract.NULL)))
    if winning_bid <= (35 if hand else 23):
        contracts.append(int(Contract.NULL))
    return contracts


def discard_pair(hand, action):
    cards = sorted(hand)
    if len(cards) != 12 or action not in range(NUM_ACTIONS):
        raise ValueError("Discard requires 12 cards and a pair action in 0..65.")
    i, j = DISCARD_PAIRS[action]
    return cards[i], cards[j]


def discard_action(hand, cards):
    ordered = sorted(hand)
    pair = tuple(sorted(ordered.index(card) for card in cards))
    if len(ordered) != 12 or len(pair) != 2 or pair[0] == pair[1]:
        raise ValueError("Expected two distinct cards from a 12-card hand.")
    return DISCARD_PAIRS.index(pair)
