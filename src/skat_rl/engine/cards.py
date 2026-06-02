from enum import IntEnum


class Suit(IntEnum):
    CLUBS = 0
    SPADES = 1
    HEARTS = 2
    DIAMONDS = 3


class Rank(IntEnum):
    SEVEN = 0
    EIGHT = 1
    NINE = 2
    QUEEN = 3
    KING = 4
    TEN = 5
    ACE = 6
    JACK = 7


SUIT_SYMBOLS = {
    Suit.CLUBS: "♣",
    Suit.SPADES: "♠",
    Suit.HEARTS: "♥",
    Suit.DIAMONDS: "♦",
}

RANK_SYMBOLS = {
    Rank.SEVEN: "7",
    Rank.EIGHT: "8",
    Rank.NINE: "9",
    Rank.QUEEN: "Q",
    Rank.KING: "K",
    Rank.TEN: "10",
    Rank.ACE: "A",
    Rank.JACK: "J",
}

CARD_POINTS = {
    Rank.JACK: 2,
    Rank.ACE: 11,
    Rank.TEN: 10,
    Rank.KING: 4,
    Rank.QUEEN: 3,
    Rank.NINE: 0,
    Rank.EIGHT: 0,
    Rank.SEVEN: 0,
}


def make_card(suit, rank):
    return int(suit) * 8 + int(rank)


def card_suit(card):
    return Suit(card // 8)


def card_rank(card):
    return Rank(card % 8)


def card_points(card):
    return CARD_POINTS[card_rank(card)]


def full_deck():
    return [make_card(suit, rank) for suit in Suit for rank in Rank]


def card_str(card):
    return f"{RANK_SYMBOLS[card_rank(card)]}{SUIT_SYMBOLS[card_suit(card)]}"


def cards_str(cards):
    return " ".join(card_str(c) for c in cards)


def sort_cards(cards):
    rank_order = {
        Rank.JACK: 0,
        Rank.ACE: 1,
        Rank.TEN: 2,
        Rank.KING: 3,
        Rank.QUEEN: 4,
        Rank.NINE: 5,
        Rank.EIGHT: 6,
        Rank.SEVEN: 7,
    }
    return sorted(
        cards,
        key=lambda card: (
            card_rank(card) != Rank.JACK,
            card_suit(card),
            rank_order[card_rank(card)],
        ),
    )
