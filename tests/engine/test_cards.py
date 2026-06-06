from skat_rl.engine.cards import (
    Rank,
    Suit,
    card_points,
    card_rank,
    card_str,
    card_suit,
    full_deck,
    make_card,
    sort_cards,
)


def test_make_card_round_trip():
    card = make_card(Suit.HEARTS, Rank.TEN)

    assert card_suit(card) == Suit.HEARTS
    assert card_rank(card) == Rank.TEN
    assert card_points(card) == 10
    assert card_str(card) == "10♥"


def test_full_deck_contains_32_unique_cards():
    deck = full_deck()

    assert len(deck) == 32
    assert len(set(deck)) == 32


def test_sort_cards_puts_jacks_first_then_descending_ranks_by_suit():
    cards = [
        make_card(Suit.DIAMONDS, Rank.SEVEN),
        make_card(Suit.CLUBS, Rank.ACE),
        make_card(Suit.HEARTS, Rank.JACK),
        make_card(Suit.CLUBS, Rank.TEN),
        make_card(Suit.SPADES, Rank.JACK),
        make_card(Suit.SPADES, Rank.KING),
        make_card(Suit.CLUBS, Rank.QUEEN),
    ]

    assert sort_cards(cards) == [
        make_card(Suit.SPADES, Rank.JACK),
        make_card(Suit.HEARTS, Rank.JACK),
        make_card(Suit.CLUBS, Rank.ACE),
        make_card(Suit.CLUBS, Rank.TEN),
        make_card(Suit.CLUBS, Rank.QUEEN),
        make_card(Suit.SPADES, Rank.KING),
        make_card(Suit.DIAMONDS, Rank.SEVEN),
    ]
