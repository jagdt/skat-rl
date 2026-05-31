from .cards import Rank, Suit, card_points, card_rank, card_suit
from .state import GameKind


JACK_TRUMP_ORDER = {
    Suit.CLUBS: 4,
    Suit.SPADES: 3,
    Suit.HEARTS: 2,
    Suit.DIAMONDS: 1,
}

SUIT_GAME_RANK_ORDER = {
    Rank.ACE: 7,
    Rank.TEN: 6,
    Rank.KING: 5,
    Rank.QUEEN: 4,
    Rank.NINE: 3,
    Rank.EIGHT: 2,
    Rank.SEVEN: 1,
}

NULL_RANK_ORDER = {
    Rank.ACE: 8,
    Rank.KING: 7,
    Rank.QUEEN: 6,
    Rank.JACK: 5,
    Rank.TEN: 4,
    Rank.NINE: 3,
    Rank.EIGHT: 2,
    Rank.SEVEN: 1,
}


def is_trump(card, game_type):
    rank = card_rank(card)
    suit = card_suit(card)

    if game_type.kind == GameKind.NULL:
        return False

    if rank == Rank.JACK:
        return True

    if game_type.kind == GameKind.GRAND:
        return False

    if game_type.kind == GameKind.SUIT:
        if game_type.trump_suit is None:
            raise ValueError("Suit game requires trump_suit.")
        return suit == Suit(game_type.trump_suit)

    raise ValueError(f"Unsupported game type: {game_type}")


def effective_suit(card, game_type):
    if is_trump(card, game_type):
        return "TRUMP"
    return card_suit(card)


def legal_moves(hand, current_trick, game_type):
    if not hand:
        return []

    if not current_trick.cards:
        return sorted(hand)

    lead = current_trick.lead_card()
    required = effective_suit(lead, game_type)

    matching = [
        card for card in hand
        if effective_suit(card, game_type) == required
    ]

    if matching:
        return sorted(matching)

    return sorted(hand)


def card_strength_in_trick(card, lead_card, game_type):
    if game_type.kind == GameKind.NULL:
        if card_suit(card) != card_suit(lead_card):
            return 0
        return NULL_RANK_ORDER[card_rank(card)]

    card_is_trump = is_trump(card, game_type)
    lead_is_trump = is_trump(lead_card, game_type)

    if card_is_trump:
        rank = card_rank(card)
        suit = card_suit(card)

        if rank == Rank.JACK:
            return 100 + JACK_TRUMP_ORDER[suit]

        return 50 + SUIT_GAME_RANK_ORDER[rank]

    if lead_is_trump:
        return 0

    if card_suit(card) != card_suit(lead_card):
        return 0

    return SUIT_GAME_RANK_ORDER[card_rank(card)]


def trick_winner(trick, game_type):
    if not trick.is_complete():
        raise ValueError("Cannot determine winner of incomplete trick.")

    lead_card = trick.lead_card()

    best_player, best_card = trick.cards[0]
    best_strength = card_strength_in_trick(best_card, lead_card, game_type)

    for player, card in trick.cards[1:]:
        strength = card_strength_in_trick(card, lead_card, game_type)
        if strength > best_strength:
            best_player = player
            best_card = card
            best_strength = strength

    return best_player


def trick_points(trick):
    return sum(card_points(card) for _, card in trick.cards)


def declarer_points(won_cards, declarer):
    return sum(card_points(card) for card in won_cards[declarer])


def declarer_took_trick(trick_winners, declarer):
    return any(winner == declarer for winner in trick_winners)


def game_result(won_cards, trick_winners, declarer, game_type):
    if game_type.kind == GameKind.NULL:
        declarer_won = not declarer_took_trick(trick_winners, declarer)

        return {
            "declarer": declarer,
            "declarer_won": declarer_won,
            "declarer_points": 0,
            "defender_points": 0,
            "schneider": False,
            "schwarz": declarer_won,
        }

    dec_points = declarer_points(won_cards, declarer)
    def_points = 120 - dec_points

    declarer_won = dec_points > 60

    if declarer_won:
        schneider = def_points <= 30
        schwarz = def_points == 0
    else:
        schneider = dec_points <= 30
        schwarz = dec_points == 0

    return {
        "declarer": declarer,
        "declarer_won": declarer_won,
        "declarer_points": dec_points,
        "defender_points": def_points,
        "schneider": schneider,
        "schwarz": schwarz,
    }