from skat_rl.engine.cards import Rank, Suit, card_points, card_rank, card_suit, full_deck
from skat_rl.engine.rules import card_strength_in_trick, is_trump, trick_winner
from skat_rl.engine.state import Trick


class HeuristicAgent:
    """
    A simple Skat heuristic.

    Declarer behavior:
    - If leading and has strong trumps, lead strong trump to pull defenders' trumps.
    - If following, win the trick cheaply if possible.
    - If cannot win, discard low-value cards.

    Defender behavior:
    - If following, win valuable tricks if possible.
    - Avoid wasting trump if the trick is low-value.
    - If leading, prefer low non-trump cards.
    """

    def act(self, observation, legal_actions):
        if not legal_actions:
            raise ValueError("No legal actions available.")

        player_id = observation["player_id"]
        declarer = observation["declarer"]
        game_type = observation["game_type"]
        current_trick_cards = observation["current_trick"]

        if player_id == declarer:
            return self._act_as_declarer(
                observation,
                player_id,
                game_type,
                current_trick_cards,
                legal_actions,
            )

        return self._act_as_defender(
            player_id,
            game_type,
            current_trick_cards,
            legal_actions,
        )

    def _act_as_declarer(self, observation, player_id, game_type, current_trick_cards, legal_actions):

        # If leading, pull trump while opponents still have trumps.
        if len(current_trick_cards) == 0:
            trumps = [card for card in legal_actions if is_trump(card, game_type)]
            opponent_trumps = self._count_outstanding_trumps(observation, game_type)

            if trumps and opponent_trumps > 0: # TODO pull only trump if the trumps are spread well or if we have enough trumps
                # Lead in pulling order: ♠J, ♥J, ♣J, ♦J, then cheap trumps.
                return max(
                    trumps,
                    key=lambda card: self._trump_pulling_order(card, game_type),
                )

            # Opponents are trump-free (or we have no trumps) — lead low-point card.
            return min(legal_actions, key=card_points)

        # If following, try to win the trick cheaply.
        winning_cards = self._winning_cards(
            player_id,
            game_type,
            current_trick_cards,
            legal_actions,
        )

        if winning_cards:
            return min(
                winning_cards,
                key=lambda card: (
                    card_points(card),
                    self._trump_strength(card, game_type),
                ),
            )

        # If unable to win, throw away lowest value card.
        return min(legal_actions, key=card_points)

    def _act_as_defender(self, player_id, game_type, current_trick_cards, legal_actions):
        # If leading, avoid leading trump. Lead low non-trump if possible.
        if len(current_trick_cards) == 0:
            non_trumps = [
                card for card in legal_actions
                if not is_trump(card, game_type)
            ]

            if non_trumps:
                return min(non_trumps, key=card_points)

            return min(legal_actions, key=card_points)

        trick_value_so_far = sum(card_points(card) for _, card in current_trick_cards)

        winning_cards = self._winning_cards(
            player_id,
            game_type,
            current_trick_cards,
            legal_actions,
        )

        if winning_cards:
            # If many points are already in the trick, try to win cheaply.
            if trick_value_so_far >= 10:
                return min(
                    winning_cards,
                    key=lambda card: (
                        card_points(card),
                        self._trump_strength(card, game_type),
                    ),
                )

            # If trick is not valuable, avoid spending trump if possible.
            non_trump_winners = [
                card for card in winning_cards
                if not is_trump(card, game_type)
            ]

            if non_trump_winners:
                return min(non_trump_winners, key=card_points)

        # If not winning, discard lowest point card.
        return min(legal_actions, key=card_points)

    def _winning_cards(self, player_id, game_type, current_trick_cards, legal_actions):
        winning_cards = []

        for card in legal_actions:
            simulated_trick = Trick(
                leader=current_trick_cards[0][0],
                cards=list(current_trick_cards) + [(player_id, card)],
            )

            # If trick is complete, use exact trick winner.
            if simulated_trick.is_complete():
                winner = trick_winner(simulated_trick, game_type)
                if winner == player_id:
                    winning_cards.append(card)

            # If trick has only 1 or 2 cards after our play, check if our card
            # is currently strongest. This is only approximate because later
            # players may still beat it.
            else:
                lead_card = simulated_trick.lead_card()
                our_strength = card_strength_in_trick(card, lead_card, game_type)

                currently_best = True
                for other_player, other_card in simulated_trick.cards:
                    if other_player == player_id:
                        continue

                    other_strength = card_strength_in_trick(
                        other_card,
                        lead_card,
                        game_type,
                    )

                    if other_strength > our_strength:
                        currently_best = False
                        break

                if currently_best:
                    winning_cards.append(card)

        return winning_cards

    def _trump_pulling_order(self, card, game_type):
        """
        Order in which the declarer should lead trumps to pull defenders.

        Higher value means this card should be led first.
        Order: ♠J, ♥J, ♣J, ♦J, 9, 8, 7, Q, K, 10, A.
        """
        if not is_trump(card, game_type):
            return 0

        rank = card_rank(card)

        if rank == Rank.JACK:
            # Spades jack first, then hearts, clubs, diamonds.
            jack_pull_order = {
                Suit.SPADES: 110,
                Suit.HEARTS: 109,
                Suit.CLUBS: 108,
                Suit.DIAMONDS: 107,
            }
            return jack_pull_order[card_suit(card)]

        # Non-jack trumps: lead cheap ones first (9,8,7,Q,K) then
        # expensive ones (10,A) to minimise point risk.
        suit_trump_pull_order = {
            Rank.NINE: 60,
            Rank.EIGHT: 50,
            Rank.SEVEN: 40,
            Rank.QUEEN: 30,
            Rank.KING: 20,
            Rank.TEN: 10,
            Rank.ACE: 5,
        }

        return suit_trump_pull_order.get(rank, 0)

    def _trump_strength(self, card, game_type):
        """
        Standard Skat trump strength (higher = stronger).

        Used for tie-breaking when picking the cheapest winning card.
        """
        if not is_trump(card, game_type):
            return 0

        rank = card_rank(card)

        if rank == Rank.JACK:
            # Standard Skat order: ♣J > ♠J > ♥J > ♦J.
            return 100 - int(card_suit(card))

        suit_trump_order = {
            Rank.ACE: 70,
            Rank.TEN: 60,
            Rank.KING: 50,
            Rank.QUEEN: 40,
            Rank.NINE: 30,
            Rank.EIGHT: 20,
            Rank.SEVEN: 10,
        }

        return suit_trump_order.get(rank, 0)

    def _count_outstanding_trumps(self, observation, game_type):
        """
        Count how many trumps opponents still hold.

        total trumps in deck
        − trumps in own hand
        − trumps already played (completed tricks + current trick)
        = trumps still in opponent hands
        """
        all_trumps = {card for card in full_deck() if is_trump(card, game_type)}

        own_hand = set(observation["own_hand"])
        own_trumps = all_trumps & own_hand

        played_cards = set()
        for trick in observation["completed_tricks"]:
            for _, card in trick["cards"]:
                played_cards.add(card)
        for _, card in observation["current_trick"]:
            played_cards.add(card)

        played_trumps = all_trumps & played_cards

        return len(all_trumps) - len(own_trumps) - len(played_trumps)