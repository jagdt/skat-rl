from skat_rl.engine.cards import Rank, Suit, card_points, card_rank, card_suit, full_deck
from skat_rl.engine.rules import card_strength_in_trick, is_trump, trick_winner
from skat_rl.engine.state import GameKind, Trick


class HeuristicAgent:
    '''
    A simple Skat heuristic.

    Declarer behavior:
    - If leading and has strong trumps, lead strong trump to pull defenders' trumps.
    - Prefer leading aces and protect tens until the matching ace is out.
    - If following, win the trick cheaply if possible.
    - Trump only when the trick is already valuable.
    - If cannot win, discard low-value cards.

    Defender behavior:
    - Prefer leading aces and protect tens until the matching ace is out.
    - If following, win valuable tricks if possible.
    - Trump when the declarer is currently winning the trick.
    - Smear high cards, especially playable tens, when the partner is winning.
    - If leading, prefer low non-trump cards.
    '''

    DECLARER_TRUMP_POINT_THRESHOLD = 9

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
            observation,
            player_id,
            game_type,
            current_trick_cards,
            legal_actions,
        )

    def _act_as_declarer(self, observation, player_id, game_type, current_trick_cards, legal_actions):
        '''
        Declarer strategy:
        - If leading:
            - If has strong trumps, lead strong trump to pull defenders' trumps.
            - Play leading cards from above is possible or otherwise lowest value card.
        - If following:
            - If can win the trick, play the cheapest non trump winning card.
            - Trump if the trick is already valuable.
            - If cannot win, discard low-value cards.

        Defender strategy:
        - If
        '''
        played_cards = self._played_cards(observation)

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

            return self._lead_card(game_type, legal_actions, played_cards)

        else: 
            # If following, try to win the trick cheaply.
            winning_cards = self._winning_cards(
                player_id,
                game_type,
                current_trick_cards,
                legal_actions,
            )
            trick_value_so_far = self._trick_value(current_trick_cards)

            if winning_cards:
                non_trump_winners = [
                    card for card in winning_cards
                    if not is_trump(card, game_type)
                ]
                if non_trump_winners:
                    return self._lowest_winning_card(
                        non_trump_winners,
                        game_type,
                        played_cards,
                    )

                # Declarer keeps trumps back unless the trick is worth fighting for.
                if trick_value_so_far >= self.DECLARER_TRUMP_POINT_THRESHOLD:
                    return min(
                        winning_cards,
                        key=lambda card: (
                            card_points(card),
                            self._trump_strength(card, game_type),
                        ),
                    )

            # If unable to win, throw away lowest value card.
            return self._lowest_discard_card(legal_actions, game_type, played_cards)

    def _act_as_defender(self, observation, player_id, game_type, current_trick_cards, legal_actions):
        played_cards = self._played_cards(observation)

        # If leading, avoid leading trump. Lead low non-trump if possible.
        if len(current_trick_cards) == 0:
            non_trumps = [
                card for card in legal_actions
                if not is_trump(card, game_type)
            ]

            if non_trumps:
                return self._lead_card(game_type, non_trumps, played_cards)

            return self._lead_card(game_type, legal_actions, played_cards)

        else:
            current_winner = self._current_winning_player(current_trick_cards, game_type)
            declarer = observation["declarer"]

            winning_cards = self._winning_cards(
                player_id,
                game_type,
                current_trick_cards,
                legal_actions,
            )

            if current_winner != declarer:
                return self._highest_discard_card(legal_actions, game_type, played_cards)

            if winning_cards:
                trump_winners = [
                    card for card in winning_cards
                    if is_trump(card, game_type)
                ]
                if trump_winners:
                    return min(
                        trump_winners,
                        key=lambda card: (
                            self._protected_ten_penalty(card, game_type, played_cards),
                            card_points(card),
                            self._trump_strength(card, game_type),
                        ),
                    )

                return self._lowest_winning_card(
                    winning_cards,
                    game_type,
                    played_cards,
                )

            # Declarer keeps the trick: give away as little as possible.
            return self._lowest_discard_card(legal_actions, game_type, played_cards)

    def _winning_cards(self, player_id, game_type, current_trick_cards, legal_actions):
        '''
        Return the subset of legal_actions that would currently win the trick if played.
        '''
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
        '''
        Order in which the declarer should lead trumps to pull defenders.

        Higher value means this card should be led first.
        Order: ♠J, ♥J, ♣J, ♦J, 9, 8, 7, Q, K, 10, A.
        '''
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
        '''
        Standard Skat trump strength (higher = stronger).

        Used for tie-breaking when picking the cheapest winning card.
        '''
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

    def _current_winning_player(self, current_trick_cards, game_type):
        '''
        Determine which player is currently winning the trick based on the cards
        '''
        lead_card = current_trick_cards[0][1]
        best_player, best_card = current_trick_cards[0]
        best_strength = card_strength_in_trick(best_card, lead_card, game_type)

        for player, card in current_trick_cards[1:]:
            strength = card_strength_in_trick(card, lead_card, game_type)
            if strength > best_strength:
                best_player = player
                best_strength = strength

        return best_player

    def _lead_card(self, game_type, legal_actions, played_cards):
        highest_remaining_cards = [
            card for card in legal_actions
            if self._is_highest_remaining_non_trump(card, game_type, played_cards)
        ]
        if highest_remaining_cards:
            return max(
                highest_remaining_cards,
                key=lambda card: (
                    card_points(card),
                    self._non_trump_rank_strength(card),
                ),
            )

        return self._lowest_discard_card(legal_actions, game_type, played_cards)

    def _is_highest_remaining_non_trump(self, card, game_type, played_cards):
        if is_trump(card, game_type):
            return False

        suit = card_suit(card)
        rank = card_rank(card)

        for higher_rank in self._higher_non_trump_ranks(rank, game_type):
            higher_card = int(suit) * 8 + int(higher_rank)
            if higher_card not in played_cards:
                return False

        return True

    def _higher_non_trump_ranks(self, rank, game_type):
        if game_type.kind == GameKind.NULL:
            ranks = (
                Rank.ACE,
                Rank.KING,
                Rank.QUEEN,
                Rank.JACK,
                Rank.TEN,
                Rank.NINE,
                Rank.EIGHT,
                Rank.SEVEN,
            )
        else:
            ranks = (
                Rank.ACE,
                Rank.TEN,
                Rank.KING,
                Rank.QUEEN,
                Rank.NINE,
                Rank.EIGHT,
                Rank.SEVEN,
            )
        return ranks[:ranks.index(rank)]

    def _non_trump_rank_strength(self, card):
        rank_order = {
            Rank.ACE: 7,
            Rank.TEN: 6,
            Rank.KING: 5,
            Rank.QUEEN: 4,
            Rank.NINE: 3,
            Rank.EIGHT: 2,
            Rank.SEVEN: 1,
        }
        return rank_order.get(card_rank(card), 0)

    def _lowest_winning_card(self, cards, game_type, played_cards):
        return min(
            cards,
            key=lambda card: (
                self._protected_ten_penalty(card, game_type, played_cards),
                card_points(card),
                self._trump_strength(card, game_type),
            ),
        )

    def _lowest_discard_card(self, cards, game_type, played_cards):
        return min(
            cards,
            key=lambda card: (
                card_points(card),
                self._trump_strength(card, game_type),
            ),
        )

    def _highest_discard_card(self, cards, game_type, played_cards):
        playable_tens = [
            card for card in cards
            if (
                card_rank(card) == Rank.TEN
                and not self._is_protected_ten(card, game_type, played_cards)
            )
        ]
        if playable_tens:
            return max(
                playable_tens,
                key=lambda card: (
                    card_points(card),
                    self._trump_strength(card, game_type),
                ),
            )

        return max(
            cards,
            key=lambda card: (
                -self._protected_ten_penalty(card, game_type, played_cards),
                card_points(card),
                self._trump_strength(card, game_type),
            ),
        )

    def _protected_ten_penalty(self, card, game_type, played_cards):
        if self._is_protected_ten(card, game_type, played_cards):
            return 1
        return 0

    def _is_protected_ten(self, card, game_type, played_cards):
        '''
        A ten is protected if the matching ace in the same suit has not yet been played.
        '''
        if card_rank(card) != Rank.TEN:
            return False

        ace = self._matching_ace(card)
        return ace not in played_cards

    def _matching_ace(self, card):
        '''
        Find the matching ace for a given card in the same suit.
        '''
        return int(card_suit(card)) * 8 + int(Rank.ACE)

    def _trick_value(self, current_trick_cards):
        '''
        Calculate the current point value of the trick based on the cards played so far.
        '''
        return sum(card_points(card) for _, card in current_trick_cards)

    def _played_cards(self, observation):
        '''
        Get a set of all cards that have been played in completed tricks and the current trick.
        '''
        played_cards = set()
        for trick in observation["completed_tricks"]:
            for _, card in trick["cards"]:
                played_cards.add(card)
        for _, card in observation["current_trick"]:
            played_cards.add(card)

        return played_cards

    def _count_outstanding_trumps(self, observation, game_type):
        '''
        Count how many trumps opponents still hold.

        total trumps in deck
        - trumps in own hand
        - trumps already played (completed tricks + current trick)
        = trumps still in opponent hands
        '''
        all_trumps = {card for card in full_deck() if is_trump(card, game_type)}

        own_hand = set(observation["own_hand"])
        own_trumps = all_trumps & own_hand

        played_cards = self._played_cards(observation)

        played_trumps = all_trumps & played_cards

        return len(all_trumps) - len(own_trumps) - len(played_trumps)
