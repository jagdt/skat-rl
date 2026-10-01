import random

from .cards import full_deck, card_rank, card_suit, Rank, Suit
from .rules import game_result, legal_moves, trick_points, trick_winner
from .state import AuctionRole, BiddingStatus, GameKind, GameState, GameType, Phase, Trick
from .scoring import game_value, tournament_rewards
from .actions import BID_VALUES, NUM_ACTIONS, BidAction, PickupAction, contract_type, discard_pair, legal_contracts


class StepResult:
    def __init__(self, state, reward, terminated, info):
        self.state = state
        self.reward = reward
        self.terminated = terminated
        self.info = info


class SkatGame:
    def __init__(self, game_type=None, declarer=0, fixed_declarer=None, seed=None):
        if fixed_declarer is not None and fixed_declarer not in range(3):
            raise ValueError("fixed_declarer must be 0, 1, 2, or None.")

        self.rng = random.Random(seed)
        self.default_game_type = game_type or GameType(GameKind.GRAND)
        self.default_declarer = declarer
        self.fixed_declarer = fixed_declarer
        self.state = None

    def reset(self, game_type=None, declarer=None, seed=None, full_game=False, forehand=None):
        if seed is not None:
            self.rng.seed(seed)

        if full_game:
            if declarer is not None or game_type is not None or self.fixed_declarer is not None:
                raise ValueError("Full games determine declarer and contract through play.")
            deck = full_deck()
            self.rng.shuffle(deck)
            return self.reset_full_from_deal(
                [deck[p * 10:(p + 1) * 10] for p in range(3)], deck[30:],
                self.rng.randrange(3) if forehand is None else forehand,
            )

        while True:
            deck = full_deck()
            self.rng.shuffle(deck)

            hands = [
                set(deck[0:10]),
                set(deck[10:20]),
                set(deck[20:30]),
            ]

            if (
                declarer is not None
                or self.fixed_declarer is None
                or self._choose_declarer(hands) == self.fixed_declarer
            ):
                break

        skat = deck[30:32]

        if declarer is None:
            if self.fixed_declarer is not None:
                declarer = self.fixed_declarer
            else:
                declarer = self._choose_declarer(hands)

        if game_type is None:
            trump_suit = self._choose_trump_suit(hands[declarer])
            game_type = GameType(GameKind.SUIT, trump_suit=trump_suit, hand=True)

        if game_type is None:
            game_type = self.default_game_type

        current_player = 0

        self.state = GameState(
            hands=hands,
            skat=skat,
            declarer=declarer,
            game_type=game_type,
            current_player=current_player,
            current_trick=Trick(leader=current_player),
        )

        return self.state

    def reset_full_from_deal(self, hands, skat, forehand=0):
        cards = [card for hand in hands for card in hand] + list(skat)
        if (len(hands) != 3 or any(len(hand) != 10 for hand in hands)
                or len(skat) != 2 or sorted(cards) != list(range(32)) or forehand not in range(3)):
            raise ValueError("Expected three 10-card hands, two Skat cards, and a valid forehand.")
        caller = (forehand + 1) % 3
        self.state = GameState(
            hands=[set(hand) for hand in hands], skat=list(skat), declarer=-1,
            game_type=None, current_player=caller, current_trick=Trick(leader=forehand),
            phase=Phase.BIDDING, forehand=forehand, auction_caller=caller, auction_holder=forehand,
        )
        self.state.bid_status[forehand] = self.state.bid_status[caller] = BiddingStatus.ACTIVE
        return self.state

    def observe(self, player):
        self._require_state()
        return self.state.clone_public_for_player(player)

    def legal_actions(self, player=None):
        self._require_state()

        if self.state.terminated:
            return []

        if player is None:
            player = self.state.current_player

        if player != self.state.current_player:
            return []

        if self.state.phase == Phase.BIDDING:
            return list(BidAction)
        if self.state.phase == Phase.PICKUP_DECISION:
            return list(PickupAction)
        if self.state.phase == Phase.DISCARD:
            return list(range(NUM_ACTIONS))
        if self.state.phase == Phase.CONTRACT_SELECTION:
            return legal_contracts(self.state.winning_bid, self.state.hand_game)

        return legal_moves(
            self.state.hands[player],
            self.state.current_trick,
            self.state.game_type,
        )

    def step(self, action):
        self._require_state()

        if self.state.terminated:
            raise RuntimeError("Cannot step terminated game. Call reset().")

        player = self.state.current_player
        legal = self.legal_actions(player)

        if action not in legal:
            raise ValueError(
                f"Illegal action {action} by player {player}. "
                f"Legal actions are {legal}."
            )

        if self.state.phase == Phase.CARD_PLAY:
            return self._step_card_play(action)

        return self._step_preplay(action)

    def _step_card_play(self, action):
        player = self.state.current_player
        self.state.hands[player].remove(action)
        self.state.current_trick.cards.append((player, action))

        reward = [0.0, 0.0, 0.0]
        info = {}

        if self.state.current_trick.is_complete():
            winner = trick_winner(
                self.state.current_trick,
                self.state.game_type,
            )

            points = trick_points(self.state.current_trick)

            for _, card in self.state.current_trick.cards:
                self.state.won_cards[winner].append(card)

            self.state.trick_winners.append(winner)
            self.state.completed_tricks.append(self.state.current_trick)

            info["trick_winner"] = winner
            info["trick_points"] = points

            null_lost = self.state.game_type.kind == GameKind.NULL and winner == self.state.declarer
            if len(self.state.completed_tricks) == 10 or null_lost:
                self.state.phase = Phase.TERMINAL

                result = game_result(
                    won_cards=self.state.won_cards,
                    trick_winners=self.state.trick_winners,
                    declarer=self.state.declarer,
                    game_type=self.state.game_type,
                    skat=self.state.skat,
                )

                declarer_cards = set(self.state.skat)
                declarer_cards.update(
                    card for trick in self.state.completed_tricks
                    for player, card in trick.cards if player == self.state.declarer
                )
                result["game_value"] = game_value(
                    self.state.game_type, declarer_cards, result["schneider"], result["schwarz"],
                )
                result["overbid"] = result["game_value"] < self.state.winning_bid
                if result["overbid"]:
                    result["declarer_won"] = False
                    base = 24 if self.state.game_type.kind == GameKind.GRAND else 12 - int(self.state.game_type.trump_suit)
                    result["game_value"] = ((self.state.winning_bid + base - 1) // base) * base
                info["result"] = result
                reward = self._terminal_reward(result)

            else:
                self.state.current_player = winner
                self.state.current_trick = Trick(leader=winner)

        else:
            self.state.current_player = (player + 1) % 3

        return StepResult(
            state=self.state,
            reward=reward,
            terminated=self.state.terminated,
            info=info,
        )

    def _step_preplay(self, action):
        state = self.state
        if state.phase == Phase.BIDDING:
            self._step_bid(action)
        elif state.phase == Phase.PICKUP_DECISION:
            state.hand_game = int(action == PickupAction.HAND)
            if state.hand_game:
                state.phase = Phase.CONTRACT_SELECTION
            else:
                state.hands[state.declarer].update(state.skat)
                state.skat = []
                state.phase = Phase.DISCARD
        elif state.phase == Phase.DISCARD:
            state.skat = list(discard_pair(state.hands[state.declarer], action))
            state.hands[state.declarer].difference_update(state.skat)
            state.phase = Phase.CONTRACT_SELECTION
        elif state.phase == Phase.CONTRACT_SELECTION:
            state.game_type = contract_type(action, bool(state.hand_game))
            state.phase = Phase.CARD_PLAY
            state.current_player = state.forehand
        info = {"passed_out": True} if state.terminated else {}
        
        return StepResult(state=state, 
            reward=[0.0] * 3, 
            terminated=state.terminated, 
            info=info
        )

    def _step_bid(self, action):
        state = self.state
        player = state.current_player
        threshold = BID_VALUES[state.bid_index]
        if action == BidAction.PASS:
            state.bid_status[player] = BiddingStatus.PASSED
            state.pass_threshold[player] = threshold
            state.pass_role[player] = state.auction_role
            if state.forehand_offer:
                state.phase = Phase.TERMINAL
                return
            winner = state.auction_holder if state.auction_role == AuctionRole.CALLER else state.auction_caller
            self._finish_duel(winner)
        elif state.forehand_offer:
            self._finish_auction(player)
        elif state.auction_role == AuctionRole.CALLER:
            state.highest_called[player] = threshold
            state.winning_bid = threshold
            state.current_player = state.auction_holder
            state.auction_role = AuctionRole.HOLDER
        else:
            state.highest_held[player] = threshold
            state.winning_bid = threshold
            if threshold == BID_VALUES[-1]:
                self._finish_auction(player)
                return
            state.bid_index += 1
            state.current_player = state.auction_caller
            state.auction_role = AuctionRole.CALLER

    def _finish_duel(self, winner):
        state = self.state
        if state.winning_bid == BID_VALUES[-1]:
            self._finish_auction(winner)
        elif not state.rearhand_entered:
            state.rearhand_entered = True
            state.auction_caller = (state.forehand + 2) % 3
            state.auction_holder = winner
            state.current_player = state.auction_caller
            state.auction_role = AuctionRole.CALLER
            state.bid_status[state.auction_caller] = BiddingStatus.ACTIVE
            state.bid_index = next(i for i, bid in enumerate(BID_VALUES) if bid > state.winning_bid)
        elif state.winning_bid == 0:
            state.forehand_offer = True
            state.current_player = winner
            state.auction_role = AuctionRole.CALLER
            state.bid_index = 0
        else:
            self._finish_auction(winner)

    def _finish_auction(self, winner):
        self.state.declarer = winner
        self.state.current_player = winner
        self.state.winning_bid = max(18, self.state.winning_bid)
        self.state.phase = Phase.PICKUP_DECISION

    def _choose_declarer(self, hands):
        '''
        Simple heuristic to choose the declarer.
        '''
        scores = []

        for player, hand in enumerate(hands):
            num_jacks = sum(1 for card in hand if card_rank(card) == Rank.JACK)
            num_aces = sum(1 for card in hand if card_rank(card) == Rank.ACE)
            num_tens = sum(0.4 for card in hand if card_rank(card) == Rank.TEN)

            score = num_jacks + num_aces + num_tens
            scores.append((score, player))

        scores.sort(key=lambda x: (-x[0], x[1]))
        return scores[0][1]

    def _choose_trump_suit(self, hand):
        '''
        Simple heuristic to choose the trump suit.
        '''
        suit_counts = {
            Suit.CLUBS: 0,
            Suit.SPADES: 0,
            Suit.HEARTS: 0,
            Suit.DIAMONDS: 0,
        }

        suit_has_ten = {
            Suit.CLUBS: False,
            Suit.SPADES: False,
            Suit.HEARTS: False,
            Suit.DIAMONDS: False,
        }

        for card in hand:
            suit = card_suit(card)
            suit_counts[suit] += 1

            if card_rank(card) == Rank.TEN:
                suit_has_ten[suit] = True

        best_suit = max(
            suit_counts,
            key=lambda suit: (suit_counts[suit], suit_has_ten[suit], -int(suit)),
        )

        return best_suit

    def _terminal_reward(self, result):
        return tournament_rewards(result["declarer"], result["declarer_won"], result["game_value"])

    def _require_state(self):
        if self.state is None:
            raise RuntimeError("Game has not been reset yet.")
