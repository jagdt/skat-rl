import random

from .cards import full_deck
from .rules import game_result, legal_moves, trick_points, trick_winner
from .state import GameKind, GameState, GameType, Trick


class StepResult:
    def __init__(self, state, reward, terminated, info):
        self.state = state
        self.reward = reward
        self.terminated = terminated
        self.info = info


class SkatGame:
    def __init__(self, game_type=None, declarer=0, seed=None):
        self.rng = random.Random(seed)
        self.default_game_type = game_type or GameType(GameKind.GRAND)
        self.default_declarer = declarer
        self.state = None

    def reset(self, game_type=None, declarer=None, seed=None):
        if seed is not None:
            self.rng.seed(seed)

        deck = full_deck()
        self.rng.shuffle(deck)

        hands = [
            set(deck[0:10]),
            set(deck[10:20]),
            set(deck[20:30]),
        ]

        skat = deck[30:32]

        if declarer is None:
            declarer = self.default_declarer

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

            if len(self.state.completed_tricks) == 10:
                self.state.terminated = True

                result = game_result(
                    won_cards=self.state.won_cards,
                    trick_winners=self.state.trick_winners,
                    declarer=self.state.declarer,
                    game_type=self.state.game_type,
                )

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

    def play_random_game(self):
        self.reset()

        final_step = None

        while not self.state.terminated:
            legal = self.legal_actions()
            action = self.rng.choice(legal)
            final_step = self.step(action)

        return final_step.info["result"]

    def _terminal_reward(self, result):
        declarer = result["declarer"]
        declarer_won = result["declarer_won"]

        reward = [0.0, 0.0, 0.0]

        if declarer_won:
            reward[declarer] = 1.0
            for player in range(3):
                if player != declarer:
                    reward[player] = -0.5
        else:
            reward[declarer] = -1.0
            for player in range(3):
                if player != declarer:
                    reward[player] = 0.5

        return reward

    def _require_state(self):
        if self.state is None:
            raise RuntimeError("Game has not been reset yet.")