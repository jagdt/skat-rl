import numpy as np
import gymnasium as gym
from gymnasium import spaces

from skat_rl.engine.game import SkatGame
from skat_rl.engine.state import GameKind
from skat_rl.engine.rules import effective_suit, points_won_by_player
from skat_rl.agents.heuristic_agent import HeuristicAgent
from skat_rl.agents.random_agent import RandomAgent


class SkatSingleAgentEnv(gym.Env):
    """
    Gymnasium environment for training one RL-controlled player.

    The RL agent controls `learning_player`.
    All other players are controlled by heuristic agents.

    Action space:
        Discrete(32), one action per card.

    Observation:
        A flat vector containing:
        - own hand
        - ordered history cards
        - ordered history players
        - current trick cards
        - current player one-hot
        - current trick leader one-hot
        - declarer one-hot
        - game type/trump encoding
        - trick number
        - current trick position
        - current declarer and defender points
        - void information
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, learning_player=0, opponent_agents=None, seed=None):
        super().__init__()

        self.learning_player = learning_player
        self.seed_value = seed

        self.game = SkatGame(seed=seed)

        if opponent_agents is None:
            opponent_agents = [
                None,
                HeuristicAgent(),
                HeuristicAgent(),
            ]

        self.opponent_agents = opponent_agents

        self.action_space = spaces.Discrete(32)

        obs_dim = (
            32              # own hand
            + 10 * 3 * 32   # history cards
            + 10 * 3 * 3    # history players
            + 32             # current trick cards
            + 3             # current player
            + 3             # current trick leader
            + 3             # declarer
            + 3             # game kind
            + 4             # trump suit
            + 1             # trick number
            + 1             # current trick position
            + 2             # declarer and defender points
            + 3 * 5         # void info
        )

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.game.reset(seed=seed)
        else:
            self.game.reset()

        self._play_until_learning_player()

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        action = int(action)

        if self.game.state.terminated:
            observation = self._get_observation()
            return observation, 0.0, True, False, {}

        if self.game.state.current_player != self.learning_player:
            raise RuntimeError("It is not currently the learning player's turn.")

        legal = self.game.legal_actions(self.learning_player)

        if action not in legal:
            raise ValueError(
                f"Illegal action {action}. Legal actions are {legal}."
            )

        step_result = self.game.step(action)

        reward = step_result.reward[self.learning_player]
        terminated = step_result.terminated
        info = dict(step_result.info)

        if not terminated:
            opponent_reward, opponent_info = self._play_until_learning_player()
            reward += opponent_reward
            info.update(opponent_info)

        observation = self._get_observation()
        terminated = self.game.state.terminated
        truncated = False

        return observation, reward, terminated, truncated, info

    def action_masks(self):
        """
        Required by sb3-contrib MaskablePPO.

        Returns a boolean mask of shape (32,).
        True means action is legal.
        False means action is illegal.
        """
        mask = np.zeros(32, dtype=bool)

        if self.game.state is None:
            return mask

        if self.game.state.terminated:
            return mask

        if self.game.state.current_player != self.learning_player:
            return mask

        legal = self.game.legal_actions(self.learning_player)

        for action in legal:
            mask[action] = True

        return mask

    def render(self):
        if self.game.state is None:
            print("No game state.")
            return

        state = self.game.state
        print(f"Current player: {state.current_player}")
        print(f"Declarer: {state.declarer}")
        print(f"Completed tricks: {len(state.completed_tricks)}")
        print(f"Current trick: {state.current_trick.cards}")

    def _play_until_learning_player(self):
        """
        Let heuristic opponents play until:
        - it is learning_player's turn, or
        - the game terminates.

        Returns reward accumulated for the learning player.
        """
        total_reward = 0.0
        info = {}

        while (
            not self.game.state.terminated
            and self.game.state.current_player != self.learning_player
        ):
            player = self.game.state.current_player
            obs = self.game.observe(player)
            legal = self.game.legal_actions(player)

            action = self.opponent_agents[player].act(obs, legal)
            step_result = self.game.step(action)

            total_reward += step_result.reward[self.learning_player]
            info.update(step_result.info)

        return total_reward, info

    def _get_observation(self):
        if self.game.state is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        state = self.game.state
        player = self.learning_player

        own_hand = np.zeros(32, dtype=np.float32)
        for card in state.hands[player]:
            own_hand[card] = 1.0

        history_cards = np.zeros((10, 3, 32), dtype=np.float32)
        history_players = np.zeros((10, 3, 3), dtype=np.float32)
        tricks = list(state.completed_tricks)
        if not state.current_trick.is_complete():
            tricks.append(state.current_trick)

        for trick_index, trick in enumerate(tricks[:10]):
            for slot_index, (card_player, card) in enumerate(trick.cards[:3]):
                history_cards[trick_index, slot_index, card] = 1.0
                history_players[trick_index, slot_index, card_player] = 1.0

        current_trick = np.zeros(32, dtype=np.float32)
        for _, card in state.current_trick.cards:
            current_trick[card] = 1.0

        current_player_one_hot = np.zeros(3, dtype=np.float32)
        current_player_one_hot[state.current_player] = 1.0

        current_trick_leader = np.zeros(3, dtype=np.float32)
        current_trick_leader[state.current_trick.leader] = 1.0

        declarer_one_hot = np.zeros(3, dtype=np.float32)
        declarer_one_hot[state.declarer] = 1.0

        game_kind = np.zeros(3, dtype=np.float32)
        if state.game_type.kind == GameKind.SUIT:
            game_kind[0] = 1.0
        elif state.game_type.kind == GameKind.GRAND:
            game_kind[1] = 1.0
        elif state.game_type.kind == GameKind.NULL:
            game_kind[2] = 1.0

        trump_suit = np.zeros(4, dtype=np.float32)
        if state.game_type.kind == GameKind.SUIT:
            trump_suit[int(state.game_type.trump_suit)] = 1.0

        trick_number = np.array(
            [len(state.completed_tricks) / 10.0],
            dtype=np.float32,
        )

        current_trick_position = np.array(
            [len(state.current_trick.cards) / 3.0],
            dtype=np.float32,
        )

        declarer_points = points_won_by_player(state.won_cards, state.declarer)
        defender_points = sum(
            points_won_by_player(state.won_cards, p)
            for p in range(3)
            if p != state.declarer
        )
        points = np.array(
            [
                declarer_points / 120.0,
                defender_points / 120.0,
            ],
            dtype=np.float32,
        )

        void_info = self._void_info()

        observation = np.concatenate(
            [
                own_hand,
                history_cards.reshape(-1),
                history_players.reshape(-1),
                current_trick,
                current_player_one_hot,
                current_trick_leader,
                declarer_one_hot,
                game_kind,
                trump_suit,
                trick_number,
                current_trick_position,
                points,
                void_info.reshape(-1),
            ]
        )

        return observation.astype(np.float32)

    def _void_info(self):
        state = self.game.state
        void_info = np.zeros((3, 5), dtype=np.float32)

        for trick in list(state.completed_tricks) + [state.current_trick]:
            if len(trick.cards) < 2:
                continue

            required_suit = effective_suit(trick.lead_card(), state.game_type)
            required_index = self._void_suit_index(required_suit)

            for player, card in trick.cards[1:]:
                if effective_suit(card, state.game_type) != required_suit:
                    void_info[player, required_index] = 1.0

        return void_info

    def _void_suit_index(self, suit):
        if suit == "TRUMP":
            return 4
        return int(suit)
