import numpy as np
import gymnasium as gym
from gymnasium import spaces

from skat_rl.engine.game import SkatGame
from skat_rl.engine.rules import effective_suit
from skat_rl.envs.observations import encode_observation, encode_belief_targets
from skat_rl.agents.heuristic_agent import HeuristicAgent
from skat_rl.agents.random_agent import RandomAgent


class SkatSingleAgentEnv(gym.Env):
    """
    Python-engine Gymnasium environment for training one RL-controlled player.

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

    def __init__(self, learning_player=0, opponent_agents=None, fixed_declarer=None, seed=None):
        super().__init__()

        self.learning_player = learning_player
        self.seed_value = seed
        self.fixed_declarer = fixed_declarer

        self.game = SkatGame(fixed_declarer=self.fixed_declarer, seed=seed)
        self._reset_void_info_cache()

        if opponent_agents is None:
            opponent_agents = [HeuristicAgent() for _ in range(3)]
            opponent_agents[learning_player] = None

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

        self._reset_void_info_cache()
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

        self._update_void_info_for_action(self.learning_player, action)
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

    def belief_targets(self):
        """Hidden-card locations relative to the learning player."""
        return encode_belief_targets(self.game.state, self.learning_player)

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
            self._update_void_info_for_action(player, action)
            step_result = self.game.step(action)

            total_reward += step_result.reward[self.learning_player]
            info.update(step_result.info)

        return total_reward, info

    def _get_observation(self):
        return encode_observation(
            self.game.state, self.learning_player, self._void_info()
        )

    def _void_info(self):
        return self._void_info_cache

    def _reset_void_info_cache(self):
        self._void_info_cache = np.zeros((3, 5), dtype=np.float32)
        self._void_info_card_count = 0

    def _update_void_info_for_action(self, player, action):
        state = self.game.state
        if state is None or state.terminated:
            return

        trick = state.current_trick
        if trick.cards:
            required_suit = effective_suit(trick.lead_card(), state.game_type)
            required_index = self._void_suit_index(required_suit)

            if effective_suit(action, state.game_type) != required_suit:
                self._void_info_cache[player, required_index] = 1.0


    def _void_suit_index(self, suit):
        if suit == "TRUMP":
            return 4
        return int(suit)
