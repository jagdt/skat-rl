import numpy as np
import gymnasium as gym
from gymnasium import spaces

from skat_rl._skat_cpp import FastSkatGame
from skat_rl.agents.heuristic_agent import HeuristicAgent
from skat_rl.engine.state import GameKind, GameType


class SkatCppSingleAgentEnv(gym.Env):
    """
    Gymnasium environment backed by the C++ single-game card-play engine.

    The C++ engine owns game mechanics, legal masks, points, void inference,
    and observation-vector construction. Python still owns reward shaping and
    opponent action selection.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, learning_player=0, opponent_agents=None, fixed_declarer=None, seed=None):
        super().__init__()

        self.learning_player = learning_player
        self.seed_value = seed
        self.fixed_declarer = fixed_declarer
        self.game = FastSkatGame()

        if opponent_agents is None:
            opponent_agents = [HeuristicAgent() for _ in range(3)]
            opponent_agents[learning_player] = None

        self.opponent_agents = opponent_agents
        self.action_space = spaces.Discrete(32)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1149,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        reset_seed = self.seed_value if seed is None else seed
        if reset_seed is None:
            reset_seed = 0
        self.seed_value = int(reset_seed) + 1

        if self.fixed_declarer is None:
            self.game.reset(int(reset_seed))
        else:
            self.game.reset_fixed_declarer(int(reset_seed), int(self.fixed_declarer))

        self._play_until_learning_player()

        return self._get_observation(), {}

    def step(self, action):
        action = int(action)

        if self.game.is_terminal():
            return self._get_observation(), 0.0, True, False, {}

        if self.game.current_player() != self.learning_player:
            raise RuntimeError("It is not currently the learning player's turn.")

        if action not in self.game.legal_actions():
            raise ValueError(
                f"Illegal action {action}. Legal actions are {self.game.legal_actions()}."
            )

        step_result = self.game.step(action)
        reward = self._reward(step_result)
        terminated = step_result["terminated"]
        info = self._info(step_result)

        if not terminated:
            opponent_reward, opponent_info = self._play_until_learning_player()
            reward += opponent_reward
            info.update(opponent_info)

        return self._get_observation(), reward, self.game.is_terminal(), False, info

    def action_masks(self):
        if self.game.is_terminal() or self.game.current_player() != self.learning_player:
            return np.zeros(32, dtype=bool)

        return np.array(self.game.legal_mask_array(), dtype=bool)

    def render(self):
        summary = self.game.state_summary()
        print(f"Current player: {summary['current_player']}")
        print(f"Declarer: {summary['declarer']}")
        print(f"Completed tricks: {summary['trick_index']}")
        print(f"Current trick: {list(zip(summary['current_trick_players'], summary['current_trick_cards']))}")

    def _get_observation(self):
        return np.array(
            self.game.observation(self.learning_player),
            dtype=np.float32,
        )

    def _play_until_learning_player(self):
        total_reward = 0.0
        info = {}

        while (
            not self.game.is_terminal()
            and self.game.current_player() != self.learning_player
        ):
            player = self.game.current_player()
            obs = self._opponent_observation(player)
            legal = self.game.legal_actions()

            action = self.opponent_agents[player].act(obs, legal)
            step_result = self.game.step(action)

            total_reward += self._reward(step_result)
            info.update(self._info(step_result))

        return total_reward, info

    def _reward(self, step_result):
        if not step_result["terminated"]:
            return 0.0

        declarer = self.game.declarer()
        declarer_won = step_result["declarer_won"]
        declarer_points = step_result["declarer_points"]

        if declarer_won:
            declarer_reward = 1.0 + 0.2 * (declarer_points - 60) / 60.0
        else:
            declarer_reward = -1.0 - 0.2 * (60 - declarer_points) / 60.0

        if self.learning_player == declarer:
            return float(declarer_reward)
        return float(-declarer_reward / 2.0)

    def _info(self, step_result):
        if not step_result["terminated"]:
            return {}

        return {
            "result": {
                "declarer": self.game.declarer(),
                "declarer_won": step_result["declarer_won"],
                "declarer_points": step_result["declarer_points"],
                "defender_points": step_result["defender_points"],
                "schneider": False,
                "schwarz": False,
            }
        }

    def _opponent_observation(self, player):
        return {
            "player_id": player,
            "own_hand": self.game.hand(player),
            "declarer": self.game.declarer(),
            "game_type": self._game_type(),
            "current_player": self.game.current_player(),
            "current_trick": self._current_trick(),
            "completed_tricks": [
                {
                    "leader": trick[0][0],
                    "cards": trick,
                }
                for trick in self._completed_tricks()
            ],
            "trick_winners": [],
            "terminated": self.game.is_terminal(),
        }

    def _completed_tricks(self):
        return [
            list(zip(players, cards))
            for players, cards in zip(
                self.game.history_players(),
                self.game.history_cards(),
            )
        ]

    def _current_trick(self):
        return list(zip(
            self.game.current_trick_players(),
            self.game.current_trick_cards(),
        ))

    def _game_type(self):
        game_kind = self.game.game_kind()
        if game_kind == 0:
            return GameType(GameKind.SUIT, trump_suit=self.game.trump_suit())
        if game_kind == 1:
            return GameType(GameKind.GRAND)
        return GameType(GameKind.NULL)
