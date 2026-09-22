import numpy as np
from gymnasium import spaces

from skat_rl._skat_cpp import BatchedFastSkatEnv


class SkatCppBatchedSingleAgentEnv:
    """
    Batched C++ single-agent Skat environment for PPO rollout collection.

    The batch owns `rollout_size` independent C++ games. A reset starts one
    game per slot and autoplays opponents until the learning player's first
    decision. Each `step(actions)` consumes one action for every active game,
    then autoplays opponents in C++ until each active game is either terminal
    or back at the learning player's turn.

    With `autoplay_opponents=False`, reset does not play any cards and step
    plays exactly one card per active game. Observations, masks and belief
    targets belong to `current_players`, which can differ across games.
    Rewards always belong to `learning_player`; episode lengths count only
    that player's decisions. The caller supplies actions for all seats.
    """

    def __init__(self, rollout_size, learning_player=0, fixed_declarer=None, seed=42,
                 autoplay_opponents=True):
        self.rollout_size = int(rollout_size)
        self.learning_player = int(learning_player)
        self.fixed_declarer = -1 if fixed_declarer is None else int(fixed_declarer)
        self.seed_value = None if seed is None else int(seed)
        self.autoplay_opponents = bool(autoplay_opponents)
        if self.rollout_size < 1:
            raise ValueError("rollout_size must be at least 1.")
        self.game = BatchedFastSkatEnv(
            self.rollout_size,
            self.learning_player,
            self.fixed_declarer,
            self.autoplay_opponents,
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.game.observation_dim(),),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.game.action_dim())

    def reset(self, seed=None):
        if seed is None:
            seed = self.seed_value
        self.game.reset_many(self._env_seeds(seed))
        if seed is not None:
            self.seed_value = int(seed) + 1
        return self._state()

    def step(self, actions):
        actions = np.asarray(actions)
        if actions.shape != (self.active_count(),):
            raise ValueError("actions must contain one action per active environment.")
        if actions.size and not np.issubdtype(actions.dtype, np.integer):
            raise ValueError("actions must be integers.")
        result = self.game.step(actions.tolist())
        return self._format_step_result(result)

    def active_count(self):
        return int(self.game.active_count())

    def close(self):
        pass

    def _env_seeds(self, seed):
        if seed is None:
            seed_sequence = np.random.SeedSequence()
        else:
            seed_sequence = np.random.SeedSequence(int(seed))
        return [
            int(child.generate_state(1, dtype=np.uint64)[0])
            for child in seed_sequence.spawn(self.rollout_size)
        ]

    def _state(self):
        return {
            "active_indices": np.asarray(self.game.active_indices(), dtype=np.int64),
            "current_players": np.asarray(self.game.current_players(), dtype=np.int64),
            "declarers": np.asarray(self.game.declarers(), dtype=np.int64),
            "observations": np.asarray(self.game.observations(), dtype=np.float32),
            "action_masks": np.asarray(self.game.action_masks(), dtype=bool),
            "belief_targets": np.asarray(self.game.belief_targets(), dtype=np.int64),
        }

    def _format_step_result(self, result):
        return {
            "env_indices": np.asarray(result["env_indices"], dtype=np.int64),
            "rewards": np.asarray(result["rewards"], dtype=np.float32),
            "terminated": np.asarray(result["terminated"], dtype=bool),
            "completed_env_indices": np.asarray(result["completed_env_indices"], dtype=np.int64),
            "completed_returns": np.asarray(result["completed_returns"], dtype=np.float32),
            "completed_lengths": np.asarray(result["completed_lengths"], dtype=np.int64),
            "active_indices": np.asarray(result["active_indices"], dtype=np.int64),
            "current_players": np.asarray(result["current_players"], dtype=np.int64),
            "declarers": np.asarray(result["declarers"], dtype=np.int64),
            "observations": np.asarray(result["observations"], dtype=np.float32),
            "action_masks": np.asarray(result["action_masks"], dtype=bool),
            "belief_targets": np.asarray(result["belief_targets"], dtype=np.int64),
        }
