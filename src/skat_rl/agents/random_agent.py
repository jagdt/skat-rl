import random


class RandomAgent:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def act(self, observation, legal_actions):
        if not legal_actions:
            raise ValueError("No legal actions available.")
        return self.rng.choice(legal_actions)