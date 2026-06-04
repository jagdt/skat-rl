from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.monitor import Monitor

from skat_rl.envs.skat_sb3_env import SkatSingleAgentEnv


def main():
    env = SkatSingleAgentEnv(
        learning_player=0,
        seed=42,
    )

    env = Monitor(env)

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    model.learn(
        total_timesteps=200_000,
        progress_bar=True,
    )

    model.save("models/maskable_ppo_skat_player0")


if __name__ == "__main__":
    main()