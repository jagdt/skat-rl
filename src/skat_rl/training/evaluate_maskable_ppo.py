# src/skat_rl/training/evaluate_maskable_ppo.py

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from skat_rl.envs.skat_sb3_env import SkatSingleAgentEnv


def evaluate(model_path, n_games=1000):
    env = SkatSingleAgentEnv(
        learning_player=0,
        seed=123,
    )

    model = MaskablePPO.load(model_path)

    wins = 0
    games_finished = 0
    total_reward = 0.0
    declarer_games = 0
    defender_games = 0
    declarer_wins = 0
    defender_wins = 0

    for game_idx in range(n_games):
        obs, info = env.reset(seed=game_idx)

        done = False
        episode_reward = 0.0

        was_declarer = env.game.state.declarer == env.learning_player

        while not done:
            action_masks = get_action_masks(env)

            action, _ = model.predict(
                obs,
                action_masks=action_masks,
                deterministic=True,
            )

            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            done = terminated or truncated

        result = info.get("result")

        if result is None:
            continue

        games_finished += 1
        total_reward += episode_reward

        player_won = False

        if was_declarer:
            declarer_games += 1
            if result["declarer_won"]:
                player_won = True
                declarer_wins += 1
        else:
            defender_games += 1
            if not result["declarer_won"]:
                player_won = True
                defender_wins += 1

        if player_won:
            wins += 1

    print(f"Games evaluated: {games_finished}")
    print(f"Player 0 total win rate: {wins / games_finished:.3f}")
    print(f"Average episode reward: {total_reward / games_finished:.3f}")

    if declarer_games > 0:
        print(f"Player 0 declarer games: {declarer_games}")
        print(f"Player 0 declarer win rate: {declarer_wins / declarer_games:.3f}")

    if defender_games > 0:
        print(f"Player 0 defender games: {defender_games}")
        print(f"Player 0 defender win rate: {defender_wins / defender_games:.3f}")


def main():
    evaluate(
        model_path="models/maskable_ppo_skat_player0",
        n_games=1000,
    )


if __name__ == "__main__":
    main()