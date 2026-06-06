# src/skat_rl/training/evaluate_maskable_ppo.py

import argparse

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
    declarer_games_by_player = [0, 0, 0]
    declarer_wins_by_player = [0, 0, 0]

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
        declarer = result["declarer"]
        declarer_games_by_player[declarer] += 1

        if result["declarer_won"]:
            declarer_wins_by_player[declarer] += 1

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

    if declarer_games > 0:
        print(f"Player 0 declarer games: {declarer_games}")
        print(f"Player 0 declarer win rate: {declarer_wins / declarer_games:.3f}")

    if defender_games > 0:
        print(f"Player 0 defender games: {defender_games}")
        print(f"Player 0 defender win rate: {defender_wins / defender_games:.3f}")

    for player in (1, 2):
        games = declarer_games_by_player[player]
        if games > 0:
            win_rate = declarer_wins_by_player[player] / games
            print(f"Player {player} declarer games: {games}")
            print(f"Player {player} declarer win rate: {win_rate:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", required=True)
    args = parser.parse_args()

    evaluate(
        model_path=f"models/maskable_ppo_skat_player0_{args.timestamp}/model.zip",
        n_games=1000,
    )


if __name__ == "__main__":
    main()