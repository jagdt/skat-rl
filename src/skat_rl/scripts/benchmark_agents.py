from skat_rl.engine.game import SkatGame
from skat_rl.engine.state import GameKind, GameType
from skat_rl.engine.cards import Suit
from skat_rl.agents.random_agent import RandomAgent
from skat_rl.agents.heuristic_agent import HeuristicAgent


def play_one_game(seed):
    game = SkatGame(
        game_type=GameType(GameKind.SUIT, trump_suit=Suit.CLUBS),
        declarer=0,
        seed=seed,
    )

    agents = [
        HeuristicAgent(),
        RandomAgent(seed=seed + 1),
        RandomAgent(seed=seed + 2),
    ]

    game.reset(seed=seed)

    while not game.state.terminated:
        player = game.state.current_player
        obs = game.observe(player)
        legal_actions = game.legal_actions(player)
        action = agents[player].act(obs, legal_actions)
        result = game.step(action)

    return result.info["result"]


def main():
    n_games = 10000
    declarer_count = [0, 0, 0]
    player_wins = [0, 0, 0]
    player_points = [0, 0, 0]

    for seed in range(n_games):
        result = play_one_game(seed)

        declarer = result["declarer"]
        declarer_count[declarer] += 1

        if result["declarer_won"]:
            player_wins[declarer] += 1

        player_points[declarer] += result["declarer_points"]

    print(f"Games: {n_games}")
    for player in range(3):
        print(f"Player {player}:")
        print(f"  Win rate: {player_wins[player] / declarer_count[player]:.3f}")
        print(f"  Average points: {player_points[player] / declarer_count[player]:.2f}")

if __name__ == "__main__":
    main()