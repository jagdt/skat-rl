import argparse

from skat_rl.engine.game import SkatGame
from skat_rl.engine.state import GameKind, GameType
from skat_rl.engine.cards import Suit, card_str, cards_str, sort_cards
from skat_rl.agents.random_agent import RandomAgent
from skat_rl.agents.heuristic_agent import HeuristicAgent


def play_one_game(seed, verbose=False):
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

    if verbose:
        _print_game_header(game, seed)

    while not game.state.terminated:
        player = game.state.current_player
        obs = game.observe(player)
        legal_actions = game.legal_actions(player)
        action = agents[player].act(obs, legal_actions)

        if verbose:
            _print_action(game, player, action, legal_actions)

        result = game.step(action)

        if verbose and "trick_winner" in result.info:
            _print_trick_result(game, result.info)

    if verbose:
        _print_game_result(result.info["result"])

    return result.info["result"]


def main():
    args = _parse_args()
    n_games = args.games
    declarer_count = [0, 0, 0]
    player_wins = [0, 0, 0]
    player_points = [0, 0, 0]

    for seed in range(n_games):
        result = play_one_game(args.seed + seed, verbose=args.verbose)

        declarer = result["declarer"]
        declarer_count[declarer] += 1

        if result["declarer_won"]:
            player_wins[declarer] += 1

        player_points[declarer] += result["declarer_points"]

    print(f"Games: {n_games}")
    for player in range(3):
        print(f"Player {player}:")
        if declarer_count[player] == 0:
            print("  Declarer games: 0")
            continue
        print(f"  Declarer games: {declarer_count[player]}")
        print(f"  Win rate: {player_wins[player] / declarer_count[player]:.3f}")
        print(f"  Average points: {player_points[player] / declarer_count[player]:.2f}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Skat agents.")
    parser.add_argument(
        "-n",
        "--games",
        type=int,
        default=10000,
        help="Number of games to play.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the first game. Later games use consecutive seeds.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print each hand, play, trick result, and game result.",
    )
    return parser.parse_args()


def _print_game_header(game, seed):
    print()
    print("=" * 72)
    print(f"Game seed {seed}")
    print(f"Game type: {_format_game_type(game.state.game_type)}")
    print(f"Declarer: Player {game.state.declarer}")
    print(f"Skat: {cards_str(game.state.skat)}")
    print()
    for player, hand in enumerate(game.state.hands):
        role = _player_role(player, game.state.declarer)
        print(f"Player {player} ({role}) hand: {cards_str(sort_cards(hand))}")
    print()


def _print_action(game, player, action, legal_actions):
    trick_no = len(game.state.completed_tricks) + 1
    trick_cards = game.state.current_trick.cards

    if not trick_cards:
        print(f"Trick {trick_no}: Player {player} leads")

    role = _player_role(player, game.state.declarer)
    legal = cards_str(legal_actions)
    print(f"  Player {player} ({role}) plays {card_str(action)}")
    print(f"    Legal: {legal}")


def _print_trick_result(game, info):
    trick = game.state.completed_tricks[-1]
    cards = "  ".join(
        f"P{player}:{card_str(card)}"
        for player, card in trick.cards
    )
    winner = info["trick_winner"]
    print(f"  Trick cards: {cards}")
    print(f"  Winner: Player {winner} for {info['trick_points']} points")
    print()


def _print_game_result(result):
    print("Game result")
    print(f"  Declarer: Player {result['declarer']}")
    print(f"  Declarer points: {result['declarer_points']}")
    print(f"  Defender points: {result['defender_points']}")
    print(f"  Declarer won: {result['declarer_won']}")
    print(f"  Schneider: {result['schneider']}")
    print(f"  Schwarz: {result['schwarz']}")


def _format_game_type(game_type):
    if game_type.kind == GameKind.SUIT:
        trump_suit = Suit(game_type.trump_suit)
        return f"{game_type.kind.value}, trump {trump_suit.name}"
    return game_type.kind.value


def _player_role(player, declarer):
    if player == declarer:
        return "declarer"
    return "defender"

if __name__ == "__main__":
    main()
