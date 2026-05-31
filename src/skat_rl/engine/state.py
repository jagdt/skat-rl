from dataclasses import dataclass, field
from enum import Enum


class GameKind(str, Enum):
    SUIT = "suit"
    GRAND = "grand"
    NULL = "null"


@dataclass
class GameType:
    kind: GameKind
    trump_suit: object = None


@dataclass
class Trick:
    leader: int
    cards: list = field(default_factory=list)  # list of (player_id, card)

    def is_complete(self):
        return len(self.cards) == 3

    def lead_card(self):
        if not self.cards:
            raise ValueError("Trick has no lead card.")
        return self.cards[0][1]


@dataclass
class GameState:
    hands: list
    skat: list
    declarer: int
    game_type: object
    current_player: int
    current_trick: object
    completed_tricks: list = field(default_factory=list)
    won_cards: list = field(default_factory=lambda: [[], [], []])
    trick_winners: list = field(default_factory=list)
    terminated: bool = False

    def clone_public_for_player(self, player_id):
        return {
            "player_id": player_id,
            "own_hand": sorted(self.hands[player_id]),
            "declarer": self.declarer,
            "game_type": self.game_type,
            "current_player": self.current_player,
            "current_trick": list(self.current_trick.cards),
            "completed_tricks": [
                {
                    "leader": trick.leader,
                    "cards": list(trick.cards),
                }
                for trick in self.completed_tricks
            ],
            "trick_winners": list(self.trick_winners),
            "terminated": self.terminated,
        }