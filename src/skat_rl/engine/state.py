from dataclasses import dataclass, field
from enum import Enum, IntEnum


class Phase(IntEnum):
    BIDDING = 0
    PICKUP_DECISION = 1
    DISCARD = 2
    CONTRACT_SELECTION = 3
    CARD_PLAY = 4
    TERMINAL = 5


class Seat(IntEnum):
    FOREHAND = 0
    MIDDLEHAND = 1
    REARHAND = 2


class AuctionRole(IntEnum):
    CALLER = 0
    HOLDER = 1


class BiddingStatus(IntEnum):
    NOT_ENTERED = 0
    ACTIVE = 1
    PASSED = 2


class GameKind(str, Enum):
    SUIT = "suit"
    GRAND = "grand"
    NULL = "null"


@dataclass
class GameType:
    kind: GameKind
    trump_suit: object = None
    hand: bool = False


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
    phase: Phase = Phase.CARD_PLAY
    forehand: int = 0
    hand_game: int = -1
    winning_bid: int = 0
    bid_status: list = field(default_factory=lambda: [BiddingStatus.NOT_ENTERED] * 3)
    highest_called: list = field(default_factory=lambda: [-1] * 3)
    highest_held: list = field(default_factory=lambda: [-1] * 3)
    pass_threshold: list = field(default_factory=lambda: [-1] * 3)
    pass_role: list = field(default_factory=lambda: [-1] * 3)
    auction_caller: int = 1
    auction_holder: int = 0
    auction_role: AuctionRole = AuctionRole.CALLER
    bid_index: int = 0
    rearhand_entered: bool = False
    forehand_offer: bool = False

    def __post_init__(self):
        if self.game_type is not None:
            self.hand_game = int(self.game_type.hand)

    @property
    def terminated(self):
        return self.phase == Phase.TERMINAL

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
            "phase": self.phase,
            "bid_index": self.bid_index,
        }
