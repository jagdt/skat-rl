"""Replay explicit ISS/SkatGame decisions, with optional card-play-only extraction."""

import bz2
import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from skat_rl.engine.cards import Rank, Suit, make_card
from skat_rl.engine.game import SkatGame
from skat_rl.engine.actions import BID_VALUES, NUM_ACTIONS, discard_action
from skat_rl.engine.state import AuctionRole, GameKind, GameState, GameType, Phase, Trick
from skat_rl.envs.observations import build_observation, contract_id, encode_belief_targets


PROPERTY = re.compile(r"([A-Z][A-Z0-9]*)\[((?:\\.|[^\\\]])*)\]", re.DOTALL)
CARD = re.compile(r"[CSHD][789QKTAJ]\Z")
CONTRACT = re.compile(r"([CSHDGN])([A-Z]*)(?:\.([^.]+)\.([^.]+))?\Z")
SUITS = dict(zip("CSHD", Suit))
RANKS = dict(zip("789QKTAJ", Rank))


class RecordError(ValueError):
    def __init__(self, reason, detail=""):
        self.reason = reason
        super().__init__(detail or reason)


def open_records(path):
    path = Path(path)
    opener = bz2.open if path.suffix == ".bz2" else open
    return opener(path, "rt", encoding="utf-8")


def parse_record(text):
    """Parse escaped SGF properties, including ISS's numeric P0/R0 names."""
    text = text.strip()
    if not text.startswith("(;") or not text.endswith(";)"):
        raise RecordError("malformed_record", "Expected a complete one-line ISS record.")
    body = text[2:-2]
    properties = {}
    position = 0
    for match in PROPERTY.finditer(body):
        if body[position:match.start()].strip():
            raise RecordError("malformed_record", "Unexpected text between SGF properties.")
        key, value = match.groups()
        if key in properties:
            raise RecordError("malformed_record", f"Duplicate property: {key}")
        properties[key] = re.sub(r"\\(.)", r"\1", value, flags=re.DOTALL)
        position = match.end()
    if body[position:].strip() or properties.get("GM") != "Skat":
        raise RecordError("malformed_record")
    required = {"PC", "ID", "SE", "MV", "R", "P0", "P1", "P2"}
    if not required <= properties.keys():
        raise RecordError("missing_fields")
    return properties


def player_rating(properties, player):
    try:
        rating = float(properties.get(f"R{player}", ""))
    except ValueError:
        return None
    return rating if math.isfinite(rating) else None


def base_player_name(name):
    return re.sub(r":\d+$", "", name)


def card_id(text):
    if not CARD.fullmatch(text):
        raise RecordError("invalid_card", text)
    return make_card(SUITS[text[0]], RANKS[text[1]])


@dataclass
class RecordedGame:
    game: SkatGame
    plays: list
    result: dict
    outcome: str
    deal_key: str
    bid_thresholds: dict = field(default_factory=dict)


def decode_game(properties, game_kinds=("suit", "grand", "null"), full_game=False):
    result_tokens = properties["R"].split()
    passed = "passed" in result_tokens
    if passed and not full_game:
        raise RecordError("passed")
    if "penalty" in result_tokens or ("overbid" in result_tokens and not full_game):
        raise RecordError("penalty_or_overbid")
    result = dict(token.split(":", 1) for token in result_tokens if ":" in token)
    if not passed and (("bidok" not in result_tokens and "overbid" not in result_tokens)
                       or not {"d", "p", "t", "to", "l", "r"} <= result.keys()):
        raise RecordError("unsupported_result")
    try:
        for key in (() if passed else ("d", "p", "t", "to", "l", "r")):
            int(result[key])
    except ValueError as error:
        raise RecordError("unsupported_result", "Non-integer result field.") from error
    if result.get("to", "-1") != "-1" or result.get("l", "-1") != "-1":
        raise RecordError("timeout_or_disconnect")
    if result.get("r", "0") != "0":
        raise RecordError("resignation")
    outcome = next((word for word in result_tokens if word in {"win", "loss"}), None)
    if outcome is None and not passed:
        raise RecordError("unsupported_result")
    tokens = properties["MV"].split()
    if len(tokens) % 2 or len(tokens) < 2 or tokens[0] != "w":
        raise RecordError("malformed_moves")
    moves = list(zip(tokens[::2], tokens[1::2]))
    deck = [card_id(card) for card in moves[0][1].split(".")]
    if len(deck) != 32 or len(set(deck)) != 32:
        raise RecordError("invalid_deal")
    hands = [set(deck[p * 10:(p + 1) * 10]) for p in range(3)]
    skat = deck[30:]
    # Canonicalize card order so duplicate deals cannot cross the data split.
    canonical = [sorted(hand) for hand in hands] + [sorted(skat)]
    deal_key = hashlib.sha256(bytes(card for group in canonical for card in group)).hexdigest()
    if passed:
        game = SkatGame()
        game.reset_full_from_deal(hands, skat)
        if moves[1:] != [("1", "p"), ("2", "p"), ("0", "p")]:
            raise RecordError("invalid_passed_auction")
        return RecordedGame(game, [(1, 0), (2, 0), (0, 0)], {}, "passed", deal_key)
    picked_up = None
    revealed = False
    declarer = None
    game_type = None
    plays = []
    bids = []
    for actor, action in moves[1:]:
        if action.split(".")[0] in {"SC", "RE", "TI", "LE"}:
            raise RecordError("reveal_or_shortened_game")
        if actor == "w":
            if picked_up is None or revealed or game_type is not None:
                raise RecordError("unexpected_server_move")
            if sorted(card_id(c) for c in action.split(".")) != sorted(skat):
                raise RecordError("invalid_skat_pickup")
            revealed = True
            continue
        if actor not in {"0", "1", "2"}:
            raise RecordError("invalid_player")
        player = int(actor)
        if game_type is not None:
            plays.append((player, card_id(action)))
        elif action == "s":
            if picked_up is not None:
                raise RecordError("invalid_skat_pickup")
            picked_up = player
            hands[player].update(skat)
        elif action in {"p", "y"} or action.isdecimal():
            if picked_up is not None:
                raise RecordError("unexpected_bid")
            bids.append((player, action))
        else:
            match = CONTRACT.fullmatch(action)
            if match is None:
                raise RecordError("unsupported_contract", action)
            kind, flags, discard1, discard2 = match.groups()
            if flags not in {"", "H"}:
                raise RecordError("unsupported_contract", action)
            game_type = (GameType(GameKind.SUIT, SUITS[kind]) if kind in SUITS
                         else GameType(GameKind.GRAND if kind == "G" else GameKind.NULL))
            game_type.hand = flags == "H"
            if game_type.kind.value not in game_kinds:
                raise RecordError("excluded_game_kind")
            declarer = player
            if result["d"] != str(declarer):
                raise RecordError("declarer_mismatch")
            if flags == "H":
                if picked_up is not None or discard1 is not None:
                    raise RecordError("invalid_hand_contract")
            else:
                if picked_up != player or not revealed or discard1 is None:
                    raise RecordError("invalid_discard")
                skat = [card_id(discard1), card_id(discard2)]
                if len(set(skat)) != 2 or not set(skat) <= hands[player]:
                    raise RecordError("invalid_discard")
                hands[player].difference_update(skat)

    if game_type is None or not (0 < len(plays) <= 30) or (game_type.kind != GameKind.NULL and len(plays) != 30):
        raise RecordError("incomplete_card_play")
    game = SkatGame()
    game.state = GameState(
        hands=hands, skat=skat, declarer=declarer, game_type=game_type,
        current_player=0, current_trick=Trick(leader=0),
    )
    record = RecordedGame(game, plays, result, outcome, deal_key)
    if full_game:
        _include_setup(record, deck, bids)
    return record


def _include_setup(record, deck, bids):
    """Recover phase actions without inventing unrecorded intermediate bids."""
    if not bids:
        raise RecordError("missing_auction")
    final = record.game.state
    game = SkatGame()
    hands = [deck[p * 10:(p + 1) * 10] for p in range(3)]
    game.reset_full_from_deal(hands, deck[30:])
    setup = []

    def append(player, action):
        if player != game.state.current_player or action not in game.legal_actions():
            raise RecordError("invalid_setup", f"Player {player}, action {action}")
        setup.append((player, action))
        game.step(action)

    for player, bid in bids:
        state = game.state
        if state.phase != Phase.BIDDING:
            raise RecordError("unexpected_bid")
        if bid.isdecimal():
            value = int(bid)
            if (state.auction_role != AuctionRole.CALLER or value not in BID_VALUES
                    or value < BID_VALUES[state.bid_index]):
                raise RecordError("invalid_bid_threshold")
            # A jump is one recorded CONTINUE decision at its actual threshold.
            state.bid_index = BID_VALUES.index(value)
            record.bid_thresholds[len(setup)] = state.bid_index
        elif bid == "y" and state.auction_role != AuctionRole.HOLDER:
            raise RecordError("invalid_bid_role")
        append(player, int(bid != "p"))
    if game.state.phase == Phase.BIDDING and game.state.forehand_offer:
        # Playing a contract after both opponents pass is forehand accepting 18.
        append(final.declarer, 1)
    if game.state.phase != Phase.PICKUP_DECISION or game.state.declarer != final.declarer:
        raise RecordError("declarer_mismatch")
    append(final.declarer, int(final.game_type.hand))
    if not final.game_type.hand:
        append(final.declarer, discard_action(game.state.hands[final.declarer], final.skat))
    append(final.declarer, int(contract_id(final.game_type)))
    record.plays = setup + record.plays
    record.game.reset_full_from_deal(hands, deck[30:])


def replay_examples(record, eligible_players, role="both", include_forced=True):
    """Buffer at most one game's examples; reject the whole game on replay errors."""
    game = record.game
    examples = []
    example_players = []
    remaining = Counter(player for player, _ in record.plays)
    final_declarer = int(record.result.get("d", -1))
    for index, (player, action) in enumerate(record.plays):
        if index in record.bid_thresholds:
            game.state.bid_index = record.bid_thresholds[index]
        remaining[player] -= 1
        if player != game.state.current_player or action not in game.legal_actions():
            raise RecordError("illegal_recorded_move", f"Player {player}, card {action}")
        legal = game.legal_actions()
        is_declarer = player == final_declarer
        role_matches = role == "both" or (role == "declarer") == is_declarer
        if eligible_players[player] and role_matches and (include_forced or len(legal) > 1):
            mask = np.zeros(NUM_ACTIONS, dtype=bool)
            mask[legal] = True
            examples.append({
                "observations": build_observation(game.state, player),
                "action_masks": mask,
                "actions": action,
                "belief_targets": encode_belief_targets(game.state, player),
                "remaining_decisions": remaining[player],
            })
            example_players.append(player)
        step = game.step(action)
    if not record.plays or not game.state.terminated:
        raise RecordError("incomplete_card_play")
    if record.outcome == "passed":
        for example in examples:
            example["terminal_rewards"] = 0.0
        return examples
    result = step.info["result"]
    if ((game.state.game_type.kind != GameKind.NULL and result["declarer_points"] != int(record.result["p"]))
            or result["declarer_won"] != (record.outcome == "win")
            or sum(p == game.state.declarer for p in game.state.trick_winners)
            != int(record.result["t"])):
        raise RecordError("result_mismatch")
    if "v" in record.result:
        try:
            recorded_value = int(record.result["v"])
        except ValueError as error:
            raise RecordError("unsupported_result", "Non-integer game value.") from error
        signed_value = result["game_value"] * (1 if result["declarer_won"] else -2)
        if recorded_value != signed_value:
            raise RecordError("game_value_mismatch")
    for example, player in zip(examples, example_players):
        example["terminal_rewards"] = step.reward[player]
    return examples
