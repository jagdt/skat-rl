"""Read the one-record-per-line ISS/SkatGame SGF dialect and replay card play."""

import bz2
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from skat_rl.engine.cards import Rank, Suit, make_card
from skat_rl.engine.game import SkatGame
from skat_rl.engine.state import GameKind, GameState, GameType, Trick
from skat_rl.envs.observations import encode_observation, encode_belief_targets


PROPERTY = re.compile(r"([A-Z][A-Z0-9]*)\[((?:\\.|[^\\\]])*)\]", re.DOTALL)
CARD = re.compile(r"[CSHD][789QKTAJ]\Z")
CONTRACT = re.compile(r"([CSHDGN])([A-Z]*)(?:\.([^.]+)\.([^.]+))?\Z")
SUITS = dict(zip("CSHD", Suit))
RANKS = dict(zip("789QKTAJ", Rank))
TRUSTED_PLAYERS = ("kermit", "zoot", "theCount")


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


def decode_game(properties, game_kinds=("suit", "grand")):
    result_tokens = properties["R"].split()
    if "passed" in result_tokens:
        raise RecordError("passed")
    if "penalty" in result_tokens or "overbid" in result_tokens:
        raise RecordError("penalty_or_overbid")
    result = dict(token.split(":", 1) for token in result_tokens if ":" in token)
    if "bidok" not in result_tokens or not {"d", "p", "t", "to", "l", "r"} <= result.keys():
        raise RecordError("unsupported_result")
    try:
        for key in ("d", "p", "t", "to", "l", "r"):
            int(result[key])
    except ValueError as error:
        raise RecordError("unsupported_result", "Non-integer result field.") from error
    if result["to"] != "-1" or result["l"] != "-1":
        raise RecordError("timeout_or_disconnect")
    if result["r"] != "0":
        raise RecordError("resignation")
    outcome = next((word for word in result_tokens if word in {"win", "loss"}), None)
    if outcome is None:
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
    picked_up = None
    revealed = False
    declarer = None
    game_type = None
    plays = []
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
        else:
            match = CONTRACT.fullmatch(action)
            if match is None:
                raise RecordError("unsupported_contract", action)
            kind, flags, discard1, discard2 = match.groups()
            if flags not in {"", "H"}:
                raise RecordError("unsupported_contract", action)
            game_type = (GameType(GameKind.SUIT, SUITS[kind]) if kind in SUITS
                         else GameType(GameKind.GRAND if kind == "G" else GameKind.NULL))
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

    if game_type is None or len(plays) != 30:
        raise RecordError("incomplete_card_play")
    game = SkatGame()
    game.state = GameState(
        hands=hands, skat=skat, declarer=declarer, game_type=game_type,
        current_player=0, current_trick=Trick(leader=0),
    )
    return RecordedGame(game, plays, result, outcome, deal_key)


def replay_examples(record, eligible_players, role="both", include_forced=False):
    """Buffer at most one game's examples; reject the whole game on replay errors."""
    game = record.game
    examples = []
    for player, action in record.plays:
        if player != game.state.current_player or action not in game.legal_actions():
            raise RecordError("illegal_recorded_move", f"Player {player}, card {action}")
        legal = game.legal_actions()
        is_declarer = player == game.state.declarer
        role_matches = role == "both" or (role == "declarer") == is_declarer
        if eligible_players[player] and role_matches and (include_forced or len(legal) > 1):
            mask = np.zeros(32, dtype=bool)
            mask[legal] = True
            examples.append({
                "observations": encode_observation(game.state, player),
                "action_masks": mask,
                "actions": action,
                "belief_targets": encode_belief_targets(game.state, player),
            })
        step = game.step(action)
    result = step.info["result"]
    if (result["declarer_points"] != int(record.result["p"])
            or result["declarer_won"] != (record.outcome == "win")
            or sum(p == game.state.declarer for p in game.state.trick_winners)
            != int(record.result["t"])):
        raise RecordError("result_mismatch")
    return examples
