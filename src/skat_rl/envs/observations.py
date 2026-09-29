"""Semantic policy observations. Hidden-card labels are a separate API.

LEFT is the next seat in engine play order: (observer + 1) % 3.
RIGHT is (observer + 2) % 3. No absolute seat IDs enter the policy input.
"""

from enum import IntEnum
from typing import TypedDict

import numpy as np
from gymnasium import spaces

from skat_rl.engine.cards import Suit
from skat_rl.engine.rules import effective_suit, points_won_by_player
from skat_rl.engine.state import GameKind, GameType


DATASET_FORMAT_VERSION = 3
NUM_CARDS = 32
NUM_PLAYERS = 3
NUM_TRICKS = 10
UNPLAYED = -1


class Phase(IntEnum):
    CARD_PLAY = 0
    TERMINAL = 1


class CardStatus(IntEnum):
    UNKNOWN = 0
    OWN = 1
    PLAYED = 2


class RelativePlayer(IntEnum):
    SELF = 0
    LEFT = 1
    RIGHT = 2


class TrickSlot(IntEnum):
    LEAD = 0
    SECOND = 1
    THIRD = 2


class Contract(IntEnum):
    CLUBS = 0
    SPADES = 1
    HEARTS = 2
    DIAMONDS = 3
    GRAND = 4
    NULL = 5


class StructuredSkatObservation(TypedDict):
    # Each array may have leading batch dimensions; scalar fields have shape ().
    phase: np.ndarray
    card_status: np.ndarray
    played_by: np.ndarray
    trick_index: np.ndarray
    trick_slot: np.ndarray
    contract: np.ndarray
    relative_declarer: np.ndarray
    relative_current_leader: np.ndarray
    declarer_points: np.ndarray
    defender_points: np.ndarray
    current_trick: np.ndarray
    void_info: np.ndarray


# Per-example shape and inclusive bounds. Compact int8 covers every field.
OBSERVATION_SPECS = {
    "phase": ((), 0, len(Phase) - 1),
    "card_status": ((NUM_CARDS,), 0, len(CardStatus) - 1),
    "played_by": ((NUM_CARDS,), UNPLAYED, NUM_PLAYERS - 1),
    "trick_index": ((NUM_CARDS,), UNPLAYED, NUM_TRICKS - 1),
    "trick_slot": ((NUM_CARDS,), UNPLAYED, len(TrickSlot) - 1),
    "contract": ((), 0, len(Contract) - 1),
    "relative_declarer": ((), 0, NUM_PLAYERS - 1),
    "relative_current_leader": ((), 0, NUM_PLAYERS - 1),
    "declarer_points": ((), 0, 120),
    "defender_points": ((), 0, 120),
    "current_trick": ((), 0, NUM_TRICKS - 1),
    "void_info": ((NUM_PLAYERS, 5), 0, 1),
}


def observation_space():
    return spaces.Dict({
        key: spaces.Box(low, high, shape=shape, dtype=np.int8) if shape else spaces.Discrete(high + 1)
        for key, (shape, low, high) in OBSERVATION_SPECS.items()
    })


def empty_observations(batch_shape=()):
    return {key: np.full(tuple(batch_shape) + shape, low, dtype=np.int8)
            for key, (shape, low, _) in OBSERVATION_SPECS.items()}


def stack_observations(observations):
    observations = list(observations)
    if not observations:
        return empty_observations((0,))
    return {key: np.stack([obs[key] for obs in observations]) for key in OBSERVATION_SPECS}


def index_observations(observations, index):
    """Index leading batch axes identically for NumPy arrays or PyTorch tensors."""
    return {key: value[index] for key, value in observations.items()}


def contract_id(game_type):
    if game_type.kind == GameKind.SUIT:
        return Contract(int(game_type.trump_suit))
    return Contract.GRAND if game_type.kind == GameKind.GRAND else Contract.NULL


def effective_suit_table():
    """Static model lookup generated from the engine rules, not duplicated rules."""
    table = np.empty((len(Contract), NUM_CARDS), dtype=np.int64)
    for contract in Contract:
        game_type = (GameType(GameKind.SUIT, Suit(contract)) if contract < Contract.GRAND
                     else GameType(GameKind.GRAND if contract == Contract.GRAND else GameKind.NULL))
        for card in range(NUM_CARDS):
            suit = effective_suit(card, game_type)
            table[contract, card] = 4 if suit == "TRUMP" else int(suit)
    return table


def build_observation(state, acting_player, void_info=None) -> StructuredSkatObservation:
    """Read only own cards and public play/score facts from authoritative state.

    Scores include completed tricks only, never the hidden Skat. At termination
    current_trick remains 9 (the final trick); phase marks the terminal state.
    Optional cached void_info is in absolute engine coordinates and is copied.
    """
    if state is None:
        raise ValueError("Game must be reset before building an observation.")
    if acting_player not in range(NUM_PLAYERS):
        raise ValueError("acting_player must be 0, 1, or 2.")
    obs = empty_observations()
    obs["card_status"][list(state.hands[acting_player])] = CardStatus.OWN
    tricks = list(state.completed_tricks)
    if not state.current_trick.is_complete():
        tricks.append(state.current_trick)
    inferred_voids = np.zeros((NUM_PLAYERS, 5), dtype=np.int8)
    for trick_index, trick in enumerate(tricks):
        if not trick.cards:
            continue
        required = effective_suit(trick.lead_card(), state.game_type)
        for slot, (player, card) in enumerate(trick.cards):
            obs["card_status"][card] = CardStatus.PLAYED
            obs["played_by"][card] = (player - acting_player) % NUM_PLAYERS
            obs["trick_index"][card] = trick_index
            obs["trick_slot"][card] = slot
            if void_info is None and slot and effective_suit(card, state.game_type) != required:
                inferred_voids[player, 4 if required == "TRUMP" else int(required)] = 1
    scalars = {
        "phase": Phase.TERMINAL if state.terminated else Phase.CARD_PLAY,
        "contract": contract_id(state.game_type),
        "relative_declarer": (state.declarer - acting_player) % NUM_PLAYERS,
        "relative_current_leader": (state.current_trick.leader - acting_player) % NUM_PLAYERS,
        "declarer_points": points_won_by_player(state.won_cards, state.declarer),
        "defender_points": sum(points_won_by_player(state.won_cards, p)
                               for p in range(NUM_PLAYERS) if p != state.declarer),
        "current_trick": min(len(state.completed_tricks), NUM_TRICKS - 1),
    }
    obs.update({key: np.asarray(value, dtype=np.int8) for key, value in scalars.items()})
    absolute_voids = inferred_voids if void_info is None else void_info
    obs["void_info"] = np.asarray(absolute_voids, dtype=np.int8)[
        (np.arange(NUM_PLAYERS) + acting_player) % NUM_PLAYERS
    ].copy()
    return obs


def encode_belief_targets(state, player):
    """Privileged labels: LEFT=0, RIGHT=1, SKAT=2; never policy inputs."""
    targets = np.full(NUM_CARDS, -1, dtype=np.int64)
    if state is None or state.terminated:
        return targets
    targets[list(state.hands[(player + 1) % NUM_PLAYERS])] = 0
    targets[list(state.hands[(player + 2) % NUM_PLAYERS])] = 1
    targets[list(state.skat)] = 2
    return targets
