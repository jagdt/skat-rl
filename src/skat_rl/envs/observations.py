"""The observation layout shared by online play and recorded-game replay."""

import numpy as np

from skat_rl.engine.rules import effective_suit, points_won_by_player
from skat_rl.engine.state import GameKind


OBSERVATION_DIM = 1149


def encode_observation(state, player, void_info=None):
    observation = np.zeros(OBSERVATION_DIM, dtype=np.float32)
    if state is None:
        return observation

    observation[list(state.hands[player])] = 1.0
    history_cards = observation[32:992].reshape(10, 3, 32)
    history_players = observation[992:1082].reshape(10, 3, 3)
    tricks = list(state.completed_tricks)
    if not state.current_trick.is_complete():
        tricks.append(state.current_trick)
    for trick_index, trick in enumerate(tricks[:10]):
        for slot, (card_player, card) in enumerate(trick.cards):
            history_cards[trick_index, slot, card] = 1.0
            history_players[trick_index, slot, card_player] = 1.0

    for _, card in state.current_trick.cards:
        observation[1082 + card] = 1.0
    observation[1114 + state.current_player] = 1.0
    observation[1117 + state.current_trick.leader] = 1.0
    observation[1120 + state.declarer] = 1.0
    kind = {GameKind.SUIT: 0, GameKind.GRAND: 1, GameKind.NULL: 2}[state.game_type.kind]
    observation[1123 + kind] = 1.0
    if state.game_type.kind == GameKind.SUIT:
        observation[1126 + int(state.game_type.trump_suit)] = 1.0
    observation[1130] = len(state.completed_tricks) / 10.0
    observation[1131] = len(state.current_trick.cards) / 3.0
    observation[1132] = points_won_by_player(state.won_cards, state.declarer) / 120.0
    observation[1133] = sum(
        points_won_by_player(state.won_cards, p) for p in range(3) if p != state.declarer
    ) / 120.0

    if void_info is None:
        void_info = np.zeros((3, 5), dtype=np.float32)
        for trick in tricks:
            if not trick.cards:
                continue
            required = effective_suit(trick.lead_card(), state.game_type)
            for card_player, card in trick.cards[1:]:
                if effective_suit(card, state.game_type) != required:
                    void_info[card_player, 4 if required == "TRUMP" else int(required)] = 1.0
    observation[1134:] = void_info.reshape(-1)
    return observation


def encode_belief_targets(state, player):
    targets = np.full(32, -1, dtype=np.int64)
    if state is None or state.terminated:
        return targets
    targets[list(state.hands[(player + 1) % 3])] = 0
    targets[list(state.hands[(player + 2) % 3])] = 1
    targets[list(state.skat)] = 2
    return targets
