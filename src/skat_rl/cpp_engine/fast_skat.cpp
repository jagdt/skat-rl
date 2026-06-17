#include "fast_skat.h"

#include <algorithm>
#include <stdexcept>

namespace skat_rl {

namespace {

constexpr std::array<int, 8> kCardPoints = {
    0,  // seven
    0,  // eight
    0,  // nine
    3,  // queen
    4,  // king
    10, // ten
    11, // ace
    2,  // jack
};

constexpr std::array<int, 4> kJackTrumpOrder = {
    4, // clubs
    3, // spades
    2, // hearts
    1, // diamonds
};

constexpr std::array<int, 8> kSuitGameRankOrder = {
    1, // seven
    2, // eight
    3, // nine
    4, // queen
    5, // king
    6, // ten
    7, // ace
    0, // jack handled separately
};

constexpr std::array<int, 8> kNullRankOrder = {
    1, // seven
    2, // eight
    3, // nine
    6, // queen
    7, // king
    4, // ten
    8, // ace
    5, // jack
};

void validate_card(int card) {
    if (card < 0 || card >= kNumCards) {
        throw std::invalid_argument("Card must be in range 0..31.");
    }
}

void validate_player(int player) {
    if (player < 0 || player >= kNumPlayers) {
        throw std::invalid_argument("Player must be in range 0..2.");
    }
}

}  // namespace

int card_suit(int card) {
    validate_card(card);
    return card / 8;
}

int card_rank(int card) {
    validate_card(card);
    return card % 8;
}

int card_points(int card) {
    return kCardPoints[card_rank(card)];
}

bool is_trump(int card, int game_kind, int trump_suit) {
    const int rank = card_rank(card);
    const int suit = card_suit(card);

    if (game_kind == NULL_GAME) {
        return false;
    }
    if (rank == 7) {
        return true;
    }
    if (game_kind == GRAND) {
        return false;
    }
    if (game_kind == SUIT) {
        if (trump_suit < 0 || trump_suit > 3) {
            throw std::invalid_argument("Suit game requires trump_suit in range 0..3.");
        }
        return suit == trump_suit;
    }
    throw std::invalid_argument("Unsupported game kind.");
}

int effective_suit(int card, int game_kind, int trump_suit) {
    if (is_trump(card, game_kind, trump_suit)) {
        return kTrumpEffectiveSuit;
    }
    return card_suit(card);
}

int card_strength_in_trick(int card, int lead_card, int game_kind, int trump_suit) {
    if (game_kind == NULL_GAME) {
        if (card_suit(card) != card_suit(lead_card)) {
            return 0;
        }
        return kNullRankOrder[card_rank(card)];
    }

    const bool card_is_trump = is_trump(card, game_kind, trump_suit);
    const bool lead_is_trump = is_trump(lead_card, game_kind, trump_suit);

    if (card_is_trump) {
        const int rank = card_rank(card);
        if (rank == 7) {
            return 100 + kJackTrumpOrder[card_suit(card)];
        }
        return 50 + kSuitGameRankOrder[rank];
    }

    if (lead_is_trump) {
        return 0;
    }
    if (card_suit(card) != card_suit(lead_card)) {
        return 0;
    }
    return kSuitGameRankOrder[card_rank(card)];
}

uint32_t cards_to_mask(const std::vector<int>& cards) {
    uint32_t mask = 0;
    for (int card : cards) {
        validate_card(card);
        mask |= (uint32_t{1} << card);
    }
    return mask;
}

std::vector<int> mask_to_cards(uint32_t mask) {
    std::vector<int> cards;
    cards.reserve(kNumCards);
    for (int card = 0; card < kNumCards; ++card) {
        if (mask & (uint32_t{1} << card)) {
            cards.push_back(card);
        }
    }
    return cards;
}

FastSkatGame::FastSkatGame() {
    clear_state();
}

void FastSkatGame::clear_state() {
    hands_.fill(0);
    skat_.fill(-1);
    declarer_ = 0;
    game_kind_ = SUIT;
    trump_suit_ = 0;
    current_player_ = 0;
    trick_index_ = 0;
    trick_pos_ = 0;
    terminated_ = false;
    declarer_took_trick_ = false;
    won_points_.fill(0);
    for (auto& trick : history_cards_) {
        trick.fill(-1);
    }
    for (auto& trick : history_players_) {
        trick.fill(-1);
    }
    current_trick_cards_.fill(-1);
    current_trick_players_.fill(-1);
}

void FastSkatGame::reset(uint32_t seed) {
    clear_state();
    rng_.seed(seed);

    std::array<int, kNumCards> deck{};
    for (int card = 0; card < kNumCards; ++card) {
        deck[card] = card;
    }
    std::shuffle(deck.begin(), deck.end(), rng_);

    for (int player = 0; player < kNumPlayers; ++player) {
        for (int i = 0; i < kCardsPerHand; ++i) {
            hands_[player] |= (uint32_t{1} << deck[player * kCardsPerHand + i]);
        }
    }
    skat_[0] = deck[30];
    skat_[1] = deck[31];
    declarer_ = choose_declarer();
    game_kind_ = SUIT;
    trump_suit_ = choose_trump_suit(hands_[declarer_]);
    current_player_ = 0;
}

void FastSkatGame::reset_fixed_declarer(uint32_t seed, int fixed_declarer) {
    validate_player(fixed_declarer);
    clear_state();
    rng_.seed(seed);

    while (true) {
        hands_.fill(0);
        std::array<int, kNumCards> deck{};
        for (int card = 0; card < kNumCards; ++card) {
            deck[card] = card;
        }
        std::shuffle(deck.begin(), deck.end(), rng_);

        for (int player = 0; player < kNumPlayers; ++player) {
            for (int i = 0; i < kCardsPerHand; ++i) {
                hands_[player] |= (uint32_t{1} << deck[player * kCardsPerHand + i]);
            }
        }

        if (choose_declarer() == fixed_declarer) {
            skat_[0] = deck[30];
            skat_[1] = deck[31];
            declarer_ = fixed_declarer;
            game_kind_ = SUIT;
            trump_suit_ = choose_trump_suit(hands_[declarer_]);
            current_player_ = 0;
            return;
        }
    }
}

void FastSkatGame::reset_from_deal(
    const std::vector<std::vector<int>>& hands,
    const std::vector<int>& skat,
    int declarer,
    int game_kind,
    int trump_suit,
    int current_player
) {
    if (hands.size() != kNumPlayers) {
        throw std::invalid_argument("hands must contain exactly three hands.");
    }
    if (skat.size() != 2) {
        throw std::invalid_argument("skat must contain exactly two cards.");
    }
    validate_player(declarer);
    validate_player(current_player);

    clear_state();
    uint32_t seen = 0;
    for (int player = 0; player < kNumPlayers; ++player) {
        if (hands[player].size() != kCardsPerHand) {
            throw std::invalid_argument("Each hand must contain exactly ten cards.");
        }
        hands_[player] = cards_to_mask(hands[player]);
        if (seen & hands_[player]) {
            throw std::invalid_argument("Duplicate card in hands.");
        }
        seen |= hands_[player];
    }
    for (int i = 0; i < 2; ++i) {
        validate_card(skat[i]);
        if (seen & (uint32_t{1} << skat[i])) {
            throw std::invalid_argument("Duplicate card in skat.");
        }
        seen |= (uint32_t{1} << skat[i]);
        skat_[i] = skat[i];
    }

    declarer_ = declarer;
    game_kind_ = game_kind;
    trump_suit_ = trump_suit;
    current_player_ = current_player;
}

uint32_t FastSkatGame::legal_mask_bits() const {
    if (terminated_) {
        return 0;
    }

    const uint32_t hand_mask = hands_[current_player_];
    if (trick_pos_ == 0) {
        return hand_mask;
    }

    const int required_suit = effective_suit(current_trick_cards_[0], game_kind_, trump_suit_);
    uint32_t matching = 0;
    for (int card = 0; card < kNumCards; ++card) {
        const uint32_t bit = (uint32_t{1} << card);
        if ((hand_mask & bit) && effective_suit(card, game_kind_, trump_suit_) == required_suit) {
            matching |= bit;
        }
    }
    return matching != 0 ? matching : hand_mask;
}

std::vector<int> FastSkatGame::legal_actions() const {
    return mask_to_cards(legal_mask_bits());
}

std::vector<bool> FastSkatGame::legal_mask_array() const {
    const uint32_t mask = legal_mask_bits();
    std::vector<bool> values(kNumCards, false);
    for (int card = 0; card < kNumCards; ++card) {
        values[card] = static_cast<bool>(mask & (uint32_t{1} << card));
    }
    return values;
}

std::vector<float> FastSkatGame::observation(int player) const {
    validate_player(player);
    std::vector<float> obs;
    obs.reserve(
        kNumCards
        + kMaxTricks * kTrickSize * kNumCards
        + kMaxTricks * kTrickSize * kNumPlayers
        + kNumCards
        + kNumPlayers
        + kNumPlayers
        + kNumPlayers
        + 3
        + 4
        + 1
        + 1
        + 2
        + kNumPlayers * 5
    );

    const auto append_one_hot = [&obs](int size, int index) {
        for (int i = 0; i < size; ++i) {
            obs.push_back(i == index ? 1.0F : 0.0F);
        }
    };

    const auto append_card_one_hot = [&obs](int card) {
        for (int i = 0; i < kNumCards; ++i) {
            obs.push_back(i == card ? 1.0F : 0.0F);
        }
    };

    for (int card = 0; card < kNumCards; ++card) {
        obs.push_back((hands_[player] & (uint32_t{1} << card)) ? 1.0F : 0.0F);
    }

    for (int trick = 0; trick < kMaxTricks; ++trick) {
        for (int slot = 0; slot < kTrickSize; ++slot) {
            int card = -1;
            if (trick < trick_index_) {
                card = history_cards_[trick][slot];
            } else if (trick == trick_index_ && slot < trick_pos_) {
                card = current_trick_cards_[slot];
            }
            append_card_one_hot(card);
        }
    }

    for (int trick = 0; trick < kMaxTricks; ++trick) {
        for (int slot = 0; slot < kTrickSize; ++slot) {
            int card_player = -1;
            if (trick < trick_index_) {
                card_player = history_players_[trick][slot];
            } else if (trick == trick_index_ && slot < trick_pos_) {
                card_player = current_trick_players_[slot];
            }
            append_one_hot(kNumPlayers, card_player);
        }
    }

    uint32_t current_trick_mask = 0;
    for (int slot = 0; slot < trick_pos_; ++slot) {
        current_trick_mask |= (uint32_t{1} << current_trick_cards_[slot]);
    }
    for (int card = 0; card < kNumCards; ++card) {
        obs.push_back((current_trick_mask & (uint32_t{1} << card)) ? 1.0F : 0.0F);
    }

    append_one_hot(kNumPlayers, current_player_);
    append_one_hot(kNumPlayers, trick_pos_ > 0 ? current_trick_players_[0] : current_player_);
    append_one_hot(kNumPlayers, declarer_);
    append_one_hot(3, game_kind_);

    for (int suit = 0; suit < 4; ++suit) {
        obs.push_back(game_kind_ == SUIT && trump_suit_ == suit ? 1.0F : 0.0F);
    }

    obs.push_back(static_cast<float>(trick_index_) / 10.0F);
    obs.push_back(static_cast<float>(trick_pos_) / 3.0F);
    obs.push_back(static_cast<float>(declarer_points()) / 120.0F);
    obs.push_back(static_cast<float>(defender_points()) / 120.0F);

    std::array<std::array<float, 5>, kNumPlayers> void_info{};
    const auto mark_voids = [this, &void_info](const std::array<int, kTrickSize>& cards,
                                               const std::array<int, kTrickSize>& players,
                                               int size) {
        if (size < 2) {
            return;
        }
        const int required_suit = effective_suit(cards[0], game_kind_, trump_suit_);
        for (int slot = 1; slot < size; ++slot) {
            if (effective_suit(cards[slot], game_kind_, trump_suit_) != required_suit) {
                void_info[players[slot]][required_suit] = 1.0F;
            }
        }
    };

    for (int trick = 0; trick < trick_index_; ++trick) {
        mark_voids(history_cards_[trick], history_players_[trick], kTrickSize);
    }
    mark_voids(current_trick_cards_, current_trick_players_, trick_pos_);

    for (int p = 0; p < kNumPlayers; ++p) {
        for (int suit = 0; suit < 5; ++suit) {
            obs.push_back(void_info[p][suit]);
        }
    }

    return obs;
}

StepInfo FastSkatGame::step(int action) {
    validate_card(action);
    if (terminated_) {
        throw std::runtime_error("Cannot step terminated game. Call reset().");
    }

    const uint32_t action_bit = (uint32_t{1} << action);
    if ((legal_mask_bits() & action_bit) == 0) {
        throw std::invalid_argument("Illegal action.");
    }

    const int player = current_player_;
    hands_[player] &= ~action_bit;
    current_trick_cards_[trick_pos_] = action;
    current_trick_players_[trick_pos_] = player;
    ++trick_pos_;

    if (trick_pos_ == kTrickSize) {
        const int winner = current_trick_winner();
        const int points = current_trick_points();
        won_points_[winner] += points;
        if (winner == declarer_) {
            declarer_took_trick_ = true;
        }

        for (int i = 0; i < kTrickSize; ++i) {
            history_cards_[trick_index_][i] = current_trick_cards_[i];
            history_players_[trick_index_][i] = current_trick_players_[i];
        }

        ++trick_index_;
        if (trick_index_ == kMaxTricks) {
            terminated_ = true;
        } else {
            current_player_ = winner;
            trick_pos_ = 0;
            current_trick_cards_.fill(-1);
            current_trick_players_.fill(-1);
        }
    } else {
        current_player_ = (player + 1) % kNumPlayers;
    }

    StepInfo info;
    info.terminated = terminated_;
    info.current_player = current_player_;
    info.trick_index = trick_index_;
    if (terminated_ && game_kind_ == NULL_GAME) {
        info.declarer_points = 0;
        info.defender_points = 0;
    } else if (terminated_) {
        info.declarer_points = declarer_points();
        info.defender_points = 120 - info.declarer_points;
    } else {
        info.declarer_points = declarer_points();
        info.defender_points = defender_points();
    }
    if (terminated_) {
        info.declarer_won = declarer_won();
    }
    return info;
}

bool FastSkatGame::is_terminal() const { return terminated_; }
int FastSkatGame::current_player() const { return current_player_; }
int FastSkatGame::declarer() const { return declarer_; }
int FastSkatGame::trick_index() const { return trick_index_; }
int FastSkatGame::game_kind() const { return game_kind_; }
int FastSkatGame::trump_suit() const { return trump_suit_; }
int FastSkatGame::trick_position() const { return trick_pos_; }

int FastSkatGame::declarer_points() const {
    return won_points_[declarer_];
}

int FastSkatGame::defender_points() const {
    int points = 0;
    for (int player = 0; player < kNumPlayers; ++player) {
        if (player != declarer_) {
            points += won_points_[player];
        }
    }
    return points;
}

std::vector<int> FastSkatGame::hand(int player) const {
    validate_player(player);
    return mask_to_cards(hands_[player]);
}

std::vector<int> FastSkatGame::skat() const {
    return {skat_[0], skat_[1]};
}

std::vector<std::vector<int>> FastSkatGame::history_cards() const {
    std::vector<std::vector<int>> history;
    history.reserve(trick_index_);
    for (int trick = 0; trick < trick_index_; ++trick) {
        history.push_back({
            history_cards_[trick][0],
            history_cards_[trick][1],
            history_cards_[trick][2],
        });
    }
    return history;
}

std::vector<std::vector<int>> FastSkatGame::history_players() const {
    std::vector<std::vector<int>> history;
    history.reserve(trick_index_);
    for (int trick = 0; trick < trick_index_; ++trick) {
        history.push_back({
            history_players_[trick][0],
            history_players_[trick][1],
            history_players_[trick][2],
        });
    }
    return history;
}

std::vector<int> FastSkatGame::current_trick_cards() const {
    std::vector<int> cards;
    cards.reserve(trick_pos_);
    for (int i = 0; i < trick_pos_; ++i) {
        cards.push_back(current_trick_cards_[i]);
    }
    return cards;
}

std::vector<int> FastSkatGame::current_trick_players() const {
    std::vector<int> players;
    players.reserve(trick_pos_);
    for (int i = 0; i < trick_pos_; ++i) {
        players.push_back(current_trick_players_[i]);
    }
    return players;
}

int FastSkatGame::choose_declarer() const {
    double best_score = -1.0;
    int best_player = 0;
    for (int player = 0; player < kNumPlayers; ++player) {
        double score = 0.0;
        for (int card : mask_to_cards(hands_[player])) {
            const int rank = card_rank(card);
            if (rank == 7) {
                score += 1.0;
            } else if (rank == 6) {
                score += 1.0;
            } else if (rank == 5) {
                score += 0.4;
            }
        }
        if (score > best_score) {
            best_score = score;
            best_player = player;
        }
    }
    return best_player;
}

int FastSkatGame::choose_trump_suit(uint32_t hand_mask) const {
    std::array<int, 4> suit_counts{};
    std::array<bool, 4> suit_has_ten{};

    for (int card : mask_to_cards(hand_mask)) {
        const int suit = card_suit(card);
        ++suit_counts[suit];
        if (card_rank(card) == 5) {
            suit_has_ten[suit] = true;
        }
    }

    int best_suit = 0;
    for (int suit = 1; suit < 4; ++suit) {
        if (
            suit_counts[suit] > suit_counts[best_suit]
            || (
                suit_counts[suit] == suit_counts[best_suit]
                && suit_has_ten[suit] > suit_has_ten[best_suit]
            )
        ) {
            best_suit = suit;
        }
    }
    return best_suit;
}

int FastSkatGame::current_trick_winner() const {
    const int lead_card = current_trick_cards_[0];
    int best_player = current_trick_players_[0];
    int best_strength = card_strength_in_trick(lead_card, lead_card, game_kind_, trump_suit_);

    for (int i = 1; i < kTrickSize; ++i) {
        const int strength = card_strength_in_trick(
            current_trick_cards_[i],
            lead_card,
            game_kind_,
            trump_suit_
        );
        if (strength > best_strength) {
            best_player = current_trick_players_[i];
            best_strength = strength;
        }
    }
    return best_player;
}

int FastSkatGame::current_trick_points() const {
    int points = 0;
    for (int i = 0; i < kTrickSize; ++i) {
        points += card_points(current_trick_cards_[i]);
    }
    return points;
}

bool FastSkatGame::declarer_won() const {
    if (game_kind_ == NULL_GAME) {
        return !declarer_took_trick_;
    }
    return declarer_points() > 60;
}

}  // namespace skat_rl
