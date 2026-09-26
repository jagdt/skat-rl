#include "fast_skat.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

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

void seed_rng(std::mt19937& rng, uint64_t seed) {
    std::seed_seq seed_sequence{
        static_cast<uint32_t>(seed),
        static_cast<uint32_t>(seed >> 32),
    };
    rng.seed(seed_sequence);
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
    hand_game_ = false;
    declarer_tricks_ = 0;
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

void FastSkatGame::reset(uint64_t seed) {
    clear_state();
    hand_game_ = true;
    seed_rng(rng_, seed);

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

void FastSkatGame::reset_fixed_declarer(uint64_t seed, int fixed_declarer) {
    validate_player(fixed_declarer);
    clear_state();
    hand_game_ = true;
    seed_rng(rng_, seed);

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
    int current_player,
    bool hand_game
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
    hand_game_ = hand_game;
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

std::vector<int> FastSkatGame::belief_targets(int player) const {
    validate_player(player);
    const int next_opponent = (player + 1) % kNumPlayers;
    const int previous_opponent = (player + 2) % kNumPlayers;
    std::vector<int> targets(kNumCards, -1);

    for (int card = 0; card < kNumCards; ++card) {
        const uint32_t bit = uint32_t{1} << card;
        if (hands_[next_opponent] & bit) {
            targets[card] = 0;
        } else if (hands_[previous_opponent] & bit) {
            targets[card] = 1;
        } else if (card == skat_[0] || card == skat_[1]) {
            targets[card] = 2;
        }
    }
    return targets;
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
            ++declarer_tricks_;
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
        // Include the hidden Skat only in the final score, not in observations.
        info.declarer_points = declarer_points() + card_points(skat_[0]) + card_points(skat_[1]);
        info.defender_points = 120 - info.declarer_points;
    } else {
        info.declarer_points = declarer_points();
        info.defender_points = defender_points();
    }
    if (terminated_) {
        info.declarer_won = declarer_won();
        info.game_value = final_game_value();
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
    return declarer_points() + card_points(skat_[0]) + card_points(skat_[1]) > 60;
}

int FastSkatGame::final_game_value() const {
    if (game_kind_ == NULL_GAME) {
        return hand_game_ ? 35 : 23;
    }
    uint32_t declarer_cards = (uint32_t{1} << skat_[0]) | (uint32_t{1} << skat_[1]);
    for (int trick = 0; trick < kMaxTricks; ++trick) {
        for (int slot = 0; slot < kTrickSize; ++slot) {
            if (history_players_[trick][slot] == declarer_) {
                declarer_cards |= uint32_t{1} << history_cards_[trick][slot];
            }
        }
    }
    std::vector<int> trumps{7, 15, 23, 31};
    if (game_kind_ == SUIT) {
        for (int rank : {6, 5, 4, 3, 2, 1, 0}) {
            trumps.push_back(trump_suit_ * 8 + rank);
        }
    }
    const bool with_top = (declarer_cards & (uint32_t{1} << trumps[0])) != 0;
    int matadors = 0;
    for (int card : trumps) {
        if (((declarer_cards & (uint32_t{1} << card)) != 0) != with_top) {
            break;
        }
        ++matadors;
    }
    const int points = declarer_points() + card_points(skat_[0]) + card_points(skat_[1]);
    const bool schneider = points <= 30 || points >= 90;
    const bool schwarz = declarer_tricks_ == 0 || declarer_tricks_ == kMaxTricks;
    const int base = game_kind_ == GRAND ? 24 : 12 - trump_suit_;
    return base * (matadors + 1 + int(hand_game_) + int(schneider) + int(schwarz));
}

BatchedFastSkatEnv::BatchedFastSkatEnv(int size, int learning_player, int fixed_declarer,
                                     bool autoplay_opponents)
    : games_(size),
      active_(size, 0),
      episode_returns_(size, 0.0F),
      episode_lengths_(size, 0),
      learning_player_(learning_player),
      fixed_declarer_(fixed_declarer),
      autoplay_opponents_(autoplay_opponents) {
    if (size < 1) {
        throw std::invalid_argument("BatchedFastSkatEnv size must be at least 1.");
    }
    validate_player(learning_player_);
    if (fixed_declarer_ != -1) {
        validate_player(fixed_declarer_);
    }
}

void BatchedFastSkatEnv::reset(uint64_t seed) {
    std::vector<uint64_t> seeds;
    seeds.reserve(size());
    for (int index = 0; index < size(); ++index) {
        seeds.push_back(seed + static_cast<uint64_t>(index));
    }
    reset_many(seeds);
}

void BatchedFastSkatEnv::reset_many(const std::vector<uint64_t>& seeds) {
    if (static_cast<int>(seeds.size()) != size()) {
        throw std::invalid_argument("seeds length must equal batched environment size.");
    }

    for (int index = 0; index < size(); ++index) {
        if (fixed_declarer_ == -1) {
            games_[index].reset(seeds[index]);
        } else {
            games_[index].reset_fixed_declarer(seeds[index], fixed_declarer_);
        }
        active_[index] = 1;
        episode_returns_[index] = 0.0F;
        episode_lengths_[index] = 0;
        if (autoplay_opponents_) {
            play_until_learning_player(games_[index]);
        }
        if (games_[index].is_terminal()) {
            active_[index] = 0;
        }
    }
}

BatchedStepInfo BatchedFastSkatEnv::step(const std::vector<int>& actions) {
    const std::vector<int> indices = active_indices();
    if (actions.size() != indices.size()) {
        throw std::invalid_argument("actions length must equal active environment count.");
    }

    // Validate the entire batch before mutating any game.
    for (std::size_t batch_index = 0; batch_index < indices.size(); ++batch_index) {
        const FastSkatGame& game = games_[indices[batch_index]];
        const int action = actions[batch_index];
        if (action < 0 || action >= kNumCards
            || (game.legal_mask_bits() & (uint32_t{1} << action)) == 0) {
            throw std::invalid_argument("Batch contains an illegal action.");
        }
    }

    BatchedStepInfo result;
    result.env_indices = indices;
    result.rewards.reserve(indices.size());
    result.terminated.reserve(indices.size());

    for (std::size_t batch_index = 0; batch_index < indices.size(); ++batch_index) {
        const int env_index = indices[batch_index];
        FastSkatGame& game = games_[env_index];

        if (game.is_terminal()) {
            throw std::runtime_error("Cannot step a terminated active environment.");
        }
        if (autoplay_opponents_ && game.current_player() != learning_player_) {
            throw std::runtime_error("Active environment is not at the learning player's turn.");
        }

        const bool learner_acted = game.current_player() == learning_player_;
        StepInfo step_info = game.step(actions[batch_index]);
        float reward = reward_for_step(game, step_info);
        if (autoplay_opponents_ && !game.is_terminal()) {
            reward += play_until_learning_player(game);
        }

        episode_returns_[env_index] += reward;
        episode_lengths_[env_index] += learner_acted ? 1 : 0;

        const bool done = game.is_terminal();
        result.rewards.push_back(reward);
        result.terminated.push_back(done ? 1 : 0);

        if (done) {
            active_[env_index] = 0;
            result.completed_env_indices.push_back(env_index);
            result.completed_returns.push_back(episode_returns_[env_index]);
            result.completed_lengths.push_back(episode_lengths_[env_index]);
        }
    }

    return result;
}

std::vector<int> BatchedFastSkatEnv::active_indices() const {
    std::vector<int> indices;
    indices.reserve(games_.size());
    for (int index = 0; index < size(); ++index) {
        if (active_[index] != 0) {
            indices.push_back(index);
        }
    }
    return indices;
}

std::vector<float> BatchedFastSkatEnv::active_observations() const {
    const std::vector<int> indices = active_indices();
    std::vector<float> observations;
    observations.reserve(indices.size() * kObservationDim);
    for (int env_index : indices) {
        const FastSkatGame& game = games_[env_index];
        std::vector<float> observation = game.observation(game.current_player());
        observations.insert(observations.end(), observation.begin(), observation.end());
    }
    return observations;
}

std::vector<int> BatchedFastSkatEnv::active_players() const {
    std::vector<int> players;
    for (int env_index : active_indices()) {
        players.push_back(games_[env_index].current_player());
    }
    return players;
}

std::vector<int> BatchedFastSkatEnv::active_declarers() const {
    std::vector<int> declarers;
    for (int env_index : active_indices()) {
        declarers.push_back(games_[env_index].declarer());
    }
    return declarers;
}

std::vector<uint8_t> BatchedFastSkatEnv::active_action_masks() const {
    const std::vector<int> indices = active_indices();
    std::vector<uint8_t> masks;
    masks.reserve(indices.size() * kNumCards);
    for (int env_index : indices) {
        const uint32_t mask_bits = games_[env_index].legal_mask_bits();
        for (int card = 0; card < kNumCards; ++card) {
            masks.push_back((mask_bits & (uint32_t{1} << card)) != 0 ? 1 : 0);
        }
    }
    return masks;
}

std::vector<int> BatchedFastSkatEnv::active_belief_targets() const {
    const std::vector<int> indices = active_indices();
    std::vector<int> targets;
    targets.reserve(indices.size() * kNumCards);
    for (int env_index : indices) {
        const FastSkatGame& game = games_[env_index];
        std::vector<int> game_targets = game.belief_targets(game.current_player());
        targets.insert(targets.end(), game_targets.begin(), game_targets.end());
    }
    return targets;
}

int BatchedFastSkatEnv::active_count() const {
    return static_cast<int>(active_indices().size());
}

int BatchedFastSkatEnv::size() const {
    return static_cast<int>(games_.size());
}

int BatchedFastSkatEnv::learning_player() const {
    return learning_player_;
}

int BatchedFastSkatEnv::observation_dim() const {
    return kObservationDim;
}

int BatchedFastSkatEnv::action_dim() const {
    return kNumCards;
}

float BatchedFastSkatEnv::play_until_learning_player(FastSkatGame& game) {
    float total_reward = 0.0F;
    while (!game.is_terminal() && game.current_player() != learning_player_) {
        const int action = choose_opponent_action(game);
        const StepInfo step_info = game.step(action);
        total_reward += reward_for_step(game, step_info);
    }
    return total_reward;
}

float BatchedFastSkatEnv::reward_for_step(const FastSkatGame& game, const StepInfo& info) const {
    if (!info.terminated) {
        return 0.0F;
    }

    const int declarer = game.declarer();
    const bool declarer_won = info.declarer_won.has_value() && info.declarer_won.value();
    if (learning_player_ == declarer) {
        return static_cast<float>(declarer_won ? info.game_value + 50 : -2 * info.game_value - 50) / 100.0F;
    }
    return declarer_won ? 0.0F : 0.4F;
}

int BatchedFastSkatEnv::choose_opponent_action(const FastSkatGame& game) const {
    const std::vector<int> legal = game.legal_actions();
    if (legal.empty()) {
        throw std::runtime_error("No legal opponent actions available.");
    }

    if (game.trick_position() == 0) {
        return choose_leading_action(game, legal);
    }
    return choose_following_action(game, legal);
}

int BatchedFastSkatEnv::choose_leading_action(const FastSkatGame& game, const std::vector<int>& legal) const {
    const int player = game.current_player();
    const int game_kind = game.game_kind();
    const int trump_suit = game.trump_suit();

    if (player == game.declarer()) {
        std::vector<int> trumps;
        for (int card : legal) {
            if (is_trump(card, game_kind, trump_suit)) {
                trumps.push_back(card);
            }
        }
        if (!trumps.empty()) {
            return *std::max_element(trumps.begin(), trumps.end(), [&](int lhs, int rhs) {
                return trump_strength(lhs, game_kind, trump_suit) < trump_strength(rhs, game_kind, trump_suit);
            });
        }
    }

    std::vector<int> non_trumps;
    for (int card : legal) {
        if (!is_trump(card, game_kind, trump_suit)) {
            non_trumps.push_back(card);
        }
    }
    if (!non_trumps.empty()) {
        return lowest_discard(non_trumps, game_kind, trump_suit);
    }
    return lowest_discard(legal, game_kind, trump_suit);
}

int BatchedFastSkatEnv::choose_following_action(const FastSkatGame& game, const std::vector<int>& legal) const {
    const int player = game.current_player();
    const int game_kind = game.game_kind();
    const int trump_suit = game.trump_suit();
    const std::vector<int> winners = winning_cards(game, legal);

    if (player == game.declarer()) {
        std::vector<int> non_trump_winners;
        for (int card : winners) {
            if (!is_trump(card, game_kind, trump_suit)) {
                non_trump_winners.push_back(card);
            }
        }
        if (!non_trump_winners.empty()) {
            return lowest_winner(non_trump_winners, game_kind, trump_suit);
        }
        if (!winners.empty() && trick_value(game) >= 9) {
            return lowest_winner(winners, game_kind, trump_suit);
        }
        return lowest_discard(legal, game_kind, trump_suit);
    }

    if (current_winning_player(game) != game.declarer()) {
        return highest_discard(legal, game_kind, trump_suit);
    }
    if (!winners.empty()) {
        return lowest_winner(winners, game_kind, trump_suit);
    }
    return lowest_discard(legal, game_kind, trump_suit);
}

std::vector<int> BatchedFastSkatEnv::winning_cards(const FastSkatGame& game, const std::vector<int>& legal) const {
    std::vector<int> winners;
    const std::vector<int> cards = game.current_trick_cards();
    if (cards.empty()) {
        return winners;
    }

    const int lead_card = cards[0];
    int best_strength = std::numeric_limits<int>::min();
    for (int card : cards) {
        best_strength = std::max(
            best_strength,
            card_strength_in_trick(card, lead_card, game.game_kind(), game.trump_suit())
        );
    }

    for (int card : legal) {
        const int strength = card_strength_in_trick(card, lead_card, game.game_kind(), game.trump_suit());
        if (strength > best_strength) {
            winners.push_back(card);
        }
    }
    return winners;
}

int BatchedFastSkatEnv::current_winning_player(const FastSkatGame& game) const {
    const std::vector<int> cards = game.current_trick_cards();
    const std::vector<int> players = game.current_trick_players();
    if (cards.empty()) {
        return game.current_player();
    }

    const int lead_card = cards[0];
    int best_player = players[0];
    int best_strength = card_strength_in_trick(cards[0], lead_card, game.game_kind(), game.trump_suit());

    for (std::size_t index = 1; index < cards.size(); ++index) {
        const int strength = card_strength_in_trick(cards[index], lead_card, game.game_kind(), game.trump_suit());
        if (strength > best_strength) {
            best_strength = strength;
            best_player = players[index];
        }
    }
    return best_player;
}

int BatchedFastSkatEnv::trick_value(const FastSkatGame& game) const {
    int value = 0;
    for (int card : game.current_trick_cards()) {
        value += card_points(card);
    }
    return value;
}

int BatchedFastSkatEnv::lowest_discard(const std::vector<int>& cards, int game_kind, int trump_suit) const {
    return *std::min_element(cards.begin(), cards.end(), [&](int lhs, int rhs) {
        return std::make_pair(card_points(lhs), trump_strength(lhs, game_kind, trump_suit))
            < std::make_pair(card_points(rhs), trump_strength(rhs, game_kind, trump_suit));
    });
}

int BatchedFastSkatEnv::highest_discard(const std::vector<int>& cards, int game_kind, int trump_suit) const {
    return *std::max_element(cards.begin(), cards.end(), [&](int lhs, int rhs) {
        return std::make_pair(card_points(lhs), trump_strength(lhs, game_kind, trump_suit))
            < std::make_pair(card_points(rhs), trump_strength(rhs, game_kind, trump_suit));
    });
}

int BatchedFastSkatEnv::lowest_winner(const std::vector<int>& cards, int game_kind, int trump_suit) const {
    return lowest_discard(cards, game_kind, trump_suit);
}

int BatchedFastSkatEnv::trump_strength(int card, int game_kind, int trump_suit) const {
    if (!is_trump(card, game_kind, trump_suit)) {
        return 0;
    }

    const int rank = card_rank(card);
    if (rank == 7) {
        return 100 - card_suit(card);
    }

    switch (rank) {
        case 6: return 70; // ace
        case 5: return 60; // ten
        case 4: return 50; // king
        case 3: return 40; // queen
        case 2: return 30; // nine
        case 1: return 20; // eight
        case 0: return 10; // seven
        default: return 0;
    }
}

}  // namespace skat_rl
