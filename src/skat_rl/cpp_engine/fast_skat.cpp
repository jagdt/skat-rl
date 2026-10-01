#include "fast_skat.h"

#include <algorithm>
#include <stdexcept>
#include <set>

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

const std::vector<int>& bid_values() {
    static const std::vector<int> values = [] {
        std::set<int> bids{23, 35, 46, 59};
        for (int base : {9, 10, 11, 12}) {
            for (int multiplier = 2; multiplier <= 18; ++multiplier) bids.insert(base * multiplier);
        }
        for (int multiplier = 2; multiplier <= 11; ++multiplier) bids.insert(24 * multiplier);
        return std::vector<int>(bids.begin(), bids.end());
    }();
    return values;
}

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
    phase_ = CARD_PLAY;
    forehand_ = 0;
    winning_bid_ = 0;
    bid_index_ = 0;
    auction_caller_ = 1;
    auction_holder_ = 0;
    auction_role_ = CALLER;
    rearhand_entered_ = false;
    forehand_offer_ = false;
    bid_status_.fill(NOT_ENTERED);
    highest_called_.fill(-1);
    highest_held_.fill(-1);
    pass_threshold_.fill(-1);
    pass_role_.fill(-1);
    trick_index_ = 0;
    trick_pos_ = 0;
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

void FastSkatGame::deal(uint64_t seed) {
    clear_state();
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
}

void FastSkatGame::reset(uint64_t seed) {
    deal(seed);
    hand_game_ = true;
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

void FastSkatGame::begin_auction(int forehand) {
    validate_player(forehand);
    phase_ = BIDDING;
    forehand_ = forehand;
    declarer_ = -1;
    game_kind_ = -1;
    trump_suit_ = -1;
    hand_game_ = false;
    auction_holder_ = forehand;
    auction_caller_ = (forehand + 1) % kNumPlayers;
    current_player_ = auction_caller_;
    bid_status_[auction_holder_] = bid_status_[auction_caller_] = ACTIVE;
}

void FastSkatGame::reset_full(uint64_t seed) {
    deal(seed);
    begin_auction(std::uniform_int_distribution<int>(0, 2)(rng_));
}

void FastSkatGame::reset_full_from_deal(const std::vector<std::vector<int>>& hands,
                                     const std::vector<int>& skat, int forehand) {
    reset_from_deal(hands, skat, 0, SUIT, 0, forehand);
    begin_auction(forehand);
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
        if (mask_to_cards(hands_[player]).size() != kCardsPerHand) {
            throw std::invalid_argument("Duplicate card within a hand.");
        }
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
    if (phase_ != CARD_PLAY) {
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
    if (phase_ == CARD_PLAY) return mask_to_cards(legal_mask_bits());
    std::vector<int> actions;
    for (int action = 0; action < kNumActions; ++action) {
        if (is_legal_action(action)) actions.push_back(action);
    }
    return actions;
}

bool FastSkatGame::is_legal_action(int action) const {
    if (is_terminal() || action < 0 || action >= kNumActions) return false;
    switch (phase_) {
        case BIDDING: case PICKUP_DECISION: return action < 2;
        case DISCARD: return true;
        case CONTRACT_SELECTION:
            return action < 5 || (action == NULL_CONTRACT && winning_bid_ <= (hand_game_ ? 35 : 23));
        case CARD_PLAY:
            return action < kNumCards && (legal_mask_bits() & (uint32_t{1} << action)) != 0;
        default: return false;
    }
}

std::vector<bool> FastSkatGame::legal_mask_array() const {
    std::vector<bool> values(kNumActions, false);
    for (int action : legal_actions()) values[action] = true;
    return values;
}

StructuredObservation FastSkatGame::build_observation(int player) const {
    validate_player(player);
    StructuredObservation obs;
    obs.phase = phase_;
    if (game_kind_ >= 0) {
        obs.contract = game_kind_ == SUIT ? trump_suit_ :
                       (game_kind_ == GRAND ? GRAND_CONTRACT : NULL_CONTRACT);
    }
    if (declarer_ >= 0) obs.relative_declarer = (declarer_ - player + kNumPlayers) % kNumPlayers;
    const int leader = trick_pos_ > 0 ? current_trick_players_[0] : current_player_;
    if (game_kind_ >= 0) obs.relative_current_leader = (leader - player + kNumPlayers) % kNumPlayers;
    obs.current_trick = std::max(0, trick_index_ - int(is_terminal()));
    obs.seat = (player - forehand_ + kNumPlayers) % kNumPlayers;
    obs.auction_role = phase_ == BIDDING ? auction_role_ : -1;
    obs.decision_threshold = decision_threshold();
    obs.hand_game = (phase_ == BIDDING || phase_ == PICKUP_DECISION || declarer_ < 0) ? -1 : int(hand_game_);
    for (int relative = 0; relative < kNumPlayers; ++relative) {
        const int absolute = (player + relative) % kNumPlayers;
        obs.bid_status[relative] = bid_status_[absolute];
        obs.highest_called[relative] = highest_called_[absolute];
        obs.highest_held[relative] = highest_held_[absolute];
        obs.pass_threshold[relative] = pass_threshold_[absolute];
        obs.pass_role[relative] = pass_role_[absolute];
    }
    // Only public trick points, never the hidden Skat.
    obs.declarer_points = declarer_points();
    obs.defender_points = defender_points();
    for (int card = 0; card < kNumCards; ++card) {
        if (hands_[player] & (uint32_t{1} << card)) {
            obs.card_status[card] = OWN;
        }
    }
    if (obs.hand_game == 0 && player == declarer_) {
        for (int card : skat_) if (card >= 0) obs.card_status[card] = KNOWN_DISCARD;
    }
    const auto record_trick = [&](const auto& cards, const auto& players, int trick, int size) {
        if (size == 0) {
            return;
        }
        const int required_suit = effective_suit(cards[0], game_kind_, trump_suit_);
        for (int slot = 0; slot < size; ++slot) {
            const int card = cards[slot];
            const int relative_player = (players[slot] - player + kNumPlayers) % kNumPlayers;
            obs.card_status[card] = PLAYED;
            obs.played_by[card] = relative_player;
            obs.trick_index[card] = trick;
            obs.trick_slot[card] = slot;
            if (slot > 0 && effective_suit(card, game_kind_, trump_suit_) != required_suit) {
                obs.void_info[relative_player * 5 + required_suit] = 1;
            }
        }
    };
    for (int trick = 0; trick < trick_index_; ++trick) {
        record_trick(history_cards_[trick], history_players_[trick], trick, kTrickSize);
    }
    if (!is_terminal()) {
        record_trick(current_trick_cards_, current_trick_players_, trick_index_, trick_pos_);
    }
    return obs;
}

std::vector<int> FastSkatGame::belief_targets(int player) const {
    validate_player(player);
    const int next_opponent = (player + 1) % kNumPlayers;
    const int previous_opponent = (player + 2) % kNumPlayers;
    std::vector<int> targets(kNumCards, -1);
    if (is_terminal()) return targets;

    for (int card = 0; card < kNumCards; ++card) {
        const uint32_t bit = uint32_t{1} << card;
        if (hands_[next_opponent] & bit) {
            targets[card] = 0;
        } else if (hands_[previous_opponent] & bit) {
            targets[card] = 1;
        } else if ((card == skat_[0] || card == skat_[1])
                   && (player != declarer_ || hand_game_ || phase_ == PICKUP_DECISION)) {
            targets[card] = 2;
        }
    }
    return targets;
}

StepInfo FastSkatGame::step(int action) {
    if (is_terminal()) {
        throw std::runtime_error("Cannot step terminated game. Call reset().");
    }

    if (!is_legal_action(action)) {
        throw std::invalid_argument("Illegal action.");
    }
    if (phase_ != CARD_PLAY) return step_preplay(action);
    const uint32_t action_bit = (uint32_t{1} << action);

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
        if (trick_index_ == kMaxTricks || (game_kind_ == NULL_GAME && winner == declarer_)) {
            phase_ = TERMINAL;
        } else {
            current_player_ = winner;
            trick_pos_ = 0;
            current_trick_cards_.fill(-1);
            current_trick_players_.fill(-1);
        }
    } else {
        current_player_ = (player + 1) % kNumPlayers;
    }

    return step_info();
}

StepInfo FastSkatGame::step_info() const {
    StepInfo info;
    info.terminated = is_terminal();
    info.phase = phase_;
    info.passed_out = is_terminal() && declarer_ < 0;
    info.current_player = current_player_;
    info.trick_index = trick_index_;
    if (info.passed_out) return info;
    if (is_terminal() && game_kind_ == NULL_GAME) {
        info.declarer_points = 0;
        info.defender_points = 0;
    } else if (is_terminal()) {
        // Include the hidden Skat only in the final score, not in observations.
        info.declarer_points = declarer_points() + card_points(skat_[0]) + card_points(skat_[1]);
        info.defender_points = 120 - info.declarer_points;
    } else {
        info.declarer_points = declarer_points();
        info.defender_points = defender_points();
    }
    if (is_terminal()) {
        info.declarer_won = declarer_won();
        info.game_value = final_game_value();
        info.overbid = raw_game_value() < winning_bid_;
    }
    return info;
}

StepInfo FastSkatGame::step_preplay(int action) {
    switch (phase_) {
        case BIDDING: step_bid(action); break;
        case PICKUP_DECISION:
            hand_game_ = action == 1;
            if (hand_game_) phase_ = CONTRACT_SELECTION;
            else {
                for (int card : skat_) hands_[declarer_] |= uint32_t{1} << card;
                skat_.fill(-1);
                phase_ = DISCARD;
            }
            break;
        case DISCARD: {
            const auto cards = hand(declarer_);
            int pair = 0;
            for (int i = 0; i < 12; ++i) {
                for (int j = i + 1; j < 12; ++j, ++pair) {
                    if (pair == action) skat_ = {cards[i], cards[j]};
                }
            }
            for (int card : skat_) hands_[declarer_] &= ~(uint32_t{1} << card);
            phase_ = CONTRACT_SELECTION;
            break;
        }
        case CONTRACT_SELECTION:
            game_kind_ = action < 4 ? SUIT : (action == GRAND_CONTRACT ? GRAND : NULL_GAME);
            trump_suit_ = action < 4 ? action : -1;
            phase_ = CARD_PLAY;
            current_player_ = forehand_;
            break;
    }
    return step_info();
}

void FastSkatGame::step_bid(int action) {
    const int player = current_player_;
    const int threshold = bid_values()[bid_index_];
    if (action == 0) {
        bid_status_[player] = PASSED;
        pass_threshold_[player] = threshold;
        pass_role_[player] = auction_role_;
        if (forehand_offer_) {
            phase_ = TERMINAL;
        } else finish_duel(auction_role_ == CALLER ? auction_holder_ : auction_caller_);
    } else if (forehand_offer_) finish_auction(player);
    else if (auction_role_ == CALLER) {
        highest_called_[player] = threshold;
        winning_bid_ = threshold;
        current_player_ = auction_holder_;
        auction_role_ = HOLDER;
    } else {
        highest_held_[player] = threshold;
        winning_bid_ = threshold;
        if (threshold == bid_values().back()) finish_auction(player);
        else {
            ++bid_index_;
            current_player_ = auction_caller_;
            auction_role_ = CALLER;
        }
    }
}

void FastSkatGame::finish_duel(int winner) {
    if (winning_bid_ == bid_values().back()) finish_auction(winner);
    else if (!rearhand_entered_) {
        rearhand_entered_ = true;
        auction_caller_ = (forehand_ + 2) % kNumPlayers;
        auction_holder_ = winner;
        current_player_ = auction_caller_;
        auction_role_ = CALLER;
        bid_status_[auction_caller_] = ACTIVE;
        bid_index_ = std::upper_bound(bid_values().begin(), bid_values().end(), winning_bid_) - bid_values().begin();
    } else if (winning_bid_ == 0) {
        forehand_offer_ = true;
        current_player_ = winner;
        auction_role_ = CALLER;
        bid_index_ = 0;
    } else finish_auction(winner);
}

void FastSkatGame::finish_auction(int winner) {
    declarer_ = current_player_ = winner;
    winning_bid_ = std::max(18, winning_bid_);
    phase_ = PICKUP_DECISION;
}

bool FastSkatGame::is_terminal() const { return phase_ == TERMINAL; }
int FastSkatGame::current_player() const { return current_player_; }
int FastSkatGame::declarer() const { return declarer_; }
int FastSkatGame::trick_index() const { return trick_index_; }
int FastSkatGame::game_kind() const { return game_kind_; }
int FastSkatGame::trump_suit() const { return trump_suit_; }
int FastSkatGame::trick_position() const { return trick_pos_; }
int FastSkatGame::phase() const { return phase_; }
int FastSkatGame::decision_threshold() const { return phase_ == BIDDING ? bid_values()[bid_index_] : -1; }
int FastSkatGame::winning_bid() const { return winning_bid_; }

int FastSkatGame::declarer_points() const {
    return declarer_ >= 0 ? won_points_[declarer_] : 0;
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
    if (skat_[0] < 0) return {};
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
    if (raw_game_value() < winning_bid_) return false;
    if (game_kind_ == NULL_GAME) {
        return !declarer_took_trick_;
    }
    return declarer_points() + card_points(skat_[0]) + card_points(skat_[1]) > 60;
}

int FastSkatGame::final_game_value() const {
    const int value = raw_game_value();
    if (value >= winning_bid_) return value;
    const int base = game_kind_ == GRAND ? 24 : 12 - trump_suit_;
    return ((winning_bid_ + base - 1) / base) * base;
}

int FastSkatGame::raw_game_value() const {
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

BatchedFastSkatEnv::BatchedFastSkatEnv(int size, int learning_player, int fixed_declarer, bool full_game)
    : games_(size),
      active_(size, 0),
      episode_lengths_(size, 0),
      learning_player_(learning_player),
      fixed_declarer_(fixed_declarer), full_game_(full_game) {
    if (size < 1) {
        throw std::invalid_argument("BatchedFastSkatEnv size must be at least 1.");
    }
    validate_player(learning_player_);
    if (full_game_ && fixed_declarer_ != -1) throw std::invalid_argument("Full games determine the declarer by bidding.");
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
        if (full_game_) {
            games_[index].reset_full(seeds[index]);
        } else if (fixed_declarer_ == -1) {
            games_[index].reset(seeds[index]);
        } else {
            games_[index].reset_fixed_declarer(seeds[index], fixed_declarer_);
        }
        active_[index] = 1;
        episode_lengths_[index] = 0;
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
        if (!game.is_legal_action(action)) {
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
        const bool learner_acted = game.current_player() == learning_player_;
        StepInfo step_info = game.step(actions[batch_index]);
        const float reward = reward_for_step(game, step_info);

        episode_lengths_[env_index] += learner_acted ? 1 : 0;

        const bool done = game.is_terminal();
        result.rewards.push_back(reward);
        result.terminated.push_back(done ? 1 : 0);

        if (done) {
            active_[env_index] = 0;
            result.completed_env_indices.push_back(env_index);
            result.completed_returns.push_back(reward);
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

std::vector<StructuredObservation> BatchedFastSkatEnv::active_observations() const {
    std::vector<StructuredObservation> observations;
    observations.reserve(active_count());
    for (int index : active_indices()) {
        const auto& game = games_[index];
        observations.push_back(game.build_observation(game.current_player()));
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
    masks.reserve(indices.size() * kNumActions);
    for (int env_index : indices) {
        for (bool legal : games_[env_index].legal_mask_array()) masks.push_back(legal ? 1 : 0);
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

int BatchedFastSkatEnv::action_dim() const {
    return kNumActions;
}

float BatchedFastSkatEnv::reward_for_step(const FastSkatGame& game, const StepInfo& info) const {
    if (!info.terminated || info.passed_out) {
        return 0.0F;
    }

    const int declarer = game.declarer();
    const bool declarer_won = info.declarer_won.has_value() && info.declarer_won.value();
    if (learning_player_ == declarer) {
        return static_cast<float>(declarer_won ? info.game_value + 50 : -2 * info.game_value - 50) / 100.0F;
    }
    return declarer_won ? 0.0F : 0.4F;
}

}  // namespace skat_rl
