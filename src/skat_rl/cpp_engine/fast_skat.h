#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

namespace skat_rl {

constexpr int kNumPlayers = 3;
constexpr int kNumCards = 32;
constexpr int kNumActions = 66;
constexpr int kCardsPerHand = 10;
constexpr int kMaxTricks = 10;
constexpr int kTrickSize = 3;
constexpr int kTrumpEffectiveSuit = 4;
enum GameKind {
    SUIT = 0,
    GRAND = 1,
    NULL_GAME = 2,
};

int card_suit(int card);
int card_rank(int card);
int card_points(int card);
bool is_trump(int card, int game_kind, int trump_suit);
int effective_suit(int card, int game_kind, int trump_suit);
int card_strength_in_trick(int card, int lead_card, int game_kind, int trump_suit);
uint32_t cards_to_mask(const std::vector<int>& cards);
std::vector<int> mask_to_cards(uint32_t mask);

// Public observation schema; LEFT = (observer + 1) % 3, RIGHT = +2.
enum CardStatus { UNKNOWN = 0, OWN = 1, PLAYED = 2, KNOWN_DISCARD = 3 };
enum Phase { BIDDING = 0, PICKUP_DECISION = 1, DISCARD = 2, CONTRACT_SELECTION = 3, CARD_PLAY = 4, TERMINAL = 5 };
enum Contract { CLUBS = 0, SPADES = 1, HEARTS = 2, DIAMONDS = 3, GRAND_CONTRACT = 4, NULL_CONTRACT = 5 };
enum AuctionRole { CALLER = 0, HOLDER = 1 };
enum BiddingStatus { NOT_ENTERED = 0, ACTIVE = 1, PASSED = 2 };
const std::vector<int>& bid_values();

struct StructuredObservation {
    int16_t phase = CARD_PLAY;
    std::array<int16_t, kNumCards> card_status{};
    std::array<int16_t, kNumCards> played_by{};
    std::array<int16_t, kNumCards> trick_index{};
    std::array<int16_t, kNumCards> trick_slot{};
    int16_t contract = -1;
    int16_t relative_declarer = -1;
    int16_t relative_current_leader = -1;
    int16_t declarer_points = 0;
    int16_t defender_points = 0;
    int16_t current_trick = 0;
    std::array<int16_t, kNumPlayers * 5> void_info{};
    int16_t seat = 0;
    int16_t auction_role = -1;
    int16_t decision_threshold = -1;
    int16_t hand_game = -1;
    std::array<int16_t, kNumPlayers> bid_status{};
    std::array<int16_t, kNumPlayers> highest_called{};
    std::array<int16_t, kNumPlayers> highest_held{};
    std::array<int16_t, kNumPlayers> pass_threshold{};
    std::array<int16_t, kNumPlayers> pass_role{};

    StructuredObservation() {
        played_by.fill(-1);
        trick_index.fill(-1);
        trick_slot.fill(-1);
    }
};

struct StepInfo {
    bool terminated = false;
    bool passed_out = false;
    bool overbid = false;
    int phase = CARD_PLAY;
    int current_player = 0;
    int trick_index = 0;
    int declarer_points = 0;
    int defender_points = 0;
    std::optional<bool> declarer_won;
    int game_value = 0;
};

class FastSkatGame {
public:
    FastSkatGame();

    void reset(uint64_t seed);
    void reset_full(uint64_t seed);
    void reset_full_from_deal(const std::vector<std::vector<int>>& hands,
                             const std::vector<int>& skat, int forehand = 0);
    void reset_fixed_declarer(uint64_t seed, int fixed_declarer);
    void reset_from_deal(
        const std::vector<std::vector<int>>& hands,
        const std::vector<int>& skat,
        int declarer,
        int game_kind,
        int trump_suit,
        int current_player,
        bool hand_game = false
    );

    std::vector<int> legal_actions() const;
    bool is_legal_action(int action) const;
    uint32_t legal_mask_bits() const;
    std::vector<bool> legal_mask_array() const;
    StructuredObservation build_observation(int player) const;
    std::vector<int> belief_targets(int player) const;
    StepInfo step(int action);

    bool is_terminal() const;
    int current_player() const;
    int declarer() const;
    int trick_index() const;
    int declarer_points() const;
    int defender_points() const;
    std::vector<int> hand(int player) const;
    std::vector<int> skat() const;
    std::vector<std::vector<int>> history_cards() const;
    std::vector<std::vector<int>> history_players() const;
    std::vector<int> current_trick_cards() const;
    std::vector<int> current_trick_players() const;
    int game_kind() const;
    int trump_suit() const;
    int trick_position() const;
    int phase() const;
    int decision_threshold() const;
    int winning_bid() const;

private:
    std::mt19937 rng_;
    std::array<uint32_t, kNumPlayers> hands_{};
    std::array<int, 2> skat_{};
    int declarer_ = 0;
    int game_kind_ = SUIT;
    int trump_suit_ = 0;
    int current_player_ = 0;
    int phase_ = CARD_PLAY;
    int forehand_ = 0;
    int winning_bid_ = 0;
    int bid_index_ = 0;
    int auction_caller_ = 1;
    int auction_holder_ = 0;
    int auction_role_ = CALLER;
    bool rearhand_entered_ = false;
    bool forehand_offer_ = false;
    std::array<int, kNumPlayers> bid_status_{};
    std::array<int, kNumPlayers> highest_called_{};
    std::array<int, kNumPlayers> highest_held_{};
    std::array<int, kNumPlayers> pass_threshold_{};
    std::array<int, kNumPlayers> pass_role_{};
    int trick_index_ = 0;
    int trick_pos_ = 0;
    bool declarer_took_trick_ = false;
    bool hand_game_ = false;
    int declarer_tricks_ = 0;
    std::array<int, kNumPlayers> won_points_{};
    std::array<std::array<int, kTrickSize>, kMaxTricks> history_cards_{};
    std::array<std::array<int, kTrickSize>, kMaxTricks> history_players_{};
    std::array<int, kTrickSize> current_trick_cards_{};
    std::array<int, kTrickSize> current_trick_players_{};

    void clear_state();
    void deal(uint64_t seed);
    void begin_auction(int forehand);
    void step_bid(int action);
    void finish_duel(int winner);
    void finish_auction(int winner);
    StepInfo step_preplay(int action);
    StepInfo step_info() const;
    int choose_declarer() const;
    int choose_trump_suit(uint32_t hand_mask) const;
    int current_trick_winner() const;
    int current_trick_points() const;
    bool declarer_won() const;
    int final_game_value() const;
    int raw_game_value() const;
};

struct BatchedStepInfo {
    std::vector<int> env_indices;
    std::vector<float> rewards;
    std::vector<uint8_t> terminated;
    std::vector<int> completed_env_indices;
    std::vector<float> completed_returns;
    std::vector<int> completed_lengths;
};

class BatchedFastSkatEnv {
public:
    BatchedFastSkatEnv(int size, int learning_player, int fixed_declarer = -1, bool full_game = false);

    void reset(uint64_t seed);
    void reset_many(const std::vector<uint64_t>& seeds);
    BatchedStepInfo step(const std::vector<int>& actions);
    std::vector<int> active_indices() const;
    std::vector<int> active_players() const;
    std::vector<int> active_declarers() const;
    std::vector<StructuredObservation> active_observations() const;
    std::vector<uint8_t> active_action_masks() const;
    std::vector<int> active_belief_targets() const;
    int active_count() const;
    int size() const;
    int learning_player() const;
    int action_dim() const;

private:
    std::vector<FastSkatGame> games_;
    std::vector<uint8_t> active_;
    std::vector<int> episode_lengths_;
    int learning_player_ = 0;
    int fixed_declarer_ = -1;
    bool full_game_ = false;
    float reward_for_step(const FastSkatGame& game, const StepInfo& info) const;
};

}  // namespace skat_rl
