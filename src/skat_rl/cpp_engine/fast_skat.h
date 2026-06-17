#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

namespace skat_rl {

constexpr int kNumPlayers = 3;
constexpr int kNumCards = 32;
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

struct StepInfo {
    bool terminated = false;
    int current_player = 0;
    int trick_index = 0;
    int declarer_points = 0;
    int defender_points = 0;
    std::optional<bool> declarer_won;
};

class FastSkatGame {
public:
    FastSkatGame();

    void reset(uint32_t seed);
    void reset_fixed_declarer(uint32_t seed, int fixed_declarer);
    void reset_from_deal(
        const std::vector<std::vector<int>>& hands,
        const std::vector<int>& skat,
        int declarer,
        int game_kind,
        int trump_suit,
        int current_player
    );

    std::vector<int> legal_actions() const;
    uint32_t legal_mask_bits() const;
    std::vector<bool> legal_mask_array() const;
    std::vector<float> observation(int player) const;
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

private:
    std::mt19937 rng_;
    std::array<uint32_t, kNumPlayers> hands_{};
    std::array<int, 2> skat_{};
    int declarer_ = 0;
    int game_kind_ = SUIT;
    int trump_suit_ = 0;
    int current_player_ = 0;
    int trick_index_ = 0;
    int trick_pos_ = 0;
    bool terminated_ = false;
    bool declarer_took_trick_ = false;
    std::array<int, kNumPlayers> won_points_{};
    std::array<std::array<int, kTrickSize>, kMaxTricks> history_cards_{};
    std::array<std::array<int, kTrickSize>, kMaxTricks> history_players_{};
    std::array<int, kTrickSize> current_trick_cards_{};
    std::array<int, kTrickSize> current_trick_players_{};

    void clear_state();
    int choose_declarer() const;
    int choose_trump_suit(uint32_t hand_mask) const;
    int current_trick_winner() const;
    int current_trick_points() const;
    bool declarer_won() const;
};

}  // namespace skat_rl
