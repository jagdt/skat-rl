#include "fast_skat.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

py::dict step_info_to_dict(const skat_rl::StepInfo& info) {
    py::dict result;
    result["terminated"] = info.terminated;
    result["current_player"] = info.current_player;
    result["trick_index"] = info.trick_index;
    result["declarer_points"] = info.declarer_points;
    result["defender_points"] = info.defender_points;
    if (info.declarer_won.has_value()) {
        result["declarer_won"] = info.declarer_won.value();
    } else {
        result["declarer_won"] = py::none();
    }
    return result;
}

py::dict state_summary(const skat_rl::FastSkatGame& game) {
    py::dict result;
    result["terminated"] = game.is_terminal();
    result["current_player"] = game.current_player();
    result["declarer"] = game.declarer();
    result["game_kind"] = game.game_kind();
    result["trump_suit"] = game.trump_suit();
    result["trick_index"] = game.trick_index();
    result["trick_position"] = game.trick_position();
    result["declarer_points"] = game.declarer_points();
    result["defender_points"] = game.defender_points();
    result["hands"] = py::make_tuple(game.hand(0), game.hand(1), game.hand(2));
    result["skat"] = game.skat();
    result["history_cards"] = game.history_cards();
    result["history_players"] = game.history_players();
    result["current_trick_cards"] = game.current_trick_cards();
    result["current_trick_players"] = game.current_trick_players();
    return result;
}

}  // namespace

PYBIND11_MODULE(_skat_cpp, m) {
    m.doc() = "C++ Skat card-play engine core";

    py::class_<skat_rl::FastSkatGame>(m, "FastSkatGame")
        .def(py::init<>())
        .def("reset", &skat_rl::FastSkatGame::reset, py::arg("seed"))
        .def(
            "reset_fixed_declarer",
            &skat_rl::FastSkatGame::reset_fixed_declarer,
            py::arg("seed"),
            py::arg("fixed_declarer")
        )
        .def(
            "reset_from_deal",
            &skat_rl::FastSkatGame::reset_from_deal,
            py::arg("hands"),
            py::arg("skat"),
            py::arg("declarer"),
            py::arg("game_kind"),
            py::arg("trump_suit"),
            py::arg("current_player") = 0
        )
        .def("legal_actions", &skat_rl::FastSkatGame::legal_actions)
        .def("legal_mask_bits", &skat_rl::FastSkatGame::legal_mask_bits)
        .def("legal_mask_array", &skat_rl::FastSkatGame::legal_mask_array)
        .def("observation", &skat_rl::FastSkatGame::observation, py::arg("player"))
        .def("step", [](skat_rl::FastSkatGame& game, int action) {
            return step_info_to_dict(game.step(action));
        })
        .def("is_terminal", &skat_rl::FastSkatGame::is_terminal)
        .def("current_player", &skat_rl::FastSkatGame::current_player)
        .def("declarer", &skat_rl::FastSkatGame::declarer)
        .def("declarer_points", &skat_rl::FastSkatGame::declarer_points)
        .def("defender_points", &skat_rl::FastSkatGame::defender_points)
        .def("hand", &skat_rl::FastSkatGame::hand, py::arg("player"))
        .def("skat", &skat_rl::FastSkatGame::skat)
        .def("history_cards", &skat_rl::FastSkatGame::history_cards)
        .def("history_players", &skat_rl::FastSkatGame::history_players)
        .def("current_trick_cards", &skat_rl::FastSkatGame::current_trick_cards)
        .def("current_trick_players", &skat_rl::FastSkatGame::current_trick_players)
        .def("trick_index", &skat_rl::FastSkatGame::trick_index)
        .def("trick_position", &skat_rl::FastSkatGame::trick_position)
        .def("game_kind", &skat_rl::FastSkatGame::game_kind)
        .def("trump_suit", &skat_rl::FastSkatGame::trump_suit)
        .def("state_summary", &state_summary);
}
