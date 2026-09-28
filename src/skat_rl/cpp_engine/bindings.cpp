#include "fast_skat.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
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
    result["game_value"] = info.game_value;
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

uint64_t uint64_from_python_int(const py::handle& value, const char* argument_name) {
    py::object index = py::reinterpret_steal<py::object>(PyNumber_Index(value.ptr()));
    if (!index) {
        PyErr_Clear();
        throw py::type_error(std::string(argument_name) + " must be an integer.");
    }

    const unsigned long long converted = PyLong_AsUnsignedLongLong(index.ptr());
    if (PyErr_Occurred()) {
        PyErr_Clear();
        throw py::value_error(
            std::string(argument_name) + " must be in range 0..2**64 - 1."
        );
    }
    return static_cast<uint64_t>(converted);
}

std::vector<uint64_t> uint64_vector_from_sequence(const py::sequence& values) {
    std::vector<uint64_t> result;
    result.reserve(static_cast<std::size_t>(values.size()));
    for (const py::handle value : values) {
        result.push_back(uint64_from_python_int(value, "seed"));
    }
    return result;
}

template <typename T>
py::array_t<int8_t> observation_field(
    const std::vector<skat_rl::StructuredObservation>& observations,
    T skat_rl::StructuredObservation::* member, std::vector<py::ssize_t> shape, bool batched
) {
    if (batched) {
        shape.insert(shape.begin(), static_cast<py::ssize_t>(observations.size()));
    }
    py::array_t<int8_t> result(shape);
    for (std::size_t row = 0; row < observations.size(); ++row) {
        std::memcpy(result.mutable_data() + row * sizeof(T), &(observations[row].*member), sizeof(T));
    }
    return result;
}

py::dict observation_dict(const std::vector<skat_rl::StructuredObservation>& observations,
                          bool batched = true) {
    using Observation = skat_rl::StructuredObservation;
    py::dict result;
    result["phase"] = observation_field(observations, &Observation::phase, {}, batched);
    result["card_status"] = observation_field(observations, &Observation::card_status, {32}, batched);
    result["played_by"] = observation_field(observations, &Observation::played_by, {32}, batched);
    result["trick_index"] = observation_field(observations, &Observation::trick_index, {32}, batched);
    result["trick_slot"] = observation_field(observations, &Observation::trick_slot, {32}, batched);
    result["contract"] = observation_field(observations, &Observation::contract, {}, batched);
    result["relative_declarer"] = observation_field(observations, &Observation::relative_declarer, {}, batched);
    result["relative_current_leader"] = observation_field(observations, &Observation::relative_current_leader, {}, batched);
    result["declarer_points"] = observation_field(observations, &Observation::declarer_points, {}, batched);
    result["defender_points"] = observation_field(observations, &Observation::defender_points, {}, batched);
    result["current_trick"] = observation_field(observations, &Observation::current_trick, {}, batched);
    result["void_info"] = observation_field(observations, &Observation::void_info, {3, 5}, batched);
    return result;
}

py::array_t<int> int_array(const std::vector<int>& values) {
    py::array_t<int> array(values.size());
    if (!values.empty()) {
        std::memcpy(array.mutable_data(), values.data(), values.size() * sizeof(int));
    }
    return array;
}

py::array_t<int> int_array_2d(const std::vector<int>& values,
                              py::ssize_t rows,
                              py::ssize_t cols) {
    py::array_t<int> array({rows, cols});
    if (!values.empty()) {
        std::memcpy(array.mutable_data(), values.data(), values.size() * sizeof(int));
    }
    return array;
}

py::array_t<float> float_array_1d(const std::vector<float>& values) {
    py::array_t<float> array(values.size());
    if (!values.empty()) {
        std::memcpy(array.mutable_data(), values.data(), values.size() * sizeof(float));
    }
    return array;
}

py::array_t<bool> bool_array(const std::vector<uint8_t>& values, py::ssize_t rows, py::ssize_t cols) {
    py::array_t<bool> array({rows, cols});
    auto* output = static_cast<bool*>(array.mutable_data());
    for (std::size_t index = 0; index < values.size(); ++index) {
        output[index] = values[index] != 0;
    }
    return array;
}

py::array_t<bool> bool_array_1d(const std::vector<uint8_t>& values) {
    py::array_t<bool> array(values.size());
    auto* output = static_cast<bool*>(array.mutable_data());
    for (std::size_t index = 0; index < values.size(); ++index) {
        output[index] = values[index] != 0;
    }
    return array;
}

py::dict batched_step_info_to_dict(const skat_rl::BatchedFastSkatEnv& env,
                                   const skat_rl::BatchedStepInfo& info) {
    py::dict result;
    result["env_indices"] = int_array(info.env_indices);
    result["rewards"] = float_array_1d(info.rewards);
    result["terminated"] = bool_array_1d(info.terminated);
    result["completed_env_indices"] = int_array(info.completed_env_indices);
    result["completed_returns"] = float_array_1d(info.completed_returns);
    result["completed_lengths"] = int_array(info.completed_lengths);

    const int active_count = env.active_count();
    result["active_indices"] = int_array(env.active_indices());
    result["current_players"] = int_array(env.active_players());
    result["declarers"] = int_array(env.active_declarers());
    result["observations"] = observation_dict(env.active_observations());
    result["action_masks"] = bool_array(
        env.active_action_masks(),
        active_count,
        skat_rl::kNumCards
    );
    result["belief_targets"] = int_array_2d(
        env.active_belief_targets(),
        active_count,
        skat_rl::kNumCards
    );
    return result;
}

}  // namespace

PYBIND11_MODULE(_skat_cpp, m) {
    m.doc() = "C++ Skat card-play engine core";

    py::class_<skat_rl::FastSkatGame>(m, "FastSkatGame")
        .def(py::init<>())
        .def(
            "reset",
            [](skat_rl::FastSkatGame& game, py::handle seed) {
                game.reset(uint64_from_python_int(seed, "seed"));
            },
            py::arg("seed")
        )
        .def(
            "reset_fixed_declarer",
            [](skat_rl::FastSkatGame& game, py::handle seed, int fixed_declarer) {
                game.reset_fixed_declarer(
                    uint64_from_python_int(seed, "seed"),
                    fixed_declarer
                );
            },
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
            py::arg("current_player") = 0,
            py::arg("hand_game") = false
        )
        .def("legal_actions", &skat_rl::FastSkatGame::legal_actions)
        .def("legal_mask_bits", &skat_rl::FastSkatGame::legal_mask_bits)
        .def("legal_mask_array", &skat_rl::FastSkatGame::legal_mask_array)
        .def("observation", [](const skat_rl::FastSkatGame& game, int player) {
            return observation_dict({game.build_observation(player)}, false);
        }, py::arg("player"))
        .def("belief_targets", &skat_rl::FastSkatGame::belief_targets, py::arg("player"))
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

    py::class_<skat_rl::BatchedFastSkatEnv>(m, "BatchedFastSkatEnv")
        .def(
            py::init<int, int, int, bool>(),
            py::arg("size"),
            py::arg("learning_player"),
            py::arg("fixed_declarer") = -1,
            py::arg("autoplay_opponents") = true
        )
        .def(
            "reset",
            [](skat_rl::BatchedFastSkatEnv& env, py::handle seed) {
                env.reset(uint64_from_python_int(seed, "seed"));
            },
            py::arg("seed")
        )
        .def(
            "reset_many",
            [](skat_rl::BatchedFastSkatEnv& env, const py::sequence& seeds) {
                env.reset_many(uint64_vector_from_sequence(seeds));
            },
            py::arg("seeds")
        )
        .def("size", &skat_rl::BatchedFastSkatEnv::size)
        .def("active_count", &skat_rl::BatchedFastSkatEnv::active_count)
        .def("learning_player", &skat_rl::BatchedFastSkatEnv::learning_player)
        .def("action_dim", &skat_rl::BatchedFastSkatEnv::action_dim)
        .def("active_indices", [](const skat_rl::BatchedFastSkatEnv& env) {
            return int_array(env.active_indices());
        })
        .def("current_players", [](const skat_rl::BatchedFastSkatEnv& env) {
            return int_array(env.active_players());
        })
        .def("declarers", [](const skat_rl::BatchedFastSkatEnv& env) {
            return int_array(env.active_declarers());
        })
        .def("observations", [](const skat_rl::BatchedFastSkatEnv& env) {
            return observation_dict(env.active_observations());
        })
        .def("action_masks", [](const skat_rl::BatchedFastSkatEnv& env) {
            return bool_array(
                env.active_action_masks(),
                env.active_count(),
                skat_rl::kNumCards
            );
        })
        .def("belief_targets", [](const skat_rl::BatchedFastSkatEnv& env) {
            return int_array_2d(
                env.active_belief_targets(),
                env.active_count(),
                skat_rl::kNumCards
            );
        })
        .def("step", [](skat_rl::BatchedFastSkatEnv& env, const std::vector<int>& actions) {
            return batched_step_info_to_dict(env, env.step(actions));
        });
}
