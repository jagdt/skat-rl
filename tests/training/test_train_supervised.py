import argparse
import bz2

import numpy as np
import pytest
import torch

from skat_rl.agents.ppo_agent import PPOAgent, PPOConfig
from skat_rl.engine.game import SkatGame
from skat_rl.engine.state import GameKind, GameType
from skat_rl.envs.observations import encode_observation
from skat_rl.training.iss_data import (
    RecordError, decode_game, open_records, parse_record, player_rating, replay_examples,
)
from skat_rl.training.prepare_supervised import prepare_dataset, session_split
from skat_rl.training.train_supervised import SupervisedBatches, run_epoch, train
from skat_rl.training.train_torch_ppo import initialize_agent


# A complete pickup/discard game from the supplied SkatGame archive.
RECORDED_GAME = (
    "(;GM[Skat]PC[ISS2]CO[]SE[167]ID[3690]DT[2016-02-27/01:24:06/UTC]"
    "P0[xskat]P1[xskat:2]P2[foo]R0[576.5]R1[576.5]R2[1220.7]"
    "MV[w HT.CQ.HK.S9.DK.CK.D9.C7.CJ.H9.S7.DA.DT.HJ.D7.SJ.SK.H7.ST.H8.CT.HQ."
    "CA.S8.C9.SA.HA.D8.DQ.C8.SQ.DJ 1 18 0 p 2 20 1 p 2 s w SQ.DJ 2 C.D8.DQ "
    "0 S9 1 S7 2 SQ 2 DJ 0 CJ 1 HJ 0 D9 1 DA 2 CA 2 C8 0 CK 1 SJ 1 DT 2 CT "
    "0 DK 2 SA 0 CQ 1 ST 0 H9 1 H7 2 HQ 2 HA 0 HK 1 H8 2 S8 0 HT 1 SK "
    "1 D7 2 C9 0 C7 ]R[d:2 win v:48 m:-3 bidok p:70 t:6 s:0 z:0 p0:0 p1:0 "
    "p2:0 l:-1 to:-1 r:0] ;)"
)


def generated_record(seed):
    def code(card):
        return "CSHD"[card // 8] + "789QKTAJ"[card % 8]

    game = SkatGame()
    state = game.reset(game_type=GameType(GameKind.GRAND), declarer=seed % 3, seed=seed)
    deck = [c for hand in state.hands for c in sorted(hand)] + state.skat
    moves = ["w", ".".join(map(code, deck)), str(state.declarer), "GH"]
    while not state.terminated:
        action = game.legal_actions()[0]
        moves.extend([str(state.current_player), code(action)])
        result = game.step(action).info.get("result")
    outcome = "win" if result["declarer_won"] else "loss"
    tricks = sum(p == state.declarer for p in state.trick_winners)
    return (f"(;GM[Skat]PC[test]SE[{seed}]ID[{seed}]P0[alice]P1[bob]P2[carol]"
            f"R0[1200]R1[900]R2[1100]MV[{' '.join(moves)}]"
            f"R[d:{state.declarer} {outcome} bidok p:{result['declarer_points']} "
            f"t:{tricks} to:-1 l:-1 r:0] ;)\n")


def prepare_args(path, output, **overrides):
    values = dict(inputs=[str(path)], output_dir=str(output), min_rating=1000,
                  min_prior_games=0, trusted_players=[], role="both", game_kinds=["suit", "grand"],
                  include_forced=False, validation_fraction=0.3, seed=42, shard_size=64, max_games=None)
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "games.sgf"
    source.write_text("".join(generated_record(seed) for seed in range(20)))
    directory = tmp_path / "prepared"
    manifest = prepare_dataset(prepare_args(source, directory))
    return directory, manifest


def test_parser_preserves_escaped_properties_and_unknown_rating():
    record = parse_record(RECORDED_GAME.replace("CO[]", r"CO[a\]b\\c]").replace("R0[576.5]", "R0[?]"))
    assert record["CO"] == "a]b\\c"
    assert player_rating(record, 0) is None
    assert player_rating(record, 2) == 1220.7
    with pytest.raises(RecordError):
        parse_record(RECORDED_GAME[:-3])


def test_real_record_replays_pickup_discards_and_teacher_selection():
    record = decode_game(parse_record(RECORDED_GAME))
    assert record.game.state.skat == [25, 27]  # D8 and DQ were discarded.
    examples = replay_examples(record, [False, False, True], include_forced=True)
    assert len(examples) == 10
    assert all(example["observations"][1116] == 1 for example in examples)
    for example in examples:
        assert example["action_masks"][example["actions"]]
        assert example["observations"][example["actions"]] == 1
        assert example["belief_targets"][example["actions"]] == -1
        assert np.all(example["observations"][:32][example["belief_targets"] >= 0] == 0)
    assert record.game.state.terminated


def test_replay_rejects_bad_turn_and_result():
    record = decode_game(parse_record(RECORDED_GAME))
    record.plays[0] = (1, record.plays[0][1])
    with pytest.raises(RecordError, match="Player"):
        replay_examples(record, [True] * 3)
    record = decode_game(parse_record(RECORDED_GAME.replace("p:70", "p:69")))
    with pytest.raises(RecordError, match="result_mismatch"):
        replay_examples(record, [True] * 3)


@pytest.mark.parametrize("old,new,reason", [
    ("to:-1", "to:0", "timeout_or_disconnect"),
    ("r:0]", "r:1]", "resignation"),
    ("bidok", "overbid", "penalty_or_overbid"),
    ("2 C.D8.DQ", "2 CO.D8.DQ", "unsupported_contract"),
    ("0 S9", "0 SC 0 S9", "reveal_or_shortened_game"),
])
def test_unsupported_records_are_explicitly_rejected(old, new, reason):
    with pytest.raises(RecordError) as error:
        decode_game(parse_record(RECORDED_GAME.replace(old, new)))
    assert error.value.reason == reason


def test_replay_observations_match_cpp_engine():
    from skat_rl._skat_cpp import FastSkatGame

    record = decode_game(parse_record(RECORDED_GAME))
    game = record.game
    state = game.state
    cpp = FastSkatGame()
    cpp.reset_from_deal([sorted(hand) for hand in state.hands], state.skat,
                        state.declarer, 0, int(state.game_type.trump_suit), 0)
    for player, action in record.plays:
        assert encode_observation(state, player) == pytest.approx(cpp.observation(player))
        game.step(action)
        cpp.step(action)


def test_compressed_input_deduplication_and_truncated_tail(tmp_path):
    records = "".join(generated_record(seed) for seed in range(20))
    duplicate_deal = generated_record(0).replace("ID[0]", "ID[100]").replace("SE[0]", "SE[100]")
    source = tmp_path / "games.sgf.bz2"
    with bz2.open(source, "wt") as stream:
        stream.write(records + records + duplicate_deal + "(;GM[Skat]")
    with open_records(source) as stream:
        assert parse_record(next(stream))["GM"] == "Skat"
    manifest = prepare_dataset(prepare_args(source, tmp_path / "dataset"))
    counts = manifest["counts"]
    assert counts["skipped_duplicate_record"] == 20
    assert counts["skipped_duplicate_deal"] == 1
    assert counts["skipped_malformed_record"] == 1
    assert counts["train_games"] + counts["validation_games"] == 20


def test_shards_exclude_weak_players_and_keep_games_in_one_split(prepared):
    directory, manifest = prepared
    identities = {}
    for split in ("train", "validation"):
        identities[split] = set()
        count = 0
        for shard in manifest["splits"][split]:
            with np.load(directory / shard["file"]) as data:
                identities[split].update(data["game_ids"].tolist())
                assert np.all(data["observations"][:, 1115] == 0)  # Bob is below 1000.
                assert np.all(data["action_masks"].sum(axis=1) > 1)
                count += len(data["actions"])
        assert count == manifest["counts"][f"{split}_examples"]
    assert not identities["train"] & identities["validation"]
    assert session_split("test", "same-session", 42, .3) == session_split("test", "same-session", 42, .3)


def test_prior_games_filter_and_trusted_override(tmp_path):
    source = tmp_path / "games.sgf"
    source.write_text("".join(generated_record(seed) for seed in range(20)))
    manifest = prepare_dataset(prepare_args(source, tmp_path / "prior", min_prior_games=5))
    assert manifest["counts"]["skipped_no_eligible_player"] == 5
    trusted = prepare_dataset(prepare_args(source, tmp_path / "trusted", min_prior_games=100,
                                          trusted_players=["bob"]))
    assert trusted["counts"]["train_games"] + trusted["counts"]["validation_games"] == 20


def test_supervised_epoch_updates_policy_but_not_value_head(prepared):
    torch.set_num_threads(1)
    directory, _ = prepared
    config = PPOConfig(observation_dim=1149, action_dim=32, architecture="transformer",
                       transformer_dim=16, transformer_layers=1, transformer_heads=2,
                       transformer_ff_dim=32, use_belief=True)
    agent = PPOAgent(config, device="cpu")
    before = {name: p.detach().clone() for name, p in agent.model.named_parameters()}
    dataset = SupervisedBatches(directory, "train", batch_size=32)
    metrics = run_epoch(agent, dataset, training=True)
    assert np.isfinite(metrics["loss"])
    assert metrics["belief_loss"] > 0
    assert any(not torch.equal(before[n], p) for n, p in agent.model.named_parameters()
               if n.startswith("policy_head."))
    assert all(torch.equal(before[n], p) for n, p in agent.model.named_parameters()
               if n.startswith("value_head."))


def test_full_training_checkpoint_initializes_ppo_with_fresh_optimizer(prepared, tmp_path):
    directory, _ = prepared
    args = argparse.Namespace(dataset=str(directory), output_dir=str(tmp_path / "model"), epochs=1,
                              batch_size=32, patience=2, workers=0, torch_threads=1, seed=42,
                              device="cpu", transformer_dim=16, transformer_layers=1,
                              transformer_heads=2, transformer_ff_dim=32, transformer_dropout=0.0,
                              learning_rate=1e-3, belief=False, belief_coef=.05)
    path = train(args)
    loaded = PPOAgent.load(path, device="cpu")
    config = PPOConfig(observation_dim=1149, action_dim=32, learning_rate=2e-4, gamma=.95)
    ppo = initialize_agent(config, path, device="cpu")
    assert ppo.config.transformer_dim == 16
    assert ppo.config.learning_rate == 2e-4
    assert ppo.config.gamma == .95
    assert not ppo.optimizer.state
    assert all(torch.equal(p, ppo.model.state_dict()[name]) for name, p in loaded.model.state_dict().items())
    assert (path.parent / "metrics.csv").exists()


def test_multiple_loader_workers_do_not_repeat_examples(prepared):
    directory, manifest = prepared
    dataset = SupervisedBatches(directory, "train", batch_size=32, shuffle=True)
    # Worker sharding is also exercised without requiring subprocess permissions.
    from unittest.mock import patch
    counts = []
    for worker_id in range(2):
        worker = argparse.Namespace(id=worker_id, num_workers=2)
        with patch("skat_rl.training.train_supervised.get_worker_info", return_value=worker):
            counts.append(sum(len(batch["actions"]) for batch in dataset))
    assert sum(counts) == manifest["counts"]["train_examples"]
