import argparse
import bz2
import json
import sys

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
from skat_rl.training.prepare_supervised import (
    _parse_args as parse_prepare_args, prepare_dataset, session_split,
)
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
    state = game.reset(game_type=GameType(GameKind.GRAND, hand=True), declarer=seed % 3, seed=seed)
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
                  min_prior_games=0, role="both", game_kinds=["suit", "grand"],
                  include_forced=True, validation_fraction=0.3, seed=42, shard_size=64, max_games=None)
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "games.sgf"
    source.write_text("".join(generated_record(seed) for seed in range(20)))
    directory = tmp_path / "prepared"
    manifest = prepare_dataset(prepare_args(source, directory))
    return directory, manifest


@pytest.mark.parametrize("arguments,expected", [
    ([], True), (["--include-forced"], True), (["--no-include-forced"], False),
])
def test_prepare_cli_includes_forced_by_default(monkeypatch, arguments, expected):
    monkeypatch.setattr(sys, "argv", [
        "prepare_supervised", "games.sgf", "--output-dir", "prepared", *arguments,
    ])
    args = parse_prepare_args()
    assert args.include_forced is expected
    assert not hasattr(args, "trusted_players")


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
    examples = replay_examples(record, [False, False, True])
    assert len(examples) == 10
    assert [e["remaining_decisions"] for e in examples] == list(range(9, -1, -1))
    assert [e["terminal_rewards"] for e in examples] == pytest.approx([0.98] * 10)
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
                assert np.all(data["action_masks"].sum(axis=1) >= 1)
                count += len(data["actions"])
        assert count == manifest["counts"][f"{split}_examples"]
        assert count == 20 * manifest["counts"][f"{split}_games"]
    assert manifest["args"]["include_forced"] is True
    assert "trusted_players" not in manifest["args"]
    assert not identities["train"] & identities["validation"]
    assert session_split("test", "same-session", 42, .3) == session_split("test", "same-session", 42, .3)


def test_rating_and_history_filters_apply_to_formerly_trusted_names(tmp_path):
    source = tmp_path / "games.sgf"
    records = "".join(generated_record(seed) for seed in range(20))
    records = (records.replace("P0[alice]", "P0[kermit]").replace("P1[bob]", "P1[zoot]")
               .replace("P2[carol]", "P2[theCount]").replace("R2[1100]", "R2[?]"))
    source.write_text(records)
    output = tmp_path / "prior"
    manifest = prepare_dataset(prepare_args(source, output, min_prior_games=5))
    assert manifest["counts"]["skipped_no_eligible_player"] == 5
    assert manifest["counts"]["train_games"] + manifest["counts"]["validation_games"] == 15
    for split in ("train", "validation"):
        assert manifest["counts"][f"{split}_examples"] == 10 * manifest["counts"][f"{split}_games"]
        for shard in manifest["splits"][split]:
            with np.load(output / shard["file"]) as data:
                assert np.all(data["observations"][:, 1114] == 1)  # Only the rated, eligible player.


def test_prepare_can_exclude_forced_moves(tmp_path):
    source = tmp_path / "games.sgf"
    source.write_text("".join(generated_record(seed) for seed in range(20)))
    output = tmp_path / "decisions"
    manifest = prepare_dataset(prepare_args(source, output, include_forced=False))
    for split in ("train", "validation"):
        assert manifest["counts"][f"{split}_examples"] < 20 * manifest["counts"][f"{split}_games"]
        for shard in manifest["splits"][split]:
            with np.load(output / shard["file"]) as data:
                assert np.all(data["action_masks"].sum(axis=1) > 1)


@pytest.mark.parametrize("value_coef", [0.0, 0.5])
def test_supervised_epoch_updates_policy_and_optional_value_head(prepared, value_coef):
    torch.set_num_threads(1)
    directory, _ = prepared
    config = PPOConfig(observation_dim=1149, action_dim=32, architecture="transformer",
                       transformer_dim=16, transformer_layers=1, transformer_heads=2,
                       transformer_ff_dim=32, use_belief=True, value_coef=value_coef)
    agent = PPOAgent(config, device="cpu")
    before = {name: p.detach().clone() for name, p in agent.model.named_parameters()}
    dataset = SupervisedBatches(directory, "train", batch_size=32)
    metrics = run_epoch(agent, dataset, training=True)
    assert np.isfinite(metrics["loss"])
    assert metrics["belief_loss"] > 0
    assert any(not torch.equal(before[n], p) for n, p in agent.model.named_parameters()
               if n.startswith("policy_head."))
    value_changed = any(not torch.equal(before[n], p) for n, p in agent.model.named_parameters()
                        if n.startswith("value_head."))
    assert value_changed == (value_coef > 0)
    assert (metrics["value_loss"] > 0) == (value_coef > 0)


def test_full_training_checkpoint_initializes_ppo_with_fresh_optimizer(prepared, tmp_path):
    directory, _ = prepared
    args = argparse.Namespace(dataset=str(directory), output_dir=str(tmp_path / "model"), epochs=1,
                              batch_size=32, patience=2, workers=0, torch_threads=1, seed=42,
                              device="cpu", transformer_dim=16, transformer_layers=1,
                              transformer_heads=2, transformer_ff_dim=32, transformer_dropout=0.0,
                              learning_rate=1e-3, belief=False, belief_coef=.05,
                              gamma=.95, value_coef=.5)
    path = train(args)
    loaded = PPOAgent.load(path, device="cpu")
    assert torch.load(path, weights_only=True)["critic_pretrained"] is True
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


def test_outcomes_are_discounted_by_player_decisions_not_selected_examples():
    record = decode_game(parse_record(RECORDED_GAME))
    selected = replay_examples(record, [False, False, True], include_forced=False)
    assert len(selected) < 10
    for example in selected:
        assert example["remaining_decisions"] == sum(example["observations"][:32]) - 1
        assert example["terminal_rewards"] == pytest.approx(.98)


def test_value_loss_uses_discounted_outcome(prepared):
    directory, _ = prepared
    config = PPOConfig(observation_dim=1149, action_dim=32, architecture="transformer",
                       transformer_dim=16, transformer_layers=1, transformer_heads=2,
                       transformer_ff_dim=32, gamma=.8)
    agent = PPOAgent(config, device="cpu")
    agent.model.eval()
    batch = next(iter(SupervisedBatches(directory, "train", 32)))
    with torch.no_grad():
        predicted = agent.model.outputs(batch["observations"])[1]
        targets = batch["terminal_rewards"] * .8 ** batch["remaining_decisions"].float()
        expected = .5 * ((predicted - targets) ** 2).mean().item()
    before = {name: p.clone() for name, p in agent.model.state_dict().items()}
    metrics = run_epoch(agent, [batch], training=False)
    assert metrics["value_loss"] == pytest.approx(expected)
    assert metrics["loss"] == pytest.approx(metrics["policy_loss"] + .5 * expected)
    assert all(torch.equal(before[n], p) for n, p in agent.model.state_dict().items())


def test_legacy_shards_require_repreparation_for_value_training(prepared):
    directory, manifest = prepared
    manifest["format_version"] = 1
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="no outcome labels"):
        SupervisedBatches(directory, "train", 32)
    batch = next(iter(SupervisedBatches(directory, "train", 32, require_value_targets=False)))
    assert "terminal_rewards" not in batch


def test_dataset_reward_scheme_must_match_ppo(prepared):
    directory, manifest = prepared
    manifest["reward_scheme"] = "other"
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="reward scheme"):
        SupervisedBatches(directory, "train", 32)


def test_replay_checks_recorded_game_value():
    record = decode_game(parse_record(RECORDED_GAME.replace("v:48", "v:49")))
    with pytest.raises(RecordError, match="game_value_mismatch"):
        replay_examples(record, [True] * 3)
