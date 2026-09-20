"""Stream ISS archives into replay-validated, compressed supervised shards."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from skat_rl.training.iss_data import (
    TRUSTED_PLAYERS, RecordError, base_player_name, decode_game, open_records,
    parse_record, player_rating, replay_examples,
)


class ShardWriter:
    def __init__(self, directory, split, shard_size):
        self.directory = directory
        self.split = split
        self.shard_size = shard_size
        self.rows = []
        self.shards = []

    def add(self, examples, game_id):
        for example in examples:
            self.rows.append({**example, "game_ids": game_id})
            if len(self.rows) >= self.shard_size:
                self.flush()

    def flush(self):
        if not self.rows:
            return
        filename = f"{self.split}-{len(self.shards):05d}.npz"
        dtypes = {"observations": np.float32, "action_masks": bool, "actions": np.uint8,
                  "belief_targets": np.int8, "game_ids": str}
        arrays = {key: np.asarray([r[key] for r in self.rows], dtype=dtype)
                  for key, dtype in dtypes.items()}
        np.savez_compressed(self.directory / filename, **arrays)
        self.shards.append({"file": filename, "examples": len(self.rows)})
        self.rows.clear()


def session_split(platform, session, seed, validation_fraction):
    identity = json.dumps([seed, platform, session]).encode()
    number = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") / 2**64
    return "validation" if number < validation_fraction else "train"


def prepare_dataset(args):
    if not np.isfinite(args.min_rating) or args.min_prior_games < 0:
        raise ValueError("Rating must be finite and min-prior-games nonnegative.")
    if not 0 < args.validation_fraction < 1 or args.shard_size < 1:
        raise ValueError("Use 0 < validation-fraction < 1 and a positive shard-size.")
    if args.max_games is not None and args.max_games < 1:
        raise ValueError("max-games must be positive.")
    for path in args.inputs:
        if not Path(path).is_file() or Path(path).stat().st_size == 0:
            raise ValueError(f"Missing or empty input: {path}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    writers = {split: ShardWriter(output, split, args.shard_size)
               for split in ("train", "validation")}
    counts = Counter()
    prior_games = Counter()
    errors = []
    trusted = set(args.trusted_players)

    # On-disk identity indexes keep deduplication memory bounded for large archives.
    with sqlite3.connect(output / "identities.sqlite") as database:
        database.execute("CREATE TABLE records (identity TEXT PRIMARY KEY)")
        database.execute("CREATE TABLE deals (identity TEXT PRIMARY KEY)")
        for path in args.inputs:
            with open_records(path) as stream:
                for line_number, line in enumerate(stream, 1):
                    if args.max_games is not None and counts["records"] >= args.max_games:
                        break
                    if not line.strip():
                        continue
                    counts["records"] += 1
                    try:
                        properties = parse_record(line)
                        game_id = json.dumps([properties["PC"], properties["ID"]])
                        inserted = database.execute(
                            "INSERT OR IGNORE INTO records VALUES (?)", (game_id,)
                        ).rowcount
                        if not inserted:
                            raise RecordError("duplicate_record")
                        names = [base_player_name(properties[f"P{p}"]) for p in range(3)]
                        eligible = []
                        for player, name in enumerate(names):
                            rating = player_rating(properties, player)
                            eligible.append(name in trusted or (
                                rating is not None and rating >= args.min_rating
                                and prior_games[(properties["PC"], name)] >= args.min_prior_games
                            ))
                        for name in set(names):
                            prior_games[(properties["PC"], name)] += 1
                        if not any(eligible):
                            raise RecordError("no_eligible_player")
                        record = decode_game(properties, args.game_kinds)
                        examples = replay_examples(record, eligible, args.role, args.include_forced)
                        if not examples:
                            raise RecordError("no_selected_moves")
                        inserted = database.execute(
                            "INSERT OR IGNORE INTO deals VALUES (?)", (record.deal_key,)
                        ).rowcount
                        if not inserted:
                            raise RecordError("duplicate_deal")
                        split = session_split(properties["PC"], properties["SE"],
                                              args.seed, args.validation_fraction)
                        writers[split].add(examples, game_id)
                        counts[f"{split}_games"] += 1
                        counts[f"{split}_examples"] += len(examples)
                    except RecordError as error:
                        counts[f"skipped_{error.reason}"] += 1
                        if len(errors) < 20:
                            errors.append({"file": str(path), "line": line_number,
                                           "reason": error.reason, "detail": str(error)})
                    if counts["records"] % 10000 == 0:
                        database.commit()
                        print(f"records={counts['records']} train={counts['train_examples']} "
                              f"validation={counts['validation_examples']}", flush=True)
            if args.max_games is not None and counts["records"] >= args.max_games:
                break
    for writer in writers.values():
        writer.flush()
    manifest = {
        "format_version": 1, "observation_dim": 1149, "action_dim": 32,
        "args": vars(args), "counts": dict(counts), "error_examples": errors,
        "splits": {split: writer.shards for split, writer in writers.items()},
    }
    with open(output / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(dict(counts), indent=2), flush=True)
    if not counts["train_examples"] or not counts["validation_examples"]:
        raise ValueError("No train or validation examples. Inspect manifest.json; use more games, "
                         "less restrictive filters, or another split seed.")
    return manifest


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Chronologically ordered .sgf or .sgf.bz2 files.")
    parser.add_argument("--output-dir", required=True, help="New directory for prepared shards.")
    parser.add_argument("--min-rating", type=float, default=1000)
    parser.add_argument("--min-prior-games", type=int, default=100)
    parser.add_argument("--trusted-players", nargs="*", default=list(TRUSTED_PLAYERS),
                        help="These accounts bypass rating/history filters; pass no names to disable.")
    parser.add_argument("--role", choices=["both", "declarer", "defender"], default="both")
    parser.add_argument("--game-kinds", nargs="+", choices=["suit", "grand"], default=["suit", "grand"])
    parser.add_argument("--include-forced", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-size", type=int, default=32768)
    parser.add_argument("--max-games", type=int, help="Limit inspected records for smoke tests.")
    return parser.parse_args()


def main():
    prepare_dataset(_parse_args())


if __name__ == "__main__":
    main()
