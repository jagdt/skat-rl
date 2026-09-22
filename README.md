# Skat-RL

Skat-RL is a reinforcement-learning playground for the card game Skat. The project contains a Skat engine with card/rule utilities and random, heuristic, Stable-Baselines3 Maskable PPO, and native PyTorch PPO agents. Native PPO also supports batched C++ self-play against a frozen trained policy.

## Architecture

The transformer has a shared Skat encoder and separate value and per-card policy heads.
An optional supervised hidden-card belief head trains the shared encoder but does
not feed its predictions into policy or value.
The Python environment (`skat_rl.envs.skat_python_env`) supports both SB3 and native
PyTorch training. A batched C++ environment (`skat_rl.envs.skat_cpp_batched_env`)
is also available.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -e .
```

Install all training dependencies, including SB3:

```bash
pip install -e ".[training]"
```

Run SB3 Maskable PPO training:

```bash
python -m skat_rl.training.train_sb3_ppo
```

Run the from-scratch PyTorch PPO implementation:

```bash
python -m skat_rl.training.train_torch_ppo
```

Plot a native PyTorch PPO run:

```bash
python -m skat_rl.training.plot_torch_ppo models/torch_ppo_skat_player0_YYYYMMDD_HHMMSS
```

## Supervised Pretraining

Prepare recorded card plays from the ISS or SkatGame archives.

```bash
python -m skat_rl.training.prepare_supervised data/skatgame-games-07-2024.sgf \
  --output-dir data/prepared_skatgame
```

The default selection includes players with a recorded rating of at least 900
and 100 earlier appearances in the supplied data. Trusted names bypass both
filters. To use only a numerical rating cutoff:

```bash
python -m skat_rl.training.prepare_supervised data/skatgame-games-07-2024.sgf \
  --output-dir data/prepared_rating950 --min-rating 950 --trusted-players
```

Filtering applies to the player making each move; all players' moves are replayed.
Both winning and losing games are retained. Defaults include ordinary suit and
Grand games, including Hand, with all 30 card plays. Null, ouvert, announced
Schneider/Schwarz, overbids, timeouts, disconnections, card reveals, and resignations
are excluded in this first importer. Forced actions are omitted unless
`--include-forced` is supplied.

Every accepted game is replayed through the Python engine with its recorded deal,
pickup/discards, and contract. Turns, legal cards, final points, trick counts, and
outcome are checked before any examples are written. Incomplete trailing SGF records
are counted as malformed; an incomplete bzip2 stream is an error. The parser targets
the archives' one-record-per-line ISS dialect, not arbitrary SGF variation trees.

Output contains compressed NumPy shards and `manifest.json` with counts, filters,
and sample rejection reasons. Train/validation are split by platform and session;
duplicate record IDs and duplicate initial deals are removed across all inputs using
an on-disk SQLite index. This prevents the same game/deal from appearing on both
sides. Output directories must not already exist. Memory is bounded by shard size,
although each training worker decompresses its own shard. The default 8192-row
shard has about 38 MB of observation data before compression.

Train the transformer using legal-action-masked cross-entropy:

```bash
python -m skat_rl.training.train_supervised data/prepared_skatgame \
  --output-dir models/skat_pretrained --epochs 10 --batch-size 256
```

The default model is the existing 4-layer, 256-dimensional transformer. Its size can
be changed using the `--transformer-*` options. `--workers` loads shards in parallel.
Training logs `metrics.csv`, saves `best.pt` and `last.pt`, and stops after three
epochs without improvement in validation policy loss (`--patience`).

Belief learning is off by default. `--belief` adds supervised hidden-card prediction
using labels stored in the shards. Hidden hands and the Skat never enter the policy
observation. The observation encoder is shared with the Python Gym environment.
The value head is not pretrained: game scores are not treated as PPO value targets.
The current observation also omits the declarer's knowledge of discarded cards,
so imitation does not capture all information the recorded player possessed.

Initialize PPO with pretrained weights:

```bash
python -m skat_rl.training.train_torch_ppo \
  --init-model models/skat_pretrained/best.pt --env cpp --learning-rate 0.0001
```

`--init-model` adopts the checkpoint's architecture and weights, while using fresh
PPO hyperparameters and optimizer state. `--continue-model` retains its existing
resume behavior. These options are mutually exclusive. Pretrained checkpoints also
load with `PPOAgent.load()` for evaluation. They omit optimizer state and do not
support resuming the supervised optimizer/epoch counter.

## Self-Play

Train against a preselected transformer checkpoint, using the same supervised
checkpoint as the learner's starting point if desired:

```bash
python -m skat_rl.training.train_torch_ppo \
  --env cpp \
  --init-model models/skat_pretrained/best.pt \
  --opponent-model models/skat_pretrained/best.pt \
  --rollout-size 512 --learning-rate 0.0001
```

Declarer selection defaults to the existing hand heuristic in all native PPO runs:
choose the largest `number_of_jacks + number_of_aces + 0.4 * number_of_tens`
(ties go to the lowest seat). This is a simple proxy for hand strength, not a
calibrated win-probability estimate. The existing trump-suit heuristic is retained.
Games remain suit card-play games without bidding or Skat pickup/discard decisions.
The learning seat stays fixed (`--learning-player`, default 0), but its role changes
with the deal. `--fixed-declarer 0`, `1`, or `2` instead filters deals until that
seat is selected. Without `--opponent-model`, heuristic card-play opponents are
used, with the same heuristic declarer selection. `--fixed-declarer -1` remains
an explicit alias for the default heuristic selection.


### Batched Turn Interface

`SkatCppBatchedSingleAgentEnv(..., autoplay_opponents=False)` exposes every turn:

- `reset()` deals games without autoplaying any cards.
- State arrays contain `active_indices`, `current_players`, `declarers`,
  `observations`, `action_masks`, and `belief_targets`.
- Each observation, mask, and belief target belongs to that row's current player.
  Belief targets are privileged supervision labels, never policy inputs.
- `step(actions)` takes one integer action per active row and plays one card in
  each game inside C++. The entire action batch is validated before any mutation.
- `env_indices`, `rewards`, and `terminated` describe the submitted rows;
  `active_indices` and the new observation arrays describe the surviving games.
  Rewards always use the configured learning player's perspective, regardless
  of who acted. Completed games disappear from subsequent active batches.

The collector groups current-player observations by learner/frozen policy, performs
at most one forward pass per nonempty policy group, and submits a single combined
action batch. Different games may have different players to act. Python performs
trajectory bookkeeping but does not loop over individual games to step the engine.
The default `autoplay_opponents=True` preserves the heuristic-opponent interface.

After changing C++ sources, rebuild the extension before running:

```bash
python setup.py build_ext --inplace
```
