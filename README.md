# Skat-RL

Skat-RL is a reinforcement-learning playground for the card game Skat. The project contains a Skat engine with card/rule utilities and random, heuristic, Stable-Baselines3 Maskable PPO, and native PyTorch PPO agents. Native PPO also supports batched C++ self-play against a frozen trained policy.

## Architecture

The transformer uses one shared encoder with exactly 36 tokens: STATE, 32 physical
CARD tokens, and three relative BID_INFO tokens. Five lightweight policy heads
handle bidding, pickup/Hand, discard, contract selection, and card play.
Only the relevant head runs for each phase; mixed-phase batches share one encoder pass.
Contract scoring and selected-contract input reuse the same contract embedding table. Bid amounts share one embedding table with separate called/held/pass/decision linear projections.
An optional supervised hidden-card belief head trains the shared encoder but does
not feed its predictions into policy or value.
The Python environment (`skat_rl.envs.skat_python_env`) supports both SB3 and native
PyTorch training. A batched C++ environment (`skat_rl.envs.skat_cpp_batched_env`)
is also available.

### Structured Observations

Policy inputs are dictionaries of named integer arrays, described by
`StructuredSkatObservation` in `envs/observations.py`. A single observation has
scalar globals and 32-element card arrays; batching adds a leading dimension `B`:

| Field | Batched shape | Meaning |
| --- | --- | --- |
| `phase` | `[B]` | Bidding, pickup, discard, contract, play, terminal: `0..5` |
| `card_status` | `[B, 32]` | `UNKNOWN=0`, `OWN=1`, `PLAYED=2`, `KNOWN_DISCARD=3` |
| `played_by` | `[B, 32]` | Relative player for each played card |
| `trick_index` | `[B, 32]` | Trick `0..9` in which each card was played |
| `trick_slot` | `[B, 32]` | Lead, second, third: `0..2` |
| `contract` | `[B]` | Clubs, spades, hearts, diamonds, Grand, Null: `0..5` |
| `relative_declarer` | `[B]` | Declarer's relative seat |
| `relative_current_leader` | `[B]` | Current trick leader's relative seat |
| `declarer_points`, `defender_points` | `[B]` each | Public points from completed tricks |
| `current_trick` | `[B]` | Trick index `0..9`; last played trick at termination |
| `void_info` | `[B, 3, 5]` | Publicly established voids by relative seat and effective suit |
| `seat` | `[B]` | Forehand, middlehand, rearhand: `0..2` |
| `auction_role` | `[B]` | Caller `0`, holder `1`; `-1` outside bidding |
| `decision_threshold` | `[B]` | Actual bid amount currently being decided; otherwise `-1` |
| `winning_bid` | `[B]` | Committed auction amount, initially `0` |
| `hand_game` | `[B]` | Unknown `-1`, pickup `0`, Hand `1` |
| `bid_status` | `[B, 3]` | SELF/LEFT/RIGHT: not entered `0`, active `1`, passed `2` |
| `highest_called`, `highest_held` | `[B, 3]` each | Separate actual bid amounts, absent `-1` |
| `pass_threshold`, `pass_role` | `[B, 3]` each | Declined amount and caller/holder role, absent `-1` |

Relative seats are `SELF=0`, `LEFT=1` (the next seat in engine play order), and
`RIGHT=2`. Absolute current-player IDs remain rollout-routing metadata, never
model inputs. Unplayed cards use `-1` for all three play-metadata fields, which
the tokenizer ignores. Current-trick cards are already `PLAYED` and have the same
metadata as completed-trick cards. No separate hand/history/current-trick one-hot
planes are stored.

Contract, relative declarer, and leader use `-1` until known. Discarded Skat cards
are `KNOWN_DISCARD` only to the pickup declarer, never to defenders. Hand declarers
do not see the Skat. Bid information persists through all later phases. Absent
fields add no embedding contribution; effective-suit embeddings start only once
the contract is known. Observations use `int16` to store actual bids through 264.

### Actions and Game Scope

Both environments use 66 masked action slots with phase-local meanings:

| Phase | Action IDs |
| --- | --- |
| Bidding | `0` pass, `1` continue (call or hold the current threshold) |
| Pickup | `0` pickup, `1` Hand |
| Discard | `0..65`: lexicographic `(i,j)`, `i<j`, from the sorted 12-card hand |
| Contract | `0..5`: clubs, spades, hearts, diamonds, Grand, Null |
| Card play | `0..31`: physical card IDs |

The discard head jointly scores `[STATE, CARD_i, CARD_j]` for all 66 pairs; the
engine and importer use the same canonical ordering. Inactive slots are masked.
Model outputs are logits `[B,66]`, values `[B]`, and optional beliefs `[B,32,3]`.
Belief is disabled during normal action collection.

Full-game training is the default. Add `--card-play-only` to preparation or training
to retain the former setup. Direct engine/wrapper callers opt in with `full_game=True`
(`FastSkatGame.reset_full(seed)` for C++). Six contracts, Hand, passed-out deals,
and terminal overbid losses are supported. Announced Schneider/Schwarz and ouvert
are not action choices. Null is masked above 23, or 35 for Hand, and ends as soon
as the declarer takes a trick. A passed-out deal returns zero for every seat.

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

Run the from-scratch PyTorch PPO implementation against Python heuristic opponents:

```bash
python -m skat_rl.training.train_torch_ppo --env python
```

Plot a native PyTorch PPO run:

```bash
python -m skat_rl.training.plot_torch_ppo models/torch_ppo_skat_player0_YYYYMMDD_HHMMSS
```

## Supervised Pretraining

Prepare recorded decisions from the ISS or SkatGame archives.

```bash
python -m skat_rl.training.prepare_supervised data/skatgame-games-07-2024.sgf \
  --output-dir data/prepared_skatgame
```

The default selection includes players with a recorded rating of at least 1000.
`--min-prior-games` optionally requires earlier appearances in the supplied data
(default: 0). Both filters apply to every player; players without ratings are
excluded. To use only a numerical rating cutoff:

```bash
python -m skat_rl.training.prepare_supervised data/skatgame-games-07-2024.sgf \
  --output-dir data/prepared_rating950 --min-rating 950 --min-prior-games 0
```

Filtering applies to the player making each move; all players' moves are replayed.
Both winning and losing games are retained. Defaults include ordinary suit, Grand,
and Null games, including Hand, passed-out deals, and replay-validated overbids.
Ouvert, announced Schneider/Schwarz, timeouts, disconnections, card reveals, and
resignations are excluded. Forced actions are included by default so
the critic also sees forced and late-game decisions. Use `--no-include-forced`
to retain only states with multiple legal actions. Forced actions have zero masked
policy loss and automatically count as correct in the current policy accuracy metric.

Every accepted game is replayed through the Python engine with its recorded deal,
auction, pickup/discards, and contract. Turns, legal actions, final points, trick counts,
outcome, and recorded game value (when present) are checked before any examples are
written. Incomplete trailing SGF records
are counted as malformed; an incomplete bzip2 stream is an error. The parser targets
the archives' one-record-per-line ISS dialect, not arbitrary SGF variation trees.
Numeric bid jumps become one CONTINUE example at the recorded threshold, not
fabricated intermediate bids. A subsequent contract after both other players pass
implies forehand accepted 18. Missing auctions are rejected in full-game mode;
`--card-play-only` can still extract those records. Null losses may end early.

Output contains compressed NumPy shards and `manifest.json` with counts, filters,
and sample rejection reasons. Train/validation are split by platform and session;
duplicate record IDs and duplicate initial deals are removed across all inputs using
an on-disk SQLite index. This prevents the same game/deal from appearing on both
sides. Output directories must not already exist. Memory is bounded by shard size,
although each training worker decompresses its own shard. The default 32768-row
shard has about 11.1 MB of observation array data before compression (170 int16
values per example), excluding labels and Python/NumPy object overhead.

Train the policy with legal-action-masked cross-entropy and the value head with
outcome regression:

```bash
python -m skat_rl.training.train_supervised data/prepared_skatgame \
  --output-dir models/skat_pretrained --epochs 10 --batch-size 256
```

The default model is the existing 4-layer, 256-dimensional transformer. Its size can
be changed using the `--transformer-*` options. `--workers` loads shards in parallel.
Training logs `metrics.csv`, saves `best.pt` and `last.pt`, and stops after three
epochs without improvement in combined validation loss (`--patience`).

Each example stores the player's final tournament reward and the number of their
decisions remaining after the recorded action (including omitted forced moves).
The value target is `gamma ** remaining_decisions * terminal_reward`, matching
PPO's discount per learner decision. Both trainers default to `--gamma 0.99`.
Use `--gamma 1` in both for undiscounted final scores. The supervised loss is
`policy_cross_entropy + value_coef * 0.5 * MSE(value, target)` plus optional belief
loss; `--value-coef` defaults to `0.5`. Training logs `value_loss` for both splits.
Outcomes are labels only, never observations. They estimate returns under the
recorded players' play; PPO must still adapt the critic to its own policy.

Prepared datasets use format version 3. Each
shard stores one `obs_<field>` array per named observation field, alongside action
masks, actions, belief targets, game IDs, and outcome labels.

Belief learning is off by default. `--belief` adds supervised hidden-card prediction
using labels stored in the shards. Hidden hands and unknown Skat cards never enter
the policy observation. Labels for own, played, or known-discard cards are ignored.
The observation encoder is shared with the Python Gym environment.

Initialize PPO with pretrained weights:

```bash
python -m skat_rl.training.train_torch_ppo \
  --init-model models/skat_pretrained/best.pt \
  --opponent-model models/skat_pretrained/best.pt --env cpp --learning-rate 0.0001
```

`--init-model` adopts the checkpoint's architecture and weights, while using fresh
PPO hyperparameters and optimizer state. `--continue-model` also restores the
saved PPO optimizer/configuration. Both require the new observation schema, as
do frozen opponents. These options are mutually exclusive. Pretrained checkpoints also
load with `PPOAgent.load()` for evaluation. They omit optimizer state and do not
support resuming the supervised optimizer/epoch counter.

## Tournament Rewards

Both engines and supervised labels use three-player Seeger-Fabian tournament
scoring, divided by 100. With unsigned game value `G`:

| Outcome | Declarer reward | Each defender's reward |
| --- | --- | --- |
| Declarer wins | `(G + 50) / 100` | `0` |
| Declarer loses | `(-2 * G - 50) / 100` | `0.4` |

Game value includes matadors (including the Skat), Hand, Schneider, and Schwarz;
Null and Null Hand have fixed values. An overbid loses with game value rounded up
to the smallest multiple of the contract's base value covering the winning bid.
Announced bonuses and ouvert are not implemented. Full-game agents choose pickup
or Hand themselves; only card-play-only generated deals use automatic Hand setup.
Recorded games retain their actual Hand/non-Hand contract.

Rewards are terminal-only and not zero-sum. In particular, a defender loss now
returns zero, not a negative reward. New return curves are not directly comparable
with old shaped-reward runs. There is no old-reward compatibility mode.

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

Full games begin with a random forehand and an auction. Every decision, including
bidding and contract selection, comes from the learner or frozen opponent. The
learning seat stays fixed (`--learning-player`, default 0), but its role varies.
C++ training requires `--opponent-model`.
Use `--env python` for heuristic opponents, which now also make setup decisions.

With `--card-play-only`, both engines retain the existing automatic declarer and
trump-suit setup. Declarer selection maximizes
`number_of_jacks + number_of_aces + 0.4 * number_of_tens`, breaking ties by seat.
`--fixed-declarer 0/1/2` filters deals to that seat and requires `--card-play-only`;
`-1` means unrestricted selection. This setup heuristic does not play any C++ turns.


### Batched Turn Interface

`SkatCppBatchedSingleAgentEnv(...)` always exposes every turn:

- `reset()` deals games without autoplaying any decisions.
- State arrays contain `active_indices`, `current_players`, `declarers`,
  `observations`, `action_masks`, and `belief_targets`.
- `observations` is a dictionary of the batched fields listed above, not a matrix.
  Each observation, mask, and belief target belongs to that row's current player.
  Belief targets are privileged supervision labels, never policy inputs.
- `step(actions)` takes one phase-local action per active row and advances one decision in
  each game inside C++. The entire action batch is validated before any mutation.
- `env_indices`, `rewards`, and `terminated` describe the submitted rows;
  `active_indices` and the new observation arrays describe the surviving games.
  Rewards always use the configured learning player's perspective, regardless
  of who acted. Completed games disappear from subsequent active batches.

The collector groups current-player observations by learner/frozen policy, performs
at most one forward pass per nonempty policy group, and submits a single combined
action batch. Different games may have different players and phases to act. Python performs
trajectory bookkeeping but does not loop over individual games to step the engine.
There is no C++ autoplay option or built-in opponent policy. Only the Python
environment retains heuristic opponents and advances their turns automatically.

After changing C++ sources, rebuild the extension before running:

```bash
python setup.py build_ext --inplace --force
```
