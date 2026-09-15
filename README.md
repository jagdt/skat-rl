# Skat-RL

Skat-RL is a reinforcement-learning playground for the card game Skat. The project contains a Skat engine with card/rule utilities and random, heuristic, Stable-Baselines3 Maskable PPO, and native PyTorch PPO agents that can train and play against random and heuristic players.

## Architecture

The transformer has a shared Skat encoder and separate value and per-card policy heads.
An optional supervised hidden-card belief head trains the shared encoder but does
not feed its predictions into policy or value.
Both a python and a batched C++ environment are available.

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

## Outlook

This is an experimental project. Things that might be implemented in the future:

- Self-play of RL agents
