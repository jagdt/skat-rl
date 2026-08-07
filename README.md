# Skat-RL

Skat-RL is a reinforcement-learning playground for the card game Skat. The project contains a Skat engine with card/rule utilities and random, heuristic, Stable-Baselines3 Maskable PPO, and native PyTorch PPO agents that can train and play against random and heuristic players.


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
python -m skat_rl.training.train_maskable_ppo
```

Run the from-scratch PyTorch PPO implementation:

```bash
python -m skat_rl.training.train_torch_ppo --total-timesteps 1000000 --n-envs 4
```

Plot a native PyTorch PPO run:

```bash
python -m skat_rl.training.plot_torch_ppo models/torch_ppo_skat_player0_YYYYMMDD_HHMMSS
```

## Outlook

This is an experimental project. Things that might be implemented in the future:

- Batched  and parallelized C++ Skat engine for fast training data generation
- Self-play of RL agents
- Transformer architecture
