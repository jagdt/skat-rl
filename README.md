# Skat-RL

Skat-RL is a reinforcement-learning playground for the card game Skat. The project contains a Skat engine with card/rule utilities and a random and a heuristic player. Currently, a Maskable PPO RL agent from Stable-Baselines3 is implemented, that can train and play against the random and heuristic players. 


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


## Outlook

This is an experimental project. Things that might be implemented in the future:

- Batched  and parallelized C++ Skat engine for fast training data generation
- Self-play of RL agents
- Transformer architecture
