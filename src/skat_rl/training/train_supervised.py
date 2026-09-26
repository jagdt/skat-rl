"""Pretrain policy and value heads from replayed moves and recorded outcomes."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from skat_rl.agents.ppo_agent import PPOAgent, PPOConfig


class SupervisedBatches(IterableDataset):
    """Keep only one decompressed shard per worker in memory."""

    def __init__(self, directory, split, batch_size, shuffle=False, seed=42):
        super().__init__()
        self.directory = Path(directory)
        with open(self.directory / "manifest.json", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if (manifest["format_version"] != 2 or manifest["observation_dim"] != 1149
                or manifest["action_dim"] != 32):
            raise ValueError("Unsupported prepared dataset format.")
        self.files = [shard["file"] for shard in manifest["splits"][split]]
        if not self.files:
            raise ValueError(f"No {split} shards in dataset.")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        worker = get_worker_info()
        worker_id, workers = (0, 1) if worker is None else (worker.id, worker.num_workers)
        rng = np.random.default_rng([self.seed, self.epoch])
        files = list(self.files)
        if self.shuffle:
            rng.shuffle(files)
        for filename in files[worker_id::workers]:
            with np.load(self.directory / filename, allow_pickle=False) as shard:
                keys = ["observations", "action_masks", "actions", "belief_targets", "terminal_rewards", "remaining_decisions"]
                arrays = {key: shard[key] for key in keys}
            indices = np.arange(len(arrays["actions"]))
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch_indices = indices[start:start + self.batch_size]
                yield {key: torch.from_numpy(value[batch_indices]) for key, value in arrays.items()}


def run_epoch(agent, loader, training):
    agent.model.train(training)
    totals = dict(examples=0, loss=0.0, policy_loss=0.0, value_loss=0.0, correct=0,
                  belief_loss=0.0, belief_correct=0, hidden_cards=0)
    with torch.set_grad_enabled(training):
        for batch_index, batch in enumerate(loader, 1):
            observations = batch["observations"].to(agent.device, dtype=torch.float32)
            masks = batch["action_masks"].to(agent.device, dtype=torch.bool)
            actions = batch["actions"].to(agent.device, dtype=torch.long)
            if not masks.gather(1, actions[:, None]).all():
                raise ValueError("Dataset contains an illegal target action.")
            logits, values, beliefs = agent.model.outputs(observations, include_belief=agent.config.use_belief)
            logits = logits.masked_fill(~masks, float("-inf"))
            policy_loss = F.cross_entropy(logits, actions)
            loss = policy_loss
            value_loss = logits.new_zeros(())
            if agent.config.value_coef > 0:
                if "terminal_rewards" not in batch or "remaining_decisions" not in batch:
                    raise ValueError("Value training requires outcome labels from prepare_supervised.")
                terminal_rewards = batch["terminal_rewards"].to(agent.device, dtype=torch.float32)
                remaining = batch["remaining_decisions"].to(agent.device, dtype=torch.float32)
                # Include future forced moves, even when they were omitted from the shards.
                targets = terminal_rewards * agent.config.gamma ** remaining
                value_loss = 0.5 * F.mse_loss(values, targets)
                loss = loss + agent.config.value_coef * value_loss
            belief_loss = logits.new_zeros(())
            hidden_count = 0
            if beliefs is not None:
                targets = batch["belief_targets"].to(agent.device, dtype=torch.long)
                hidden = targets >= 0
                hidden_count = int(hidden.sum())
                if hidden_count:
                    belief_loss = F.cross_entropy(beliefs[hidden], targets[hidden])
                    loss = loss + agent.config.belief_coef * belief_loss
                    totals["belief_correct"] += int((beliefs.argmax(-1)[hidden] == targets[hidden]).sum())
            if not torch.isfinite(loss):
                raise ValueError("Non-finite supervised training loss.")
            if training:
                agent.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.model.parameters(), agent.config.max_grad_norm)
                agent.optimizer.step()
            size = len(actions)
            totals["examples"] += size
            totals["loss"] += float(loss.detach()) * size
            totals["policy_loss"] += float(policy_loss.detach()) * size
            totals["value_loss"] += float(value_loss.detach()) * size
            totals["correct"] += int((logits.argmax(-1) == actions).sum())
            totals["belief_loss"] += float(belief_loss.detach()) * hidden_count
            totals["hidden_cards"] += hidden_count
            if training and batch_index % 100 == 0:
                print(f"batches={batch_index} examples={totals['examples']} "
                      f"policy_loss={totals['policy_loss'] / totals['examples']:.4f} "
                      f"value_loss={totals['value_loss'] / totals['examples']:.4f}", flush=True)
    if not totals["examples"]:
        raise ValueError("Dataset yielded no examples.")
    return {
        "examples": totals["examples"],
        "loss": totals["loss"] / totals["examples"],
        "policy_loss": totals["policy_loss"] / totals["examples"],
        "value_loss": totals["value_loss"] / totals["examples"],
        "accuracy": totals["correct"] / totals["examples"],
        "belief_loss": totals["belief_loss"] / max(totals["hidden_cards"], 1),
        "belief_accuracy": totals["belief_correct"] / max(totals["hidden_cards"], 1),
    }


def save_pretrained(agent, path, epoch):
    # PPO starts with a fresh optimizer, even when both heads are pretrained.
    torch.save({"config": asdict(agent.config), "model_state_dict": agent.model.state_dict(),
                "supervised_epoch": epoch, "critic_pretrained": agent.config.value_coef > 0}, path)


def train(args):
    if min(args.epochs, args.batch_size, args.patience, args.torch_threads) < 1 or args.workers < 0:
        raise ValueError("Epochs, batch-size, patience and torch-threads must be positive; workers >= 0.")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive.")
    if not np.isfinite(args.value_coef) or args.value_coef < 0:
        raise ValueError("value-coef must be finite and nonnegative.")
    if not np.isfinite(args.gamma) or not 0 <= args.gamma <= 1:
        raise ValueError("gamma must be between 0 and 1.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    config = PPOConfig(
        observation_dim=1149, action_dim=32, architecture="transformer",
        transformer_dim=args.transformer_dim, transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads, transformer_ff_dim=args.transformer_ff_dim,
        transformer_dropout=args.transformer_dropout, learning_rate=args.learning_rate,
        use_belief=args.belief, belief_coef=args.belief_coef,
        value_coef=args.value_coef, gamma=args.gamma,
    )
    agent = PPOAgent(config, device=args.device)
    datasets = {split: SupervisedBatches(args.dataset, split, args.batch_size,
                                       shuffle=split == "train", seed=args.seed)
                for split in ("train", "validation")}
    loaders = {split: DataLoader(dataset, batch_size=None, num_workers=args.workers,
                                pin_memory=agent.device.type == "cuda")
               for split, dataset in datasets.items()}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    with open(output / "config.json", "w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "model": asdict(config)}, handle, indent=2)
    best_loss = float("inf")
    stale_epochs = 0
    with open(output / "metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = None
        for epoch in range(1, args.epochs + 1):
            datasets["train"].epoch = epoch
            train_metrics = run_epoch(agent, loaders["train"], training=True)
            validation_metrics = run_epoch(agent, loaders["validation"], training=False)
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
                   **{f"validation_{k}": v for k, v in validation_metrics.items()}}
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            save_pretrained(agent, output / "last.pt", epoch)
            if validation_metrics["loss"] < best_loss:
                best_loss = validation_metrics["loss"]
                stale_epochs = 0
                save_pretrained(agent, output / "best.pt", epoch)
            else:
                stale_epochs += 1
            print(f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
                  f"validation_loss={validation_metrics['loss']:.4f} "
                  f"validation_policy_loss={validation_metrics['policy_loss']:.4f} "
                  f"validation_accuracy={validation_metrics['accuracy']:.3f} "
                  f"validation_value_loss={validation_metrics['value_loss']:.4f}", flush=True)
            if stale_epochs >= args.patience:
                print("Stopping after validation loss stopped improving.", flush=True)
                break
    print(f"Best pretrained checkpoint: {output / 'best.pt'}")
    return output / "best.pt"


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Directory produced by prepare_supervised.")
    parser.add_argument("--output-dir", required=True, help="New directory for checkpoints and metrics.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--transformer-dim", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-ff-dim", type=int, default=1024)
    parser.add_argument("--transformer-dropout", type=float, default=0.0)
    parser.add_argument("--belief", action="store_true")
    parser.add_argument("--belief-coef", type=float, default=0.05)
    parser.add_argument("--value-coef", type=float, default=0.5,
                        help="Weight for outcome regression; 0 disables value pretraining.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount per player decision, matching PPO.")
    return parser.parse_args()


def main():
    train(_parse_args())


if __name__ == "__main__":
    main()
