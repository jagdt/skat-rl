import argparse
import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPISODE_PLOT_NAME = "episodes.png"
METRICS_PLOT_NAME = "metrics.png"


def main():
    args = _parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    episode_plot = plot_episode_logs(
        run_dir,
        rolling_window=args.rolling_window,
        output_path=args.episode_output,
    )
    metrics_plot = plot_metrics(
        run_dir,
        output_path=args.metrics_output,
    )

    if episode_plot is not None:
        print(f"Saved episode plot to {episode_plot}")
    if metrics_plot is not None:
        print(f"Saved metrics plot to {metrics_plot}")
    if episode_plot is None and metrics_plot is None:
        print(f"No plottable logs found in {run_dir}")


def plot_episode_logs(run_dir, rolling_window=100, output_path=None):
    episodes_path = Path(run_dir) / "episodes.csv"
    if not episodes_path.exists():
        return None

    rows = _read_csv(episodes_path)
    episodes = _series(rows, "episode", default_index=True)
    returns = _series(rows, "return")

    if not episodes or not returns:
        return None

    output_path = _output_path(run_dir, output_path, EPISODE_PLOT_NAME)
    rolling_returns = rolling_average(returns, rolling_window)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(episodes, returns, linewidth=1.0, alpha=0.35, label="Episode return")
    ax.plot(
        episodes,
        rolling_returns,
        linewidth=2.0,
        label=f"Rolling mean ({rolling_window} episodes)",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("PyTorch PPO Episode Returns")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_metrics(run_dir, output_path=None):
    metrics_path = Path(run_dir) / "metrics.csv"
    if not metrics_path.exists():
        return None

    rows = _read_csv(metrics_path)
    if not rows:
        return None

    x_values = _series(rows, "total_timesteps", default_index=True)
    metric_names = [
        "mean_episode_return",
        "mean_episode_length",
        "loss",
        "policy_loss",
        "value_loss",
        "belief_loss",
        "belief_accuracy",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
    ]
    metric_values = {
        name: _series(rows, name)
        for name in metric_names
    }
    metric_values = {
        name: values
        for name, values in metric_values.items()
        if values
    }
    if not metric_values:
        return None

    output_path = _output_path(run_dir, output_path, METRICS_PLOT_NAME)
    n_cols = 2
    n_rows = math.ceil(len(metric_values) / n_cols)
    fig, _ = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows), squeeze=False)

    for ax, (metric_name, values) in zip(fig.axes, metric_values.items()):
        ax.plot(x_values[:len(values)], values, linewidth=1.8)
        ax.set_title(metric_name)
        ax.set_xlabel("Timesteps")
        ax.grid(True, alpha=0.3)

    for ax in fig.axes[len(metric_values):]:
        ax.remove()

    fig.suptitle("PyTorch PPO Metrics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def rolling_average(values, window):
    if window < 1:
        raise ValueError("rolling window must be at least 1")

    averages = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
            divisor = window
        else:
            divisor = index + 1
        averages.append(total / divisor)

    return averages


def _parse_args():
    parser = argparse.ArgumentParser(description="Plot logs from a native PyTorch PPO run.")
    parser.add_argument("run_dir", help="Training run directory containing metrics.csv and/or episodes.csv.")
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=100,
        help="Episode rolling-average window.",
    )
    parser.add_argument(
        "--episode-output",
        help=f"Episode plot path. Defaults to <run-dir>/{EPISODE_PLOT_NAME}.",
    )
    parser.add_argument(
        "--metrics-output",
        help=f"Metrics plot path. Defaults to <run-dir>/{METRICS_PLOT_NAME}.",
    )
    return parser.parse_args()


def _read_csv(path):
    with open(path, encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _series(rows, key, default_index=False):
    values = []
    for index, row in enumerate(rows, start=1):
        value = row.get(key)
        if value in (None, ""):
            if default_index:
                values.append(float(index))
            continue

        try:
            values.append(float(value))
        except ValueError:
            if default_index:
                values.append(float(index))

    return values


def _output_path(run_dir, output_path, default_name):
    if output_path is None:
        return Path(run_dir) / default_name
    return Path(output_path)


if __name__ == "__main__":
    main()
