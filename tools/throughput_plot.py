#!/usr/bin/env python3
"""Plot throughput comparison between experiment runs.

Generates two plots:
1. Cumulative programs evaluated over wall time
2. Best fitness over wall time

Usage:
    PYTHONPATH=. python tools/throughput_plot.py --experiment hover/steady-state-v2
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import redis

from tools.experiment.manifest import load_manifest


def get_time_series(r: redis.Redis, prefix: str, metric: str) -> list[tuple[float, float]]:
    """Extract (timestamp, value) pairs from Redis metrics history."""
    key = f"{prefix}:metrics:history:{metric}"
    if r.type(key) != b"list":
        return []
    raw = r.lrange(key, 0, -1)
    points = []
    for item in raw:
        d = json.loads(item)
        if d.get("v") is not None:
            points.append((d["t"], d["v"]))
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()

    m = load_manifest(args.experiment)
    out_dir = Path(args.output or f"experiments/{args.experiment}/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect data per run
    run_data = {}
    for run in m.runs:
        r = redis.Redis(db=run.db)
        completed = get_time_series(r, run.prefix, "dag_runner:dag_runs_completed")
        fitness = get_time_series(r, run.prefix, "program_metrics:valid_frontier_fitness")
        mutations = get_time_series(r, run.prefix, "evolution_engine:mutations_created")
        if completed:
            t0 = completed[0][0]
            run_data[run.label] = {
                "condition": run.condition,
                "completed": [(t - t0, v) for t, v in completed],
                "fitness": [(t - t0, v) for t, v in fitness],
                "mutations": [(t - t0, v) for t, v in mutations],
            }

    if not run_data:
        print("No data yet")
        return

    # Color scheme
    colors = {
        "control": {"V1": "#1f77b4", "V2": "#aec7e8"},
        "treatment": {"V3": "#d62728", "V4": "#ff9896"},
    }

    # Plot 1: Cumulative programs evaluated over wall time
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    for label, data in sorted(run_data.items()):
        cond = data["condition"]
        c = colors.get(cond, {}).get(label, "gray")
        times = [t / 3600 for t, _ in data["completed"]]
        cumulative = []
        total = 0
        for _, v in data["completed"]:
            total += (v if v else 0)
            cumulative.append(total)
        if times:
            ax.plot(times, cumulative, label=f"{label} ({cond})", color=c, linewidth=2)
    ax.set_xlabel("Wall time (hours)")
    ax.set_ylabel("Cumulative programs evaluated")
    ax.set_title("Throughput: Programs evaluated over time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Best fitness over wall time
    ax = axes[1]
    for label, data in sorted(run_data.items()):
        cond = data["condition"]
        c = colors.get(cond, {}).get(label, "gray")
        if data["fitness"]:
            times = [t / 3600 for t, _ in data["fitness"]]
            fits = [v * 100 for _, v in data["fitness"]]
            # Running max for frontier
            running_max = []
            best = 0
            for f in fits:
                best = max(best, f)
                running_max.append(best)
            ax.plot(times, running_max, label=f"{label} ({cond})", color=c, linewidth=2)
    ax.set_xlabel("Wall time (hours)")
    ax.set_ylabel("Best fitness (%)")
    ax.set_title("Fitness over wall time (LOCF)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "throughput_comparison.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()

    # Plot 3: Mutations created over time
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, data in sorted(run_data.items()):
        cond = data["condition"]
        c = colors.get(cond, {}).get(label, "gray")
        if data["mutations"]:
            times = [t / 3600 for t, _ in data["mutations"]]
            cumulative = []
            total = 0
            for _, v in data["mutations"]:
                total += (v if v else 0)
                cumulative.append(total)
            ax.plot(times, cumulative, label=f"{label} ({cond})", color=c, linewidth=2)
    ax.set_xlabel("Wall time (hours)")
    ax.set_ylabel("Cumulative mutations created")
    ax.set_title("Mutation rate: Steady-state vs Generational")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / "mutation_rate.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    main()
