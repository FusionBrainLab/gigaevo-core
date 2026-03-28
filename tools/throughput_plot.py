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

    # Downsample: keep 1 point per minute to avoid huge arrays
    def downsample(series, interval=60):
        if not series:
            return series
        result = [series[0]]
        for t, v in series[1:]:
            if t - result[-1][0] >= interval:
                result.append((t, v))
        if series[-1] != result[-1]:
            result.append(series[-1])
        return result

    # Plot 1: Programs processed + mutations created over wall time
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    for label, data in sorted(run_data.items()):
        cond = data["condition"]
        c = colors.get(cond, {}).get(label, "gray")
        # mutations_created is a running total (not incremental)
        pts = downsample(data["mutations"])
        if pts:
            times = [t / 3600 for t, _ in pts]
            values = [v for _, v in pts]
            ax.plot(times, values, label=f"{label} ({cond})", color=c, linewidth=2)
    ax.set_xlabel("Wall time (hours)")
    ax.set_ylabel("Mutations created (total)")
    ax.set_title("Mutation rate: Steady-state vs Generational")
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
    ax.set_title("Best fitness over wall time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "throughput_comparison.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()

    # Plot 3: Programs processed over time
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, data in sorted(run_data.items()):
        cond = data["condition"]
        c = colors.get(cond, {}).get(label, "gray")
        # programs_processed is also a running total
        pts = downsample(data["completed"])
        if pts:
            times = [t / 3600 for t, _ in pts]
            values = [v for _, v in pts]
            ax.plot(times, values, label=f"{label} ({cond})", color=c, linewidth=2)
    ax.set_xlabel("Wall time (hours)")
    ax.set_ylabel("Programs evaluated (total)")
    ax.set_title("Throughput: Programs evaluated over time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / "mutation_rate.png"  # keep same filename for watchdog
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    main()
