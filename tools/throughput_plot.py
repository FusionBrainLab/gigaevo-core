#!/usr/bin/env python3
"""Plot throughput comparison between experiment runs.

Generates two plots:
1. Mutations created + programs evaluated over wall time (with rolling bands)
2. Best fitness over wall time (with per-sample scatter)

Usage:
    PYTHONPATH=. python tools/throughput_plot.py --experiment hover/steady-state-v2
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


def get_per_program_fitness(r: redis.Redis, prefix: str) -> list[tuple[float, float]]:
    """Get per-program fitness with timestamps (from valid_iter_fitness_mean)."""
    key = f"{prefix}:metrics:history:program_metrics:valid_iter_fitness_mean"
    if r.type(key) != b"list":
        return []
    raw = r.lrange(key, 0, -1)
    points = []
    for item in raw:
        d = json.loads(item)
        if d.get("v") is not None:
            points.append((d["t"], d["v"]))
    return points


def downsample(series, interval=60):
    """Keep 1 point per interval seconds."""
    if not series:
        return series
    result = [series[0]]
    for t, v in series[1:]:
        if t - result[-1][0] >= interval:
            result.append((t, v))
    if series[-1] != result[-1]:
        result.append(series[-1])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    m = load_manifest(args.experiment)
    out_dir = Path(args.output or f"experiments/{args.experiment}/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    run_data = {}
    for run in m.runs:
        r = redis.Redis(db=run.db)
        completed = get_time_series(r, run.prefix, "dag_runner:dag_runs_completed")
        mutations = get_time_series(r, run.prefix, "evolution_engine:mutations_created")
        frontier = get_time_series(r, run.prefix, "program_metrics:valid_frontier_fitness")
        per_program = get_per_program_fitness(r, run.prefix)
        if completed:
            t0 = completed[0][0]
            run_data[run.label] = {
                "condition": run.condition,
                "completed": [(t - t0, v) for t, v in completed],
                "mutations": [(t - t0, v) for t, v in mutations],
                "frontier": [(t - t0, v) for t, v in frontier],
                "per_program": [(t - t0, v) for t, v in per_program],
            }

    if not run_data:
        print("No data yet")
        return

    colors = {
        "control": {"V1": "#1f77b4", "V2": "#6baed6"},
        "treatment": {"V3": "#d62728", "V4": "#fc9272"},
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── Plot 1: Mutations created over wall time ──
    ax = axes[0][0]
    for label, data in sorted(run_data.items()):
        cond = data["condition"]
        c = colors.get(cond, {}).get(label, "gray")
        pts = downsample(data["mutations"])
        if pts:
            times = [t / 3600 for t, _ in pts]
            values = [v for _, v in pts]
            ax.plot(times, values, label=f"{label} ({cond})", color=c, linewidth=2)
    ax.set_xlabel("Wall time (hours)")
    ax.set_ylabel("Total mutations created")
    ax.set_title("Mutation rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Programs evaluated over wall time ──
    ax = axes[0][1]
    for label, data in sorted(run_data.items()):
        cond = data["condition"]
        c = colors.get(cond, {}).get(label, "gray")
        pts = downsample(data["completed"])
        if pts:
            times = [t / 3600 for t, _ in pts]
            values = [v for _, v in pts]
            ax.plot(times, values, label=f"{label} ({cond})", color=c, linewidth=2)
    ax.set_xlabel("Wall time (hours)")
    ax.set_ylabel("Total programs evaluated")
    ax.set_title("Evaluation throughput")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 3: Fitness frontier over wall time ──
    ax = axes[1][0]
    for label, data in sorted(run_data.items()):
        cond = data["condition"]
        c = colors.get(cond, {}).get(label, "gray")
        if data["frontier"]:
            times = [t / 3600 for t, _ in data["frontier"]]
            fits = [v * 100 for _, v in data["frontier"]]
            best_so_far = []
            b = 0
            for f in fits:
                b = max(b, f)
                best_so_far.append(b)
            ax.plot(times, best_so_far, label=f"{label} ({cond})", color=c, linewidth=2.5)
    ax.set_xlabel("Wall time (hours)")
    ax.set_ylabel("Best fitness (%)")
    ax.set_title("Fitness frontier over wall time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 4: Per-program fitness scatter + rolling mean ──
    ax = axes[1][1]
    for label, data in sorted(run_data.items()):
        cond = data["condition"]
        c = colors.get(cond, {}).get(label, "gray")
        pts = data["per_program"]
        if len(pts) > 2:
            times = np.array([t / 3600 for t, _ in pts])
            fits = np.array([v * 100 for _, v in pts])
            # Scatter (small, transparent)
            ax.scatter(times, fits, color=c, alpha=0.15, s=8, edgecolors="none")
            # Rolling mean (window = 10% of points, min 3)
            window = max(3, len(fits) // 10)
            if len(fits) >= window:
                rolling_mean = np.convolve(fits, np.ones(window) / window, mode="valid")
                rolling_t = times[window - 1:]
                ax.plot(rolling_t, rolling_mean, color=c, linewidth=2,
                        label=f"{label} ({cond})")
    ax.set_xlabel("Wall time (hours)")
    ax.set_ylabel("Per-program fitness (%)")
    ax.set_title("Per-program fitness (scatter + rolling mean)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Experiment: {args.experiment}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = out_dir / "throughput_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    main()
