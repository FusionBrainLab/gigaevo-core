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

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))

    # ── Plot 1: Mutations created over wall time ──
    ax = axes[0, 0]
    for label, data in sorted(run_data.items()):
        c = colors.get(data["condition"], {}).get(label, "gray")
        pts = downsample(data["mutations"])
        if pts:
            ax.plot([t / 3600 for t, _ in pts], [v for _, v in pts],
                    label=f"{label} ({data['condition']})", color=c, linewidth=2)
    ax.set_xlabel("Wall time (hours)")
    ax.set_ylabel("Total mutations created")
    ax.set_title("Mutation rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Programs evaluated over wall time ──
    ax = axes[0, 1]
    for label, data in sorted(run_data.items()):
        c = colors.get(data["condition"], {}).get(label, "gray")
        pts = downsample(data["completed"])
        if pts:
            ax.plot([t / 3600 for t, _ in pts], [v for _, v in pts],
                    label=f"{label} ({data['condition']})", color=c, linewidth=2)
    ax.set_xlabel("Wall time (hours)")
    ax.set_ylabel("Total programs evaluated")
    ax.set_title("Evaluation throughput")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 3: Fitness distribution (violin) by condition ──
    ax = axes[0, 2]
    ctrl_fits, treat_fits = [], []
    for label, data in run_data.items():
        fits = [v * 100 for _, v in data["per_program"] if v > 0]
        if data["condition"] == "control":
            ctrl_fits.extend(fits)
        else:
            treat_fits.extend(fits)
    violin_data = []
    violin_labels = []
    if ctrl_fits:
        violin_data.append(ctrl_fits)
        violin_labels.append(f"Control\n(n={len(ctrl_fits)})")
    if treat_fits:
        violin_data.append(treat_fits)
        violin_labels.append(f"Treatment\n(n={len(treat_fits)})")
    if violin_data:
        parts = ax.violinplot(violin_data, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts.get("bodies", [])):
            pc.set_facecolor(["#1f77b4", "#d62728"][i])
            pc.set_alpha(0.4)
        ax.set_xticks(range(1, len(violin_labels) + 1))
        ax.set_xticklabels(violin_labels)
        # Add individual run box plots overlaid
        for i, (label, data) in enumerate(sorted(run_data.items())):
            fits = [v * 100 for _, v in data["per_program"] if v > 0]
            if not fits:
                continue
            cond_idx = 1 if data["condition"] == "control" else 2
            c = colors.get(data["condition"], {}).get(label, "gray")
            offset = -0.15 if label in ("V1", "V3") else 0.15
            bp = ax.boxplot(
                [fits], positions=[cond_idx + offset], widths=0.12,
                patch_artist=True, showfliers=False,
            )
            bp["boxes"][0].set_facecolor(c)
            bp["boxes"][0].set_alpha(0.6)
            bp["medians"][0].set_color("black")
    ax.set_ylabel("Fitness (%)")
    ax.set_title("Fitness distribution by condition")
    ax.grid(True, alpha=0.3, axis="y")

    # ── Plot 4: Fitness scatter + frontier over wall time ──
    ax = axes[1, 0]
    for label, data in sorted(run_data.items()):
        c = colors.get(data["condition"], {}).get(label, "gray")
        pts = data["per_program"]
        if pts:
            ax.scatter([t / 3600 for t, _ in pts], [v * 100 for _, v in pts],
                       color=c, alpha=0.15, s=10, edgecolors="none")
        if data["frontier"]:
            ft = [t / 3600 for t, _ in data["frontier"]]
            ff = [v * 100 for _, v in data["frontier"]]
            best = []
            b = 0
            for f in ff:
                b = max(b, f)
                best.append(b)
            ax.plot(ft, best, color=c, linewidth=2.5, label=f"{label} ({data['condition']})")
    ax.set_xlabel("Wall time (hours)")
    ax.set_ylabel("Fitness (%)")
    ax.set_title("Fitness: all programs (scatter) + frontier")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 5: Fitness frontier vs generation/epoch ──
    ax = axes[1, 1]
    for label, data in sorted(run_data.items()):
        c = colors.get(data["condition"], {}).get(label, "gray")
        if data["frontier"]:
            gens = list(range(len(data["frontier"])))
            ff = [v * 100 for _, v in data["frontier"]]
            best = []
            b = 0
            for f in ff:
                b = max(b, f)
                best.append(b)
            ax.plot(gens, best, color=c, linewidth=2.5, marker="o", markersize=5,
                    label=f"{label} ({data['condition']})")
    ax.set_xlabel("Generation / Epoch")
    ax.set_ylabel("Best fitness (%)")
    ax.set_title("Fitness frontier vs generation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 6: Eval duration over time ──
    ax = axes[1, 2]
    for label, data in sorted(run_data.items()):
        c = colors.get(data["condition"], {}).get(label, "gray")
        # Use per_program timestamps to compute inter-eval intervals
        pts = data["per_program"]
        if len(pts) > 1:
            times = [t / 3600 for t, _ in pts]
            fits = [v * 100 for _, v in pts]
            # Color by fitness: valid (>0) vs invalid (0)
            valid_t = [t for t, f in zip(times, fits) if f > 0]
            valid_f = [f for f in fits if f > 0]
            invalid_t = [t for t, f in zip(times, fits) if f == 0]
            if valid_t:
                ax.scatter(valid_t, valid_f, color=c, alpha=0.3, s=15,
                           edgecolors="none", label=f"{label} valid")
            if invalid_t:
                ax.scatter(invalid_t, [0] * len(invalid_t), color=c, alpha=0.5,
                           s=15, marker="x")
    ax.set_xlabel("Wall time (hours)")
    ax.set_ylabel("Fitness (%) — 0 = invalid")
    ax.set_title("Valid vs invalid programs over time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Experiment: {args.experiment}", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = out_dir / "throughput_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    main()
