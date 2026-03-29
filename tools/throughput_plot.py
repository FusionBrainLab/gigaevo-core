#!/usr/bin/env python3
"""Plot throughput comparison dashboard for experiment runs.

Uses ONLY event-driven metrics (one entry per program evaluation) which have
full experiment history. Avoids polled metrics (mutations_created,
dag_runs_completed) which are capped at 10k entries and only show recent data.

Panels (2x3):
  (0,0) Programs evaluated vs wall time (cumulative)
  (0,1) Throughput rate (programs/hour, rolling 1h window)
  (0,2) Best fitness vs time (frontier + per-program scatter)
  (1,0) Fitness distribution (box plots, all valid programs)
  (1,1) Invalidity rate vs time
  (1,2) Sample efficiency (fitness vs cumulative evaluations)

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

# ── Color scheme (colorblind-friendly, high contrast) ──

COLORS = {
    "V1": "#2166ac",  # Dark blue (control)
    "V2": "#67a9cf",  # Light blue (control)
    "V3": "#b2182b",  # Dark red (treatment)
    "V4": "#ef8a62",  # Orange-red (treatment)
}

LINESTYLES = {
    "V1": "-",   # Solid (replicate 1)
    "V2": "--",  # Dashed (replicate 2)
    "V3": "-",
    "V4": "--",
}


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
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    m = load_manifest(args.experiment)
    out_dir = Path(args.output or f"experiments/{args.experiment}/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load event-driven metrics from Redis ──
    run_data = {}
    for run in m.runs:
        r = redis.Redis(db=run.db)
        total_count = get_time_series(r, run.prefix, "program_metrics:programs_total_count")
        valid_count = get_time_series(r, run.prefix, "program_metrics:programs_valid_count")
        frontier = get_time_series(r, run.prefix, "program_metrics:valid_frontier_fitness")
        per_program = get_time_series(r, run.prefix, "program_metrics:valid_iter_fitness_mean")

        if not total_count:
            continue

        # Shared t0: earliest timestamp across event-driven metrics
        all_starts = [s[0][0] for s in [total_count, valid_count, frontier, per_program] if s]
        t0 = min(all_starts)

        def _rel(series):
            return [(t - t0, v) for t, v in series]

        run_data[run.label] = {
            "condition": run.condition,
            "total_count": _rel(total_count),     # (rel_t, cumulative_count)
            "valid_count": _rel(valid_count),
            "frontier": _rel(frontier),            # (rel_t, fitness_fraction)
            "per_program": _rel(per_program),      # (rel_t, fitness_fraction)
        }

    if not run_data:
        print("No data yet")
        return

    # ── Baseline reference ──
    baseline = None
    if hasattr(m, "baseline") and m.baseline and hasattr(m.baseline, "mean"):
        baseline = m.baseline.mean  # percentage (e.g. 81.78)

    # ── Figure setup ──
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.patch.set_facecolor("white")

    def _c(label):
        return COLORS.get(label, "gray")

    def _ls(label):
        return LINESTYLES.get(label, "-")

    # Build shared legend handles
    legend_handles = []
    legend_labels = []

    # ── Panel (0,0): Programs Evaluated vs Wall Time ──
    ax = axes[0, 0]
    for label, data in sorted(run_data.items()):
        pts = data["total_count"]
        if pts:
            times = [t / 3600 for t, _ in pts]
            vals = [v - 1 for _, v in pts]  # start at 0 (first program = 1)
            line, = ax.step(times, vals, where="post",
                            color=_c(label), linestyle=_ls(label), linewidth=2,
                            marker=".", markersize=3, markevery=max(1, len(times) // 30))
            line.set_label(f"{label} ({data['condition']})")
            legend_handles.append(line)
            legend_labels.append(f"{label} ({data['condition']})")
    ax.set_xlabel("Wall time (hours)", fontsize=11)
    ax.set_ylabel("Programs evaluated", fontsize=11)
    ax.set_title("Programs Evaluated", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # ── Panel (0,1): Throughput Rate (programs/hour, computed on regular grid) ──
    ax = axes[0, 1]
    for label, data in sorted(run_data.items()):
        pts = data["total_count"]
        if len(pts) < 3:
            continue
        times = np.array([t for t, _ in pts])
        vals = np.array([v for _, v in pts])
        # Compute rate on a regular 1-hour grid
        t_max = times[-1]
        grid = np.arange(3600, t_max + 1, 3600)  # every hour starting at hour 1
        if len(grid) < 1:
            continue
        # Interpolate total_count at grid points and 1h earlier
        v_now = np.interp(grid, times, vals)
        v_prev = np.interp(grid - 3600, times, vals)
        rate = v_now - v_prev  # programs in that hour
        ax.plot(grid / 3600, rate,
                color=_c(label), linestyle=_ls(label), linewidth=2,
                marker="o", markersize=4)
    ax.set_xlabel("Wall time (hours)", fontsize=11)
    ax.set_ylabel("Programs / hour", fontsize=11)
    ax.set_title("Throughput Rate (1h rolling)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # ── Panel (0,2): Best Fitness vs Time + per-program scatter ──
    ax = axes[0, 2]
    all_fitness_vals = []
    for label, data in sorted(run_data.items()):
        # Per-program scatter behind frontier
        pp = data["per_program"]
        if pp:
            pp_times = [t / 3600 for t, _ in pp]
            pp_vals = [v * 100 for _, v in pp]
            all_fitness_vals.extend(pp_vals)
            ax.scatter(pp_times, pp_vals, color=_c(label), alpha=0.15, s=15, edgecolors="none")
        # Frontier step plot
        pts = data["frontier"]
        if pts:
            times = [t / 3600 for t, _ in pts]
            vals = [v * 100 for _, v in pts]
            all_fitness_vals.extend(vals)
            ax.step(times, vals, where="post",
                    color=_c(label), linestyle=_ls(label), linewidth=2.5)
            ax.annotate(
                f"{label}: {vals[-1]:.1f}%", xy=(times[-1], vals[-1]),
                xytext=(6, 0), textcoords="offset points",
                fontsize=8, fontweight="bold", color=_c(label), va="center",
            )
    if baseline:
        ax.axhline(y=baseline, color="#555555", linestyle=":", linewidth=1.5, alpha=0.7,
                    label=f"baseline ({baseline:.1f}%)")
    if all_fitness_vals:
        y_min = min(all_fitness_vals)
        y_max = max(all_fitness_vals)
        pad = max((y_max - y_min) * 0.15, 1.0)
        ax.set_ylim(y_min - pad, y_max + pad)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlabel("Wall time (hours)", fontsize=11)
    ax.set_ylabel("Fitness (%)", fontsize=11)
    ax.set_title("Best Fitness vs Time", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # ── Panel (1,0): Fitness Distribution (all valid programs, no v>0 filter) ──
    ax = axes[1, 0]
    box_data = []
    box_labels = []
    box_colors = []
    for label, data in sorted(run_data.items()):
        fits = [v * 100 for _, v in data["per_program"]]
        n_total = int(data["total_count"][-1][1]) if data["total_count"] else 0
        n_valid = len(fits)
        if fits:
            box_data.append(fits)
            box_labels.append(f"{label}\n({data['condition'][:4]})\n{n_valid}/{n_total}")
            box_colors.append(_c(label))
    if box_data:
        bp = ax.boxplot(
            box_data, patch_artist=True, showfliers=True,
            flierprops=dict(marker=".", markersize=3, alpha=0.3),
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=1),
        )
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.set_xticklabels(box_labels, fontsize=9)
        if baseline:
            ax.axhline(y=baseline, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.set_ylabel("Fitness (%)", fontsize=11)
    ax.set_title("Fitness Distribution (valid programs)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel (1,1): Invalidity Rate vs Time ──
    ax = axes[1, 1]
    for label, data in sorted(run_data.items()):
        total_pts = data["total_count"]
        valid_pts = data["valid_count"]
        if len(total_pts) >= 2 and len(valid_pts) >= 2:
            # total_count and valid_count share timestamps — pair directly
            total_v = np.array([v for _, v in total_pts])
            valid_v = np.array([v for _, v in valid_pts])
            total_t = np.array([t for t, _ in total_pts])
            # Use min of lengths in case they differ slightly
            n = min(len(total_v), len(valid_v))
            mask = total_v[:n] > 0
            inv_rate = np.clip((1 - valid_v[:n][mask] / total_v[:n][mask]) * 100, 0, 100)
            inv_times = total_t[:n][mask] / 3600
            ax.plot(inv_times, inv_rate,
                    color=_c(label), linestyle=_ls(label), linewidth=2)
    ax.set_xlabel("Wall time (hours)", fontsize=11)
    ax.set_ylabel("Invalid programs (%)", fontsize=11)
    ax.set_title("Invalidity Rate", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # ── Panel (1,2): Sample Efficiency — Fitness vs Cumulative Evaluations ──
    ax = axes[1, 2]
    for label, data in sorted(run_data.items()):
        total_pts = data["total_count"]
        if not total_pts:
            continue
        tc_times = np.array([t for t, _ in total_pts])
        tc_vals = np.array([v for _, v in total_pts])

        # Per-program scatter: x = total_count at that timestamp
        pp = data["per_program"]
        if pp:
            pp_times = np.array([t for t, _ in pp])
            pp_fits = np.array([v * 100 for _, v in pp])
            pp_evals = np.interp(pp_times, tc_times, tc_vals)
            ax.scatter(pp_evals, pp_fits,
                       color=_c(label), alpha=0.3, s=20, edgecolors="none")

        # Frontier step: x = total_count at frontier timestamp
        fr = data["frontier"]
        if fr:
            fr_times = np.array([t for t, _ in fr])
            fr_fits = np.array([v * 100 for _, v in fr])
            fr_evals = np.interp(fr_times, tc_times, tc_vals)
            ax.step(fr_evals, fr_fits, where="post",
                    color=_c(label), linestyle=_ls(label), linewidth=2, alpha=0.9)
    if baseline:
        ax.axhline(y=baseline, color="#555555", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Cumulative programs evaluated", fontsize=11)
    ax.set_ylabel("Fitness (%)", fontsize=11)
    ax.set_title("Sample Efficiency", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # ── Shared legend at top ──
    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc="upper center", ncol=len(legend_handles),
            fontsize=12, frameon=True, framealpha=0.9,
            bbox_to_anchor=(0.5, 0.99),
        )

    plt.suptitle(
        f"Experiment: {args.experiment}", fontsize=16, fontweight="bold", y=1.01
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "throughput_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    main()
