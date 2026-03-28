#!/usr/bin/env python3
"""Plot throughput comparison between experiment runs.

Generates a 2x3 dashboard:
  (0,0) Mutations created vs wall time
  (0,1) Programs evaluated vs wall time
  (0,2) Best fitness (frontier) vs wall time  <-- key comparison
  (1,0) Fitness distribution (grouped box plots)
  (1,1) Invalidity rate vs wall time
  (1,2) Efficiency: per-program fitness vs cumulative eval count

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

CONDITION_COLORS = {
    "control": "#2166ac",
    "treatment": "#b2182b",
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


def _condition_mean_band(run_data, metric_key, ax, time_unit=3600, grid_step=60):
    """Draw condition-mean shaded bands behind individual run lines."""
    by_cond = {}
    for label, data in run_data.items():
        cond = data["condition"]
        by_cond.setdefault(cond, []).append(data[metric_key])

    for cond, series_list in by_cond.items():
        if len(series_list) < 2:
            continue
        # Find common time range
        t_max = min(s[-1][0] for s in series_list if s) if all(s for s in series_list) else 0
        if t_max <= 0:
            continue
        grid = np.arange(0, t_max, grid_step)
        if len(grid) < 2:
            continue
        interpolated = []
        for s in series_list:
            ts = np.array([t for t, _ in s])
            vs = np.array([v for _, v in s])
            interpolated.append(np.interp(grid, ts, vs))
        stacked = np.array(interpolated)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        color = CONDITION_COLORS.get(cond, "gray")
        ax.fill_between(
            grid / time_unit, mean - std, mean + std,
            color=color, alpha=0.08, linewidth=0,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    m = load_manifest(args.experiment)
    out_dir = Path(args.output or f"experiments/{args.experiment}/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data from Redis ──
    run_data = {}
    for run in m.runs:
        r = redis.Redis(db=run.db)
        completed = get_time_series(r, run.prefix, "dag_runner:dag_runs_completed")
        mutations = get_time_series(r, run.prefix, "evolution_engine:mutations_created")
        frontier = get_time_series(r, run.prefix, "program_metrics:valid_frontier_fitness")
        per_program = get_time_series(r, run.prefix, "program_metrics:valid_iter_fitness_mean")
        total_count = get_time_series(r, run.prefix, "program_metrics:programs_total_count")
        valid_count = get_time_series(r, run.prefix, "program_metrics:programs_valid_count")
        if completed or frontier:
            # Shared t0 from first frontier entry (earliest data point)
            all_ts = [s[0][0] for s in [frontier, completed, mutations, per_program, total_count, valid_count] if s]
            t0 = min(all_ts) if all_ts else 0

            def _rel(series):
                return [(t - t0, v) for t, v in series]

            # For cumulative counters, subtract initial value to start at 0
            def _rel_zeroed(series):
                if not series:
                    return []
                v0 = series[0][1]
                return [(t - t0, v - v0) for t, v in series]

            run_data[run.label] = {
                "condition": run.condition,
                "completed": _rel_zeroed(completed),
                "mutations": _rel_zeroed(mutations),
                "frontier": _rel(frontier),
                "per_program": _rel(per_program),
                "total_count": _rel(total_count),
                "valid_count": _rel(valid_count),
            }

    if not run_data:
        print("No data yet")
        return

    # ── Baseline reference ──
    baseline = None
    if hasattr(m, "baseline") and m.baseline and hasattr(m.baseline, "mean"):
        baseline = m.baseline.mean  # Already in percentage (e.g. 81.78)

    # ── Figure setup ──
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.patch.set_facecolor("white")

    # Shared legend handles
    legend_handles = []
    legend_labels = []

    def _color(label):
        return COLORS.get(label, "gray")

    def _ls(label):
        return LINESTYLES.get(label, "-")

    def _label_str(label, data):
        return f"{label} ({data['condition']})"

    # ── Panel (0,0): Mutations Created vs Wall Time ──
    ax = axes[0, 0]
    for label, data in sorted(run_data.items()):
        pts = downsample(data["mutations"])
        if pts:
            line, = ax.plot(
                [t / 3600 for t, _ in pts], [v for _, v in pts],
                color=_color(label), linestyle=_ls(label), linewidth=2,
            )
            if label not in [l for l in [h.get_label() for h in legend_handles]]:
                line.set_label(_label_str(label, data))
                legend_handles.append(line)
                legend_labels.append(_label_str(label, data))
    _condition_mean_band(run_data, "mutations", ax)
    ax.set_xlabel("Wall time (hours)", fontsize=11)
    ax.set_ylabel("Total mutations created", fontsize=11)
    ax.set_title("Mutation Rate", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # ── Panel (0,1): Programs Evaluated vs Wall Time ──
    ax = axes[0, 1]
    for label, data in sorted(run_data.items()):
        pts = downsample(data["completed"])
        if pts:
            ax.plot(
                [t / 3600 for t, _ in pts], [v for _, v in pts],
                color=_color(label), linestyle=_ls(label), linewidth=2,
            )
    _condition_mean_band(run_data, "completed", ax)
    ax.set_xlabel("Wall time (hours)", fontsize=11)
    ax.set_ylabel("Total programs evaluated", fontsize=11)
    ax.set_title("Evaluation Throughput", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # ── Panel (0,2): Best Fitness (Frontier) vs Wall Time ──
    # valid_frontier_fitness is already the running best — just plot it directly
    ax = axes[0, 2]
    all_frontier_vals = []
    for label, data in sorted(run_data.items()):
        pts = data["frontier"]
        if pts:
            times = [t / 3600 for t, _ in pts]
            vals = [v * 100 for _, v in pts]
            all_frontier_vals.extend(vals)
            ax.step(times, vals, where="post",
                    color=_color(label), linestyle=_ls(label), linewidth=2.5)
            # Annotate endpoint with current best
            ax.annotate(
                f"{label}: {vals[-1]:.1f}%", xy=(times[-1], vals[-1]),
                xytext=(6, 0), textcoords="offset points",
                fontsize=8, fontweight="bold", color=_color(label), va="center",
            )
    if baseline:
        ax.axhline(y=baseline, color="#555555", linestyle=":", linewidth=1.5, alpha=0.7,
                    label=f"baseline ({baseline:.1f}%)")
    if all_frontier_vals:
        y_min = min(all_frontier_vals)
        y_max = max(all_frontier_vals)
        pad = max((y_max - y_min) * 0.15, 1.0)
        ax.set_ylim(y_min - pad, y_max + pad)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlabel("Wall time (hours)", fontsize=11)
    ax.set_ylabel("Best fitness (%)", fontsize=11)
    ax.set_title("Best Fitness vs Time", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # ── Panel (1,0): Fitness Distribution (grouped box plots) ──
    ax = axes[1, 0]
    box_data = []
    box_labels = []
    box_colors = []
    for label, data in sorted(run_data.items()):
        fits = [v * 100 for _, v in data["per_program"] if v > 0]
        if fits:
            box_data.append(fits)
            box_labels.append(f"{label}\n({data['condition'][:4]})\nn={len(fits)}")
            box_colors.append(_color(label))
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
    ax.set_title("Fitness Distribution by Run", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel (1,1): Invalidity Rate vs Wall Time ──
    ax = axes[1, 1]
    for label, data in sorted(run_data.items()):
        total_pts = data["total_count"]
        valid_pts = data["valid_count"]
        if len(total_pts) >= 2 and len(valid_pts) >= 2:
            # Interpolate valid_count onto total_count's timestamps
            total_t = np.array([t for t, _ in total_pts])
            total_v = np.array([v for _, v in total_pts])
            valid_t = np.array([t for t, _ in valid_pts])
            valid_v = np.array([v for _, v in valid_pts])
            # Interpolate valid count at each total_count timestamp
            valid_at_total = np.interp(total_t, valid_t, valid_v)
            # Compute invalidity rate, clamp to [0, 100]
            mask = total_v > 0
            inv_rate = np.clip((total_v[mask] - valid_at_total[mask]) / total_v[mask] * 100, 0, 100)
            inv_times = total_t[mask] / 3600
            if len(inv_times) > 0:
                ax.plot(inv_times, inv_rate,
                        color=_color(label), linestyle=_ls(label), linewidth=2)
    ax.set_xlabel("Wall time (hours)", fontsize=11)
    ax.set_ylabel("Invalid programs (%)", fontsize=11)
    ax.set_title("Invalidity Rate", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # ── Panel (1,2): Efficiency — Frontier fitness vs cumulative eval count ──
    ax = axes[1, 2]
    for label, data in sorted(run_data.items()):
        # Per-program scatter: x = eval count at that timestamp, y = fitness
        pts = data["per_program"]
        completed_pts = data["completed"]
        if pts and completed_pts:
            # For each per_program point, find the eval count at that time
            comp_times = np.array([t for t, _ in completed_pts])
            comp_vals = np.array([v for _, v in completed_pts])
            prog_times = np.array([t for t, _ in pts])
            prog_fits = np.array([v * 100 for _, v in pts])
            # Interpolate eval count at each program's timestamp
            eval_counts = np.interp(prog_times, comp_times, comp_vals)
            ax.scatter(
                eval_counts, prog_fits,
                color=_color(label), alpha=0.3, s=20, edgecolors="none",
            )
        # Frontier line: x = eval count at frontier timestamp, y = frontier value
        frontier_pts = data["frontier"]
        if frontier_pts and completed_pts:
            comp_times = np.array([t for t, _ in completed_pts])
            comp_vals = np.array([v for _, v in completed_pts])
            fr_times = np.array([t for t, _ in frontier_pts])
            fr_fits = np.array([v * 100 for _, v in frontier_pts])
            fr_evals = np.interp(fr_times, comp_times, comp_vals)
            # Frontier is already running best — plot directly
            ax.step(fr_evals, fr_fits, where="post",
                    color=_color(label), linestyle=_ls(label),
                    linewidth=2, alpha=0.9)
    if baseline:
        ax.axhline(y=baseline, color="#555555", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Cumulative programs evaluated", fontsize=11)
    ax.set_ylabel("Fitness (%)", fontsize=11)
    ax.set_title("Efficiency: Fitness per Evaluation", fontsize=13, fontweight="bold")
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
