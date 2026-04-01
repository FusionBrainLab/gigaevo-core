#!/usr/bin/env python3
"""Pareto frontier analysis: fitness vs chain complexity.

Generates a publication-quality multi-panel figure showing the
relationship between chain structure (retrieval calls, LLM steps,
DAG complexity) and fitness across all evolved programs.

Usage:
    PYTHONPATH=. python tools/pareto_plot.py --data /tmp/hover_ssv2_programs.json
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "V1": "#2166ac",
    "V2": "#67a9cf",  # control blues
    "V3": "#b2182b",
    "V4": "#ef8a62",  # treatment reds
}
COND_COLORS = {"control": "#2166ac", "treatment": "#b2182b"}
MARKERS = {"V1": "o", "V2": "s", "V3": "^", "V4": "D"}


def pareto_front(xs, ys, maximize_y=True):
    """Compute 2D Pareto front (minimize x, maximize y by default)."""
    points = sorted(zip(xs, ys), key=lambda p: p[0])
    front_x, front_y = [], []
    best_y = -np.inf if maximize_y else np.inf
    for x, y in points:
        if (maximize_y and y > best_y) or (not maximize_y and y < best_y):
            front_x.append(x)
            front_y.append(y)
            best_y = y
    return front_x, front_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--output",
        default="experiments/hover/steady-state-v2/plots/pareto_analysis.png",
    )
    parser.add_argument("--baseline", type=float, default=81.78)
    args = parser.parse_args()

    with open(args.data) as f:
        programs = json.load(f)

    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    fig.patch.set_facecolor("white")

    # ── Panel A (0,0): Fitness vs Retrieval Calls (the key tradeoff) ──
    ax = axes[0, 0]
    for label in ["V1", "V2", "V3", "V4"]:
        progs = [p for p in programs if p["label"] == label]
        xs = [p["n_tool_steps"] for p in progs]
        ys = [p["fitness"] * 100 for p in progs]
        ax.scatter(
            xs,
            ys,
            c=COLORS[label],
            marker=MARKERS[label],
            alpha=0.4,
            s=30,
            label=f"{label} ({progs[0]['condition']})" if progs else label,
            edgecolors="none",
        )
    # Pareto front per condition
    for cond, color in COND_COLORS.items():
        progs = [p for p in programs if p["condition"] == cond]
        xs = [p["n_tool_steps"] for p in progs]
        ys = [p["fitness"] * 100 for p in progs]
        fx, fy = pareto_front(xs, ys)
        ax.step(fx, fy, where="post", color=color, linewidth=2.5, alpha=0.8)
    ax.axhline(y=args.baseline, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("Retrieval calls per sample", fontsize=11)
    ax.set_ylabel("Fitness (%)", fontsize=11)
    ax.set_title("A. Fitness vs Retrieval Budget", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.2)

    # ── Panel B (0,1): Fitness vs Total Steps ──
    ax = axes[0, 1]
    for label in ["V1", "V2", "V3", "V4"]:
        progs = [p for p in programs if p["label"] == label]
        xs = [p["n_steps"] for p in progs]
        ys = [p["fitness"] * 100 for p in progs]
        ax.scatter(
            xs,
            ys,
            c=COLORS[label],
            marker=MARKERS[label],
            alpha=0.4,
            s=30,
            edgecolors="none",
        )
    ax.axhline(y=args.baseline, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("Total chain steps", fontsize=11)
    ax.set_ylabel("Fitness (%)", fontsize=11)
    ax.set_title("B. Fitness vs Chain Length", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)

    # ── Panel C (0,2): Fitness vs DAG Complexity (total deps) ──
    ax = axes[0, 2]
    for label in ["V1", "V2", "V3", "V4"]:
        progs = [p for p in programs if p["label"] == label]
        xs = [p["total_deps"] for p in progs]
        ys = [p["fitness"] * 100 for p in progs]
        ax.scatter(
            xs,
            ys,
            c=COLORS[label],
            marker=MARKERS[label],
            alpha=0.4,
            s=30,
            edgecolors="none",
        )
    ax.axhline(y=args.baseline, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("Total dependency edges", fontsize=11)
    ax.set_ylabel("Fitness (%)", fontsize=11)
    ax.set_title("C. Fitness vs DAG Complexity", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)

    # ── Panel D (1,0): Parallel vs Sequential — box plot comparison ──
    ax = axes[1, 0]
    groups = {
        "Sequential\nControl": [
            p["fitness"] * 100
            for p in programs
            if not p["has_parallel"] and p["condition"] == "control"
        ],
        "Parallel\nControl": [
            p["fitness"] * 100
            for p in programs
            if p["has_parallel"] and p["condition"] == "control"
        ],
        "Sequential\nTreatment": [
            p["fitness"] * 100
            for p in programs
            if not p["has_parallel"] and p["condition"] == "treatment"
        ],
        "Parallel\nTreatment": [
            p["fitness"] * 100
            for p in programs
            if p["has_parallel"] and p["condition"] == "treatment"
        ],
    }
    positions = [1, 2, 3.5, 4.5]
    colors_box = [
        COND_COLORS["control"],
        COND_COLORS["control"],
        COND_COLORS["treatment"],
        COND_COLORS["treatment"],
    ]
    hatches = ["", "///", "", "///"]
    bp = ax.boxplot(
        [groups[k] for k in groups],
        positions=positions,
        patch_artist=True,
        widths=0.6,
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker=".", markersize=3, alpha=0.3),
    )
    for patch, color, hatch in zip(bp["boxes"], colors_box, hatches):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
        patch.set_hatch(hatch)
    ax.set_xticks(positions)
    ax.set_xticklabels(list(groups.keys()), fontsize=9)
    for k, pos in zip(groups, positions):
        ax.text(
            pos,
            ax.get_ylim()[0] + 1,
            f"n={len(groups[k])}",
            ha="center",
            fontsize=7,
            color="gray",
        )
    ax.axhline(y=args.baseline, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_ylabel("Fitness (%)", fontsize=11)
    ax.set_title("D. Parallel Branching Effect", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    # ── Panel E (1,1): LLM Steps vs Retrieval Calls (colored by fitness) ──
    ax = axes[1, 1]
    xs = [p["n_tool_steps"] for p in programs]
    ys = [p["n_llm_steps"] for p in programs]
    cs = [p["fitness"] * 100 for p in programs]
    # Add jitter for overlapping points
    jitter = 0.15
    xs_j = [x + np.random.uniform(-jitter, jitter) for x in xs]
    ys_j = [y + np.random.uniform(-jitter, jitter) for y in ys]
    sc = ax.scatter(
        xs_j,
        ys_j,
        c=cs,
        cmap="RdYlGn",
        s=25,
        alpha=0.6,
        edgecolors="none",
        vmin=50,
        vmax=86,
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("Fitness (%)", fontsize=9)
    ax.set_xlabel("Retrieval calls", fontsize=11)
    ax.set_ylabel("LLM reasoning steps", fontsize=11)
    ax.set_title(
        "E. Architecture Space (colored by fitness)", fontsize=13, fontweight="bold"
    )
    ax.grid(True, alpha=0.2)

    # ── Panel F (1,2): Fitness distribution by retrieval budget ──
    ax = axes[1, 2]
    tool_counts = sorted(set(p["n_tool_steps"] for p in programs))
    for tc in tool_counts:
        fits_ctrl = [
            p["fitness"] * 100
            for p in programs
            if p["n_tool_steps"] == tc and p["condition"] == "control"
        ]
        fits_treat = [
            p["fitness"] * 100
            for p in programs
            if p["n_tool_steps"] == tc and p["condition"] == "treatment"
        ]
        offset = 0.15
        if fits_ctrl:
            bp1 = ax.boxplot(
                [fits_ctrl],
                positions=[tc - offset],
                widths=0.25,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=1.5),
                flierprops=dict(marker=".", markersize=2, alpha=0.3),
            )
            bp1["boxes"][0].set_facecolor(COND_COLORS["control"])
            bp1["boxes"][0].set_alpha(0.5)
        if fits_treat:
            bp2 = ax.boxplot(
                [fits_treat],
                positions=[tc + offset],
                widths=0.25,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=1.5),
                flierprops=dict(marker=".", markersize=2, alpha=0.3),
            )
            bp2["boxes"][0].set_facecolor(COND_COLORS["treatment"])
            bp2["boxes"][0].set_alpha(0.5)
    ax.axhline(y=args.baseline, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    # Manual legend
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor=COND_COLORS["control"], alpha=0.5, label="Control"),
            Patch(facecolor=COND_COLORS["treatment"], alpha=0.5, label="Treatment"),
        ],
        fontsize=9,
        loc="lower right",
    )
    ax.set_xlabel("Retrieval calls per sample", fontsize=11)
    ax.set_ylabel("Fitness (%)", fontsize=11)
    ax.set_title("F. Fitness by Retrieval Budget", fontsize=13, fontweight="bold")
    ax.set_xticks(tool_counts)
    ax.grid(True, alpha=0.2, axis="y")

    plt.suptitle(
        "Pareto Analysis: Chain Architecture vs Fitness — hover/steady-state-v2",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
