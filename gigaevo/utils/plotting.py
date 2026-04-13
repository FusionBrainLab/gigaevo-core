"""Frontier annotation for comparison plots.

Ported from tools/comparison.py — annotates significant frontier jumps
on matplotlib axes.
"""

from __future__ import annotations

import pandas as pd


def annotate_frontier_points(
    ax,
    x_vals,
    frontier_vals,
    minimize: bool,
    max_annotations: int,
    color,
    min_improvement_pct: float = 5.0,
):
    if len(x_vals) == 0 or len(frontier_vals) == 0:
        return

    fp = pd.DataFrame(
        {
            "iteration": x_vals,
            "frontier": frontier_vals,
        }
    ).dropna()

    if len(fp) < 2:
        return

    first_val = fp["frontier"].iloc[0]
    last_val = fp["frontier"].iloc[-1]
    total_improvement = abs(last_val - first_val)

    if total_improvement == 0:
        return

    fp["prev"] = fp["frontier"].shift(1)
    fp["jump_size"] = abs(fp["frontier"] - fp["prev"])

    jumps = fp.iloc[1:].copy()
    jumps = jumps[jumps["jump_size"] > 0]

    if jumps.empty:
        return

    min_jump = total_improvement * (min_improvement_pct / 100.0)
    significant_jumps = jumps[jumps["jump_size"] >= min_jump].copy()

    if significant_jumps.empty:
        return

    significant_jumps = significant_jumps.sort_values("iteration", ascending=False)
    if max_annotations > 0:
        significant_jumps = significant_jumps.head(max_annotations)

    significant_jumps = significant_jumps.sort_values("iteration")

    used_y_offsets: list[float] = []
    for _, row in significant_jumps.iterrows():
        x = row["iteration"]
        y = row["frontier"]

        y_offset = 6
        for prev_y in used_y_offsets:
            if abs(y - prev_y) < 0.02:
                y_offset += 10
        used_y_offsets.append(y)

        ax.annotate(
            f"{y:.5g}",
            xy=(x, y),
            xytext=(4, y_offset),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=6,
            fontweight="bold",
            color=color,
            zorder=10,
        )
