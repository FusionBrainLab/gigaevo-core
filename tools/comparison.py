#!/usr/bin/env python3
"""
Compare multiple evolution runs (from different Redis DBs) by plotting
rolling fitness vs iteration on a single chart.

For each run, we compute an iteration-ordered rolling mean and std-dev of
the fitness metric and plot both the mean line and the ±1 std band.

Example usage:
    python tools/evolution_runs_comparison.py \
        --redis-host localhost --redis-port 6379 \
        --run myprefixA@0:Run_A \
        --run myprefixB@1:Run_B \
        --iteration-rolling-window 5 \
        --output-folder results/compare

Run format: --run <prefix>@<db>[:<label>]
Host/port default to the provided --redis-host/--redis-port. You can repeat --run
for as many runs as you want.
"""

import argparse
import asyncio
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tools.utils import (
    RedisRunConfig,
    fetch_evolution_dataframe,
    prepare_iteration_dataframe,
)


def _configure_plotting_style():
    sns.set_theme(style="whitegrid", context="talk", palette="deep")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "font.size": 13,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "axes.labelweight": "semibold",
            "legend.fontsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "figure.dpi": 150,
        }
    )


def _parse_run_arg(arg: str, default_host: str, default_port: int) -> RedisRunConfig:
    """Parse --run argument of the form prefix@db[:label]."""
    # Accept optional label after ':' to keep '@' unambiguous for db
    label: Optional[str] = None
    if ":" in arg:
        prefix_db, label = arg.split(":", 1)
    else:
        prefix_db = arg

    if "@" not in prefix_db:
        raise ValueError("--run format must be prefix@db[:label]")
    prefix, db_str = prefix_db.split("@", 1)
    db = int(db_str)
    return RedisRunConfig(
        redis_host=default_host,
        redis_port=default_port,
        redis_db=db,
        redis_prefix=prefix,
        label=label,
    )


async def _load_runs(configs: list[RedisRunConfig]):
    tasks = [fetch_evolution_dataframe(cfg, add_stage_results=False) for cfg in configs]
    dfs = await asyncio.gather(*tasks)
    return dfs


def _smooth_series(series, window: int):
    if window is None or window <= 1:
        return series
    return series.rolling(window=window, min_periods=1, center=True).mean()


def plot_comparison(
    prepared_dfs: List[Tuple[str, object]],
    *,
    output_folder: Path,
    save_plots: bool = True,
    smooth_window: int = 0,
    smooth_frontier_window: int = 0,
    annotate_frontier: bool = False,
    frontier_annotations_max: int = 15,
    fitness_col: str = "metric_fitness",
    iteration_col: str = "metadata_iteration",
    minimize: bool = False,
):
    if not prepared_dfs:
        logger.error("No prepared data to plot")
        return

    _configure_plotting_style()

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = sns.color_palette(n_colors=len(prepared_dfs))

    for idx, (label, df) in enumerate(prepared_dfs):
        if df.empty:
            logger.warning(f"Run '{label}' has no valid iteration data; skipping plot")
            continue

        c = colors[idx]
        mean_s = _smooth_series(df["running_mean_fitness"], smooth_window)
        std_s = _smooth_series(df["running_std_fitness"], smooth_window)
        plus_s = mean_s + std_s
        minus_s = mean_s - std_s
        # Frontier (cumulative best per iteration, optionally smoothed)
        frontier = df.get("frontier_fitness")
        if frontier is not None:
            frontier_s = _smooth_series(frontier, smooth_frontier_window)
        ax.plot(
            df[iteration_col],
            mean_s,
            linewidth=2.5,
            color=c,
            label=f"{label} (mean)",
        )
        ax.fill_between(
            df[iteration_col],
            minus_s,
            plus_s,
            color=c,
            alpha=0.15,
            label=None,
        )
        # Frontier line
        if frontier is not None:
            ax.plot(
                df[iteration_col],
                frontier_s,
                linewidth=2.0,
                color=c,
                linestyle="--",
                alpha=0.9,
                label=f"{label} (frontier)",
            )
            if annotate_frontier:
                # Annotate improvement points on the (unsmoothed) frontier
                fp = (
                    df[[iteration_col, "frontier_fitness"]]
                    .dropna()
                    .sort_values(iteration_col)
                    .drop_duplicates()
                    .reset_index(drop=True)
                )
                if not fp.empty:
                    diffs = fp["frontier_fitness"].diff()
                    # For minimization, improvement means decrease (diff < 0)
                    # For maximization, improvement means increase (diff > 0)
                    if minimize:
                        improve_mask = diffs.fillna(True) < 0
                    else:
                        improve_mask = diffs.fillna(True) > 0
                    improvements = fp[improve_mask]
                    if not improvements.empty:
                        # Downsample annotations if too many
                        if (
                            frontier_annotations_max > 0
                            and len(improvements) > frontier_annotations_max
                        ):
                            idxs = np.linspace(
                                0,
                                len(improvements) - 1,
                                frontier_annotations_max,
                                dtype=int,
                            )
                            improvements = improvements.iloc[idxs]
                        for _, row in improvements.iterrows():
                            x = row[iteration_col]
                            y = row["frontier_fitness"]
                            ax.annotate(
                                f"{y:.4f}",
                                (x, y),
                                textcoords="offset points",
                                xytext=(0, 6),
                                ha="center",
                                fontsize=9,
                                color=c,
                                alpha=0.9,
                            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness")
    ax.set_title("Evolution Runs: Rolling Fitness vs Iteration")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_plots:
        output_folder.mkdir(parents=True, exist_ok=True)
        png = output_folder / "evolution_runs_comparison.png"
        pdf = output_folder / "evolution_runs_comparison.pdf"
        fig.savefig(png, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        logger.info(f"Saved comparison plot to {png.name} & {pdf.name}")

    plt.show()


async def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple evolution runs by plotting rolling fitness vs iteration"
    )
    parser.add_argument("--redis-host", default="localhost", help="Default Redis host")
    parser.add_argument(
        "--redis-port", type=int, default=6379, help="Default Redis port"
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec: prefix@db[:label]. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--iteration-rolling-window",
        type=int,
        default=5,
        help="Rolling window size for iteration-based running mean/std (default: 5)",
    )
    parser.add_argument(
        "--outlier-method",
        type=str,
        choices=["iqr", "mad", "zscore", "percentile"],
        default="percentile",
        help=(
            "Outlier detection method (default: percentile). "
            "percentile=Simple percentile cutoff (recommended), "
            "mad=Median Absolute Deviation (robust), "
            "iqr=Interquartile Range (Tukey), zscore=Modified Z-score"
        ),
    )
    parser.add_argument(
        "--outlier-multiplier",
        type=float,
        default=None,
        help=(
            "Outlier method multiplier/threshold. Defaults: iqr=1.5, mad=3.5, zscore=3.0. "
            "Higher values = fewer outliers removed."
        ),
    )
    parser.add_argument(
        "--outlier-lower-percentile",
        type=float,
        default=5.0,
        help="For percentile method: lower percentile cutoff (default: 5.0)",
    )
    parser.add_argument(
        "--outlier-upper-percentile",
        type=float,
        default=95.0,
        help="For percentile method: upper percentile cutoff (default: 95.0)",
    )
    parser.add_argument(
        "--no-outlier-removal",
        action="store_true",
        help="Disable outlier removal",
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        help="Folder to save the comparison plot",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save plots to disk (show only)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=0,
        help="Additional smoothing window (rolling, centered) applied to mean/std before plotting. 0 disables.",
    )
    parser.add_argument(
        "--smooth-frontier-window",
        type=int,
        default=0,
        help="Smoothing window (rolling, centered) for the frontier line. 0 disables.",
    )
    parser.add_argument(
        "--annotate-frontier",
        action="store_true",
        help="Annotate frontier improvement points with their values",
    )
    parser.add_argument(
        "--frontier-annotations-max",
        type=int,
        default=15,
        help="Maximum number of frontier annotations per run (0 = unlimited)",
    )
    parser.add_argument(
        "--fitness-col",
        type=str,
        default="metric_fitness",
        help="Fitness column name (default: metric_fitness)",
    )
    parser.add_argument(
        "--iteration-col",
        type=str,
        default="metadata_iteration",
        help="Iteration column name (default: metadata_iteration)",
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Set if lower fitness values are better (minimization problem)",
    )
    args = parser.parse_args()
    output_folder = Path(args.output_folder)

    # Parse run specs
    run_configs: List[RedisRunConfig] = []
    for run_arg in args.run:
        cfg = _parse_run_arg(run_arg, args.redis_host, args.redis_port)
        run_configs.append(cfg)

    logger.info(
        f"Loaded {len(run_configs)} runs: {[c.display_label() for c in run_configs]}"
    )

    # Fetch data concurrently
    dfs = await _load_runs(run_configs)

    prepared: List[Tuple[str, object]] = []
    for cfg, df in zip(run_configs, dfs):
        if df is None or df.empty:
            logger.warning(f"Run '{cfg.display_label()}': no data found")
            continue
        prepared_df = prepare_iteration_dataframe(
            df,
            iteration_rolling_window=args.iteration_rolling_window,
            remove_outliers=not args.no_outlier_removal,
            outlier_method=args.outlier_method,
            outlier_multiplier=args.outlier_multiplier,
            outlier_lower_percentile=args.outlier_lower_percentile,
            outlier_upper_percentile=args.outlier_upper_percentile,
            fitness_col=args.fitness_col,
            iteration_col=args.iteration_col,
            minimize=args.minimize,
        )
        if prepared_df.empty:
            logger.warning(
                f"Run '{cfg.display_label()}': no valid iteration/fitness data after filtering"
            )
            continue
        prepared.append((cfg.display_label(), prepared_df))

    plot_comparison(
        prepared,
        output_folder=output_folder,
        save_plots=not args.no_save,
        smooth_window=args.smooth_window,
        smooth_frontier_window=args.smooth_frontier_window,
        annotate_frontier=args.annotate_frontier,
        frontier_annotations_max=args.frontier_annotations_max,
        fitness_col=args.fitness_col,
        iteration_col=args.iteration_col,
        minimize=args.minimize,
    )


if __name__ == "__main__":
    asyncio.run(main())
