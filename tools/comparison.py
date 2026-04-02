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
from typing import Literal

from loguru import logger
import matplotlib

matplotlib.use("Agg")  # headless by default; override with --show or MPLBACKEND env var
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

from tools.utils import (
    RedisRunConfig,
    add_frontier_from_redis_to_dataframe,
    fetch_evolution_dataframe,
    prepare_iteration_dataframe,
)

# Standard matplotlib tab10 colors (the classic default)
MATPLOTLIB_COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
]


def _configure_plotting_style(use_latex: bool = False):
    """Configure matplotlib for publication-quality plots.

    Args:
        use_latex: If True, use LaTeX for text rendering (requires LaTeX installation).
    """
    # Reset to defaults first for clean slate
    plt.rcdefaults()

    # Use seaborn's clean style as base
    sns.set_theme(style="ticks", context="paper")

    rc_params = {
        # Typography - clean, professional fonts
        "font.family": "serif" if use_latex else "sans-serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
            "Computer Modern Roman",
        ],
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        # Axes
        "axes.titlesize": 12,
        "axes.titleweight": "medium",
        "axes.titlepad": 10,
        "axes.labelsize": 11,
        "axes.labelweight": "medium",
        "axes.labelpad": 6,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Grid - subtle, unobtrusive
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.color": "#E0E0E0",
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        # Ticks
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Legend
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#CCCCCC",
        "legend.fancybox": False,
        "legend.borderpad": 0.5,
        "legend.labelspacing": 0.4,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.5,
        # Lines
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        # Figure
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "figure.constrained_layout.use": True,
        # Saving
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        # PDF/PS options for vector graphics
        "pdf.fonttype": 42,  # TrueType fonts (editable in Illustrator)
        "ps.fonttype": 42,
    }

    if use_latex:
        rc_params.update(
            {
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
            }
        )

    plt.rcParams.update(rc_params)


def _parse_run_arg(arg: str, default_host: str, default_port: int) -> RedisRunConfig:
    """Parse --run argument of the form prefix@db[:label]."""
    # Accept optional label after ':' to keep '@' unambiguous for db
    label: str | None = None
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


SmoothingMethod = Literal["lowess", "ema", "savgol", "gaussian", "rolling", "none"]


def _smooth_series(
    series,
    window: int,
    method: SmoothingMethod = "lowess",
    polyorder: int = 3,
    lowess_frac: float | None = None,
):
    """Apply smoothing to a series using various methods.

    Args:
        series: Input data series (pandas Series or numpy array).
        window: Window size for smoothing. Interpretation varies by method:
            - lowess: Used to compute fraction if lowess_frac not provided
            - ema: Span for exponential moving average
            - savgol: Window size (will be made odd)
            - gaussian: Sigma value
            - rolling: Window size
        method: Smoothing method:
            - "lowess": LOWESS (Locally Weighted Scatterplot Smoothing) - RECOMMENDED
              Produces very smooth curves without ringing artifacts.
            - "ema": Exponential Moving Average - smooth, fast, no edge artifacts.
            - "savgol": Savitzky-Golay - preserves peaks but can have ringing.
            - "gaussian": Gaussian kernel - smooth but can blur features.
            - "rolling": Simple moving average - can create blocky artifacts.
            - "none": No smoothing.
        polyorder: Polynomial order for Savitzky-Golay filter (default: 3).
        lowess_frac: Fraction of data used for LOWESS smoothing (0.0-1.0).
            If None, computed from window as window / len(data).
            Smaller = less smoothing, larger = more smoothing.

    Returns:
        Smoothed series of the same type as input.
    """
    if window is None or window <= 1 or method == "none":
        return series

    import pandas as pd

    # Convert to numpy for processing, handle NaNs
    values = series.values if hasattr(series, "values") else np.asarray(series)
    is_series = isinstance(series, pd.Series)
    n = len(values)

    # Handle NaN values by interpolating, smoothing, then restoring NaN positions
    nan_mask = np.isnan(values)
    if nan_mask.all():
        return series

    # Interpolate NaNs for smoothing
    if nan_mask.any():
        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) < 2:
            return series
        values_interp = np.interp(
            np.arange(n),
            valid_idx,
            values[valid_idx],
        )
    else:
        values_interp = values.copy()

    if method == "lowess":
        # LOWESS - excellent for smooth curves without artifacts
        # Fraction controls smoothness: 0.05-0.1 = local, 0.2-0.3 = moderate, 0.5+ = very smooth
        if lowess_frac is not None:
            frac = lowess_frac
        else:
            # Convert window to fraction, with sensible bounds
            frac = min(max(window / n, 0.05), 0.5)

        x = np.arange(n)
        # LOWESS returns sorted by x, so we just take the smoothed y values
        smoothed_result = lowess(values_interp, x, frac=frac, return_sorted=True)
        smoothed = smoothed_result[:, 1]

    elif method == "ema":
        # Exponential Moving Average - smooth without edge effects
        # Use pandas EWM for proper handling
        df_temp = pd.Series(values_interp)
        # span parameter: larger = smoother
        smoothed = df_temp.ewm(span=window, adjust=True, min_periods=1).mean().values

    elif method == "savgol":
        # Savitzky-Golay filter - can preserve features but may ring
        win = int(window)
        if win % 2 == 0:
            win += 1
        win = max(win, polyorder + 2)
        if win > n:
            win = n if n % 2 == 1 else n - 1
            win = max(win, polyorder + 2)
        if n >= win:
            smoothed = savgol_filter(values_interp, win, polyorder, mode="interp")
        else:
            smoothed = values_interp

    elif method == "gaussian":
        # Gaussian filter - smooth, window is sigma
        sigma = window / 2.0
        smoothed = gaussian_filter1d(values_interp, sigma=sigma, mode="reflect")

    elif method == "rolling":
        # Simple rolling mean
        kernel = np.ones(window) / window
        padded = np.pad(values_interp, (window // 2, window // 2), mode="reflect")
        smoothed = np.convolve(padded, kernel, mode="valid")
        if len(smoothed) > n:
            excess = len(smoothed) - n
            smoothed = smoothed[excess // 2 : len(smoothed) - (excess - excess // 2)]

    else:
        smoothed = values_interp

    # Restore NaN positions
    if nan_mask.any():
        smoothed[nan_mask] = np.nan

    if is_series:
        return pd.Series(smoothed, index=series.index)
    return smoothed


def _aggregate_per_iteration(
    df,
    iteration_col: str,
    fitness_col: str,
) -> tuple:
    """Aggregate data per iteration to reduce noise before smoothing.

    Returns (iterations, means, stds, frontier) arrays with one value per unique iteration.
    """
    # Group by iteration and compute mean/std
    grouped = (
        df.groupby(iteration_col)
        .agg(
            {
                "running_mean_fitness": "mean",
                "running_std_fitness": lambda x: np.sqrt((x**2).mean()),  # RMS of stds
                "frontier_fitness": "last",  # Frontier is already cumulative
            }
        )
        .reset_index()
    )

    grouped = grouped.sort_values(iteration_col)

    return (
        grouped[iteration_col].values,
        grouped["running_mean_fitness"].values,
        grouped["running_std_fitness"].values,
        grouped["frontier_fitness"].values,
    )


def _select_frontier_improvements(df, iteration_col: str, minimize: bool):
    fp = (
        df[[iteration_col, "frontier_fitness"]]
        .dropna()
        .sort_values(iteration_col)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if fp.empty:
        return fp

    diffs = fp["frontier_fitness"].diff()
    if minimize:
        improve_mask = diffs < 0
    else:
        improve_mask = diffs > 0
    return fp[improve_mask]


def plot_comparison(
    prepared_dfs: list[tuple[str, pd.DataFrame]],
    *,
    output_folder: Path,
    save_plots: bool = True,
    output_stem: str = "evolution_runs_comparison",
    show_plot: bool = True,
    smooth_window: int = 0,
    smooth_method: SmoothingMethod = "lowess",
    lowess_frac: float | None = None,
    annotate_frontier: bool = False,
    frontier_annotations_max: int = 5,
    min_improvement_pct: float = 5.0,
    fitness_col: str = "metric_fitness",
    iteration_col: str = "metadata_iteration",
    minimize: bool = False,
    use_latex: bool = False,
    figure_width: float = 7.0,
    figure_height: float = 4.5,
    show_std_band: bool = True,
    show_frontier: bool = True,
    legend_location: str = "best",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    aggregate_iterations: bool = True,
    original_memory_score: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    smooth_frontier_window: int = 0,
) -> tuple[Path, Path] | None:
    """Plot comparison of multiple evolution runs with publication-quality styling.

    Args:
        prepared_dfs: List of (label, dataframe) tuples.
        output_folder: Directory for saving plots.
        save_plots: Whether to save plots to disk.
        output_stem: Base filename used for saved plots.
        show_plot: Whether to show the plot interactively.
        smooth_window: Smoothing window for mean/std curves.
        smooth_method: Smoothing algorithm (for mean/std only; frontier is never smoothed). Options:
            - "lowess": LOWESS smoothing (RECOMMENDED - very smooth, no artifacts)
            - "ema": Exponential Moving Average (smooth, fast)
            - "savgol": Savitzky-Golay (preserves peaks but can ring)
            - "gaussian": Gaussian kernel
            - "rolling": Simple moving average
            - "none": No smoothing
        lowess_frac: Fraction for LOWESS smoothing (0.0-1.0). Higher = smoother.
            If None, computed automatically from smooth_window.
        annotate_frontier: Whether to annotate frontier improvement points.
        frontier_annotations_max: Maximum frontier annotations per run.
        min_improvement_pct: Minimum improvement (% of total) to annotate.
        fitness_col: Name of fitness column.
        iteration_col: Name of iteration column.
        minimize: True if lower fitness is better.
        use_latex: Use LaTeX for text rendering.
        figure_width: Figure width in inches (default: 7.0 for single-column).
        figure_height: Figure height in inches.
        show_std_band: Show standard deviation bands (disable for cleaner plots with many runs).
        show_frontier: Show frontier (best-so-far) lines.
        legend_location: Legend location ("best", "upper right", "outside", etc.).
        title: Custom title (default: auto-generated).
        xlabel: Custom x-axis label.
        ylabel: Custom y-axis label.
        aggregate_iterations: If True, aggregate data per iteration before smoothing
            to reduce point-to-point noise. Recommended for cleaner plots.
    """
    if not prepared_dfs:
        logger.error("No prepared data to plot")
        return None

    _configure_plotting_style(use_latex=use_latex)

    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    # Use standard matplotlib colors
    colors = MATPLOTLIB_COLORS
    n_runs = len(prepared_dfs)

    # Adjust visual parameters based on number of runs
    if n_runs <= 3:
        line_width = 2.0
        band_alpha = 0.15
        frontier_alpha = 0.8
    elif n_runs <= 5:
        line_width = 1.6
        band_alpha = 0.10
        frontier_alpha = 0.7
    else:
        # Many runs: thinner lines, lighter bands
        line_width = 1.2
        band_alpha = 0.08
        frontier_alpha = 0.6

    for idx, (label, df) in enumerate(prepared_dfs):
        if df.empty:
            logger.warning(f"Run '{label}' has no valid iteration data; skipping plot")
            continue

        color = colors[idx % len(colors)]

        # Aggregate per iteration to reduce noise (one point per iteration)
        if aggregate_iterations:
            x_vals, mean_vals, std_vals, frontier_vals = _aggregate_per_iteration(
                df, iteration_col, fitness_col
            )
        else:
            x_vals = df[iteration_col].values
            mean_vals = df["running_mean_fitness"].values
            std_vals = df["running_std_fitness"].values
            frontier_vals = df.get("frontier_fitness")
            if frontier_vals is not None:
                frontier_vals = frontier_vals.values

        # Apply smoothing
        mean_s = _smooth_series(
            mean_vals, smooth_window, method=smooth_method, lowess_frac=lowess_frac
        )
        std_s = _smooth_series(
            std_vals, smooth_window, method=smooth_method, lowess_frac=lowess_frac
        )

        # Ensure std is non-negative after smoothing
        std_s = np.maximum(std_s, 0)

        # Plot confidence band (skip if disabled or too many runs)
        if show_std_band:
            plus_s = mean_s + std_s
            minus_s = mean_s - std_s
            ax.fill_between(
                x_vals,
                minus_s,
                plus_s,
                color=color,
                alpha=band_alpha,
                linewidth=0,
                zorder=1,
            )

        # Plot mean line
        ax.plot(
            x_vals,
            mean_s,
            linewidth=line_width,
            color=color,
            label=label,
            zorder=3,
            solid_capstyle="round",
        )

        # Frontier line (cumulative best per iteration) - NEVER smoothed, strict step function
        if show_frontier and frontier_vals is not None:
            ax.step(
                x_vals,
                frontier_vals,
                where="post",  # Step occurs after the x value (standard for cumulative best)
                linewidth=line_width * 0.8,
                color=color,
                linestyle="--",
                alpha=frontier_alpha,
                zorder=2,
            )

            if annotate_frontier:
                # Annotate significant frontier jumps
                _annotate_frontier_points(
                    ax,
                    x_vals,
                    frontier_vals,
                    minimize,
                    frontier_annotations_max,
                    color,
                    min_improvement_pct=min_improvement_pct,
                )

    # Axis labels
    ax.set_xlabel(xlabel or "Iteration")
    ax.set_ylabel(ylabel or "Fitness")

    # Title (optional for papers, often omitted)
    if title:
        ax.set_title(title, pad=12)

    # Configure axis formatting
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=8))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

    # Subtle axis formatting
    ax.tick_params(axis="both", which="major", length=4, width=0.8)
    ax.tick_params(axis="both", which="minor", length=2, width=0.5)

    if original_memory_score is not None:
        ax.axhline(
            y=original_memory_score,
            color="red",
            linestyle="--",
            linewidth=2.0,
            label=f"Original memory score ({original_memory_score:g})",
        )

    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    else:
        # Zoom y-axis to data range (don't waste space starting at 0)
        auto_y_min, auto_y_max = ax.get_ylim()
        y_range = auto_y_max - auto_y_min
        pad = max(y_range * 0.15, 0.005)
        ax.set_ylim(bottom=auto_y_min - pad, top=auto_y_max + pad)

    # Legend - compact styling
    legend_kwargs = {
        "frameon": True,
        "fontsize": 8 if n_runs > 5 else 9,
        "handlelength": 1.5,
        "labelspacing": 0.3,
    }

    if legend_location == "outside":
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            **legend_kwargs,
        )
    else:
        ax.legend(loc=legend_location, **legend_kwargs)

    # Finalize layout
    fig.tight_layout()
    saved_paths: tuple[Path, Path] | None = None
    if save_plots:
        output_folder.mkdir(parents=True, exist_ok=True)
        png = output_folder / f"{output_stem}.png"
        pdf = output_folder / f"{output_stem}.pdf"
        svg = output_folder / f"{output_stem}.svg"

        fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf, bbox_inches="tight", facecolor="white")
        fig.savefig(svg, bbox_inches="tight", facecolor="white")

        logger.info(f"Saved comparison plots: {png.name}, {pdf.name}, {svg.name}")
        saved_paths = (png, pdf)
    if show_plot:
        plt.show()
    plt.close(fig)
    return saved_paths


def _annotate_frontier_points(
    ax,
    x_vals,
    frontier_vals,
    minimize: bool,
    max_annotations: int,
    color,
    min_improvement_pct: float = 5.0,
):
    """Annotate the latest N frontier jumps, filtered by minimum delta.

    Args:
        ax: Matplotlib axes object.
        x_vals: Array of x values (iterations).
        frontier_vals: Frontier values (monotonic step function).
        minimize: True if lower is better.
        max_annotations: Maximum number of jumps to annotate (latest N by iteration).
        color: Color for the annotation text.
        min_improvement_pct: Minimum jump size as % of total improvement to show.
    """
    import pandas as pd

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

    # Find all jumps (where frontier value changes)
    fp["prev"] = fp["frontier"].shift(1)
    fp["jump_size"] = abs(fp["frontier"] - fp["prev"])

    # First row has no previous, skip it for jump detection
    jumps = fp.iloc[1:].copy()
    jumps = jumps[jumps["jump_size"] > 0]  # Only actual changes

    if jumps.empty:
        return

    # Filter by minimum improvement threshold
    min_jump = total_improvement * (min_improvement_pct / 100.0)
    significant_jumps = jumps[jumps["jump_size"] >= min_jump].copy()

    if significant_jumps.empty:
        return

    # Take the latest N jumps by iteration (we care about late steps mostly)
    significant_jumps = significant_jumps.sort_values("iteration", ascending=False)
    if max_annotations > 0:
        significant_jumps = significant_jumps.head(max_annotations)

    # Sort back by iteration (ascending) for consistent display
    significant_jumps = significant_jumps.sort_values("iteration")

    # Annotate significant jumps with staggered offsets to avoid overlap
    used_y_offsets = []
    for _, row in significant_jumps.iterrows():
        x = row["iteration"]
        y = row["frontier"]

        # Stagger vertically: alternate up/right offsets to avoid overlap
        y_offset = 6
        for prev_y in used_y_offsets:
            if abs(y - prev_y) < 0.02:  # Close to a previous annotation
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


async def export_run_plot(
    config: RedisRunConfig,
    *,
    output_folder: Path,
    output_stem: str,
    iteration_rolling_window: int = 5,
    remove_outliers: bool = True,
    outlier_method: str = "percentile",
    outlier_multiplier: float | None = None,
    outlier_lower_percentile: float = 5.0,
    outlier_upper_percentile: float = 95.0,
    smooth_window: int = 0,
    smooth_frontier_window: int = 0,
    annotate_frontier: bool = False,
    frontier_annotations_max: int = 15,
    fitness_col: str = "metric_fitness",
    iteration_col: str = "metadata_iteration",
    minimize: bool = False,
    original_memory_score: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> tuple[Path, Path] | None:
    df = await fetch_evolution_dataframe(config, add_stage_results=False)
    if df.empty:
        logger.warning("Run '{}' has no data for plotting", config.display_label())
        return None

    prepared_df = prepare_iteration_dataframe(
        df,
        iteration_rolling_window=iteration_rolling_window,
        remove_outliers=remove_outliers,
        outlier_method=outlier_method,
        outlier_multiplier=outlier_multiplier,
        outlier_lower_percentile=outlier_lower_percentile,
        outlier_upper_percentile=outlier_upper_percentile,
        fitness_col=fitness_col,
        iteration_col=iteration_col,
        minimize=minimize,
    )
    if prepared_df.empty:
        logger.warning(
            "Run '{}' has no plottable iteration data after filtering",
            config.display_label(),
        )
        return None

    return plot_comparison(
        [(config.display_label(), prepared_df)],
        output_folder=output_folder,
        save_plots=True,
        output_stem=output_stem,
        show_plot=False,
        smooth_window=smooth_window,
        annotate_frontier=annotate_frontier,
        frontier_annotations_max=frontier_annotations_max,
        fitness_col=fitness_col,
        iteration_col=iteration_col,
        minimize=minimize,
        original_memory_score=original_memory_score,
        y_min=y_min,
        y_max=y_max,
        smooth_frontier_window=smooth_frontier_window,
    )


async def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple evolution runs by plotting rolling fitness vs iteration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # === Redis Connection ===
    conn_group = parser.add_argument_group("Redis Connection")
    conn_group.add_argument(
        "--redis-host", default="localhost", help="Default Redis host"
    )
    conn_group.add_argument(
        "--redis-port", type=int, default=6379, help="Default Redis port"
    )
    conn_group.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec: prefix@db[:label]. Repeat for multiple runs.",
    )

    # === Data Processing ===
    data_group = parser.add_argument_group("Data Processing")
    data_group.add_argument(
        "--iteration-rolling-window",
        type=int,
        default=5,
        help="Rolling window size for iteration-based running mean/std",
    )
    data_group.add_argument(
        "--outlier-method",
        type=str,
        choices=["iqr", "mad", "zscore", "percentile"],
        default="percentile",
        help=(
            "Outlier detection method. "
            "percentile=Simple percentile cutoff (recommended), "
            "mad=Median Absolute Deviation (robust), "
            "iqr=Interquartile Range (Tukey), zscore=Modified Z-score"
        ),
    )
    data_group.add_argument(
        "--outlier-multiplier",
        type=float,
        default=None,
        help=(
            "Outlier method multiplier/threshold. Defaults: iqr=1.5, mad=3.5, zscore=3.0. "
            "Higher values = fewer outliers removed."
        ),
    )
    data_group.add_argument(
        "--outlier-lower-percentile",
        type=float,
        default=5.0,
        help="For percentile method: lower percentile cutoff",
    )
    data_group.add_argument(
        "--outlier-upper-percentile",
        type=float,
        default=95.0,
        help="For percentile method: upper percentile cutoff",
    )
    data_group.add_argument(
        "--no-outlier-removal",
        action="store_true",
        help="Disable outlier removal",
    )
    data_group.add_argument(
        "--fitness-col",
        type=str,
        default="metric_fitness",
        help="Fitness column name",
    )
    data_group.add_argument(
        "--iteration-col",
        type=str,
        default="metadata_iteration",
        help="Iteration column name",
    )
    data_group.add_argument(
        "--minimize",
        action="store_true",
        help="Set if lower fitness values are better (minimization problem)",
    )

    # === Smoothing Options ===
    smooth_group = parser.add_argument_group("Smoothing Options")
    smooth_group.add_argument(
        "--smooth-window",
        type=int,
        default=10,
        help="Smoothing window for mean/std curves. 0 disables smoothing.",
    )
    smooth_group.add_argument(
        "--smooth-method",
        type=str,
        choices=["lowess", "ema", "savgol", "gaussian", "rolling", "none"],
        default="lowess",
        help=(
            "Smoothing algorithm: "
            "lowess=LOWESS (RECOMMENDED - very smooth, no artifacts), "
            "ema=Exponential Moving Average (smooth, fast), "
            "savgol=Savitzky-Golay (preserves peaks but can ring), "
            "gaussian=Gaussian kernel, "
            "rolling=Simple moving average, "
            "none=No smoothing"
        ),
    )
    smooth_group.add_argument(
        "--lowess-frac",
        type=float,
        default=None,
        help=(
            "LOWESS smoothing fraction (0.0-1.0). Higher = smoother. "
            "If not specified, computed from smooth-window. "
            "Recommended: 0.1 (local), 0.2 (moderate), 0.3+ (very smooth)"
        ),
    )
    smooth_group.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Disable per-iteration aggregation (may increase spikiness)",
    )

    # === Plot Appearance ===
    style_group = parser.add_argument_group("Plot Appearance")
    style_group.add_argument(
        "--figure-width",
        type=float,
        default=7.0,
        help="Figure width in inches (7.0 for single-column, 14.0 for double)",
    )
    style_group.add_argument(
        "--figure-height",
        type=float,
        default=4.5,
        help="Figure height in inches",
    )
    style_group.add_argument(
        "--use-latex",
        action="store_true",
        help="Use LaTeX for text rendering (requires LaTeX installation)",
    )
    style_group.add_argument(
        "--no-std-band",
        action="store_true",
        help="Hide standard deviation bands (cleaner with many runs)",
    )
    style_group.add_argument(
        "--no-frontier",
        action="store_true",
        help="Hide frontier (best-so-far) lines",
    )
    style_group.add_argument(
        "--legend-location",
        type=str,
        default="best",
        choices=[
            "best",
            "upper right",
            "upper left",
            "lower right",
            "lower left",
            "outside",
        ],
        help="Legend placement",
    )
    style_group.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom plot title (omit for no title, common in papers)",
    )
    style_group.add_argument(
        "--xlabel",
        type=str,
        default=None,
        help="Custom x-axis label",
    )
    style_group.add_argument(
        "--ylabel",
        type=str,
        default=None,
        help="Custom y-axis label",
    )

    # === Annotations ===
    annot_group = parser.add_argument_group("Annotations")
    annot_group.add_argument(
        "--annotate-frontier",
        action="store_true",
        help="Annotate frontier improvement points with their values",
    )
    annot_group.add_argument(
        "--frontier-annotations-max",
        type=int,
        default=5,
        help="Top-N largest jumps to annotate (0 = unlimited after filtering)",
    )
    annot_group.add_argument(
        "--min-improvement-pct",
        type=float,
        default=5.0,
        help=(
            "Minimum jump size (as %% of total improvement) to annotate. "
            "Higher = fewer annotations. Default: 5.0%%"
        ),
    )

    # === Output ===
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output-folder",
        required=True,
        help="Folder to save the comparison plot",
    )
    output_group.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save plots to disk (show only)",
    )
    output_group.add_argument(
        "--show",
        action="store_true",
        help="Open interactive display window after saving (requires a display; sets MPLBACKEND)",
    )

    args = parser.parse_args()
    output_folder = Path(args.output_folder)

    # Parse run specs
    run_configs: list[RedisRunConfig] = []
    for run_arg in args.run:
        cfg = _parse_run_arg(run_arg, args.redis_host, args.redis_port)
        run_configs.append(cfg)

    logger.info(
        f"Loaded {len(run_configs)} runs: {[c.display_label() for c in run_configs]}"
    )

    # Fetch data concurrently
    dfs = await _load_runs(run_configs)

    prepared: list[tuple[str, object]] = []
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

        # Try to fetch the authoritative frontier from Redis metrics history.
        # This handles NO_CACHE stage re-evaluations correctly (metric changes).
        prepared_df = add_frontier_from_redis_to_dataframe(
            prepared_df,
            redis_host=cfg.redis_host,
            redis_port=cfg.redis_port,
            redis_db=cfg.redis_db,
            redis_prefix=cfg.redis_prefix,
            metric_key=args.fitness_col,
            iteration_col=args.iteration_col,
        )

        prepared.append((cfg.display_label(), prepared_df))

    plot_comparison(
        prepared,
        output_folder=output_folder,
        save_plots=not args.no_save,
        smooth_window=args.smooth_window,
        smooth_method=args.smooth_method,
        lowess_frac=args.lowess_frac,
        annotate_frontier=args.annotate_frontier,
        frontier_annotations_max=args.frontier_annotations_max,
        min_improvement_pct=args.min_improvement_pct,
        fitness_col=args.fitness_col,
        iteration_col=args.iteration_col,
        minimize=args.minimize,
        use_latex=args.use_latex,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        show_std_band=not args.no_std_band,
        show_frontier=not args.no_frontier,
        legend_location=args.legend_location,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        aggregate_iterations=not args.no_aggregate,
        show_plot=args.show,
    )


if __name__ == "__main__":
    asyncio.run(main())
