"""Plot sub-group: comparison and trajectory plotting commands."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal

import click
import pandas as pd

from gigaevo.cli.run_resolver import RunResolver
from gigaevo.utils.redis import RedisRunConfig

SmoothingMethod = Literal["lowess", "ema", "savgol", "gaussian", "rolling", "none"]


def _build_redis_config(
    run_config,
    redis_host: str = "localhost",
    redis_port: int = 6379,
) -> RedisRunConfig:
    """Build a RedisRunConfig from a monitoring RunConfig."""
    spec = run_config.run_spec
    return RedisRunConfig(
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=spec.db,
        redis_prefix=spec.prefix,
        label=spec.label,
    )


def _fetch_run_data(
    run_configs: list,
    redis_host: str,
    redis_port: int,
    metric: str = "fitness",
) -> list[tuple[str, pd.DataFrame]]:
    """Fetch and prepare DataFrames for each run. Returns list of (label, df)."""
    from tools.utils import (
        fetch_evolution_dataframe,
        prepare_iteration_dataframe,
    )

    results: list[tuple[str, pd.DataFrame]] = []
    for rc in run_configs:
        config = _build_redis_config(rc, redis_host, redis_port)
        raw_df = asyncio.run(fetch_evolution_dataframe(config, add_stage_results=False))
        if raw_df.empty:
            continue
        prepared = prepare_iteration_dataframe(
            raw_df,
            fitness_col=f"metric_{metric}",
        )
        if prepared.empty:
            continue
        label = config.display_label()
        results.append((label, prepared))
    return results


def _smooth_series(series, window: int, method: SmoothingMethod):
    """Apply smoothing to a pandas Series. Lazy-imports scipy/statsmodels."""
    import numpy as np

    if method == "none" or window <= 1:
        return series

    values = series.values
    n = len(values)
    nan_mask = np.isnan(values)
    if nan_mask.all():
        return series

    if nan_mask.any():
        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) < 2:
            return series
        values_interp = np.interp(np.arange(n), valid_idx, values[valid_idx])
    else:
        values_interp = values.copy()

    if method == "lowess":
        from statsmodels.nonparametric.smoothers_lowess import lowess

        frac = min(max(window / n, 0.05), 0.5)
        x = np.arange(n)
        result = lowess(values_interp, x, frac=frac, return_sorted=True)
        smoothed = result[:, 1]
    elif method == "ema":
        smoothed = (
            pd.Series(values_interp)
            .ewm(span=window, adjust=True, min_periods=1)
            .mean()
            .values
        )
    elif method == "savgol":
        from scipy.signal import savgol_filter

        win = int(window)
        if win % 2 == 0:
            win += 1
        win = max(win, 5)
        if win > n:
            win = n if n % 2 == 1 else n - 1
            win = max(win, 5)
        if n >= win:
            smoothed = savgol_filter(values_interp, win, 3, mode="interp")
        else:
            smoothed = values_interp
    elif method == "gaussian":
        from scipy.ndimage import gaussian_filter1d

        sigma = window / 2.0
        smoothed = gaussian_filter1d(values_interp, sigma=sigma, mode="reflect")
    elif method == "rolling":
        kernel = np.ones(window) / window
        padded = np.pad(values_interp, (window // 2, window // 2), mode="reflect")
        smoothed = np.convolve(padded, kernel, mode="valid")
        if len(smoothed) > n:
            excess = len(smoothed) - n
            smoothed = smoothed[excess // 2 : len(smoothed) - (excess - excess // 2)]
    else:
        smoothed = values_interp

    if nan_mask.any():
        smoothed[nan_mask] = np.nan

    return pd.Series(smoothed, index=series.index)


@click.group()
def plot() -> None:
    """Generate plots from evolution runs."""


@plot.command("comparison")
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Output directory for plot files.",
)
@click.option(
    "--smoothing",
    type=click.Choice(
        ["lowess", "ema", "savgol", "gaussian", "rolling", "none"],
        case_sensitive=False,
    ),
    default="lowess",
    help="Smoothing method.",
)
@click.option("--window", type=int, default=5, help="Smoothing window size.")
@click.option("--show", is_flag=True, default=False, help="Show plot interactively.")
@click.option("--metric", default="fitness", help="Metric to plot.")
@click.pass_context
def comparison(
    ctx: click.Context,
    output_dir: str,
    smoothing: str,
    window: int,
    show: bool,
    metric: str,
) -> None:
    """Plot fitness comparison across runs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_configs = RunResolver.resolve(
        ctx.obj["experiment"],
        ctx.obj["runs"],
        ctx.obj["redis_host"],
        ctx.obj["redis_port"],
    )

    prepared_dfs = _fetch_run_data(
        run_configs,
        ctx.obj["redis_host"],
        ctx.obj["redis_port"],
        metric=metric,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    iteration_col = "metadata_iteration"

    run_labels = []
    for i, (label, df) in enumerate(prepared_dfs):
        color = colors[i % len(colors)]
        iters = df[iteration_col]

        mean_vals = df["running_mean_fitness"]
        if smoothing != "none" and window > 1:
            mean_vals = _smooth_series(mean_vals, window, smoothing)

        ax.plot(iters, mean_vals, label=f"{label} (mean)", color=color, linewidth=1.5)

        if "running_std_fitness" in df.columns:
            std_vals = df["running_std_fitness"]
            ax.fill_between(
                iters,
                mean_vals - std_vals,
                mean_vals + std_vals,
                alpha=0.15,
                color=color,
            )

        if "frontier_fitness" in df.columns:
            ax.plot(
                iters,
                df["frontier_fitness"],
                label=f"{label} (best)",
                color=color,
                linewidth=1.0,
                linestyle="--",
            )

        run_labels.append(label)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Evolution Runs Comparison ({metric})")
    ax.legend(fontsize=8)

    stem = "evolution_runs_comparison"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_path / f"{stem}.{ext}", dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)

    summary = {
        "output_dir": str(out_path),
        "runs": run_labels,
        "smoothing": smoothing,
        "metric": metric,
        "files": [f"{stem}.png", f"{stem}.pdf", f"{stem}.svg"],
    }
    click.echo(json.dumps(summary, indent=2))


@plot.command("trajectory")
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Output directory for plot files.",
)
@click.option("--metric", default="fitness", help="Metric to plot.")
@click.option("--pdf", is_flag=True, default=False, help="Also save as PDF.")
@click.option(
    "--no-best", is_flag=True, default=False, help="Suppress best fitness line."
)
@click.option(
    "--no-mean", is_flag=True, default=False, help="Suppress mean fitness line."
)
@click.option(
    "--no-std",
    is_flag=True,
    default=False,
    help="Suppress standard deviation band.",
)
@click.pass_context
def trajectory(
    ctx: click.Context,
    output_dir: str,
    metric: str,
    pdf: bool,
    no_best: bool,
    no_mean: bool,
    no_std: bool,
) -> None:
    """Plot fitness trajectory for a run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_configs = RunResolver.resolve(
        ctx.obj["experiment"],
        ctx.obj["runs"],
        ctx.obj["redis_host"],
        ctx.obj["redis_port"],
    )

    prepared_dfs = _fetch_run_data(
        run_configs,
        ctx.obj["redis_host"],
        ctx.obj["redis_port"],
        metric=metric,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    iteration_col = "metadata_iteration"

    for label, df in prepared_dfs:
        iters = df[iteration_col]

        if not no_mean and "running_mean_fitness" in df.columns:
            ax.plot(
                iters,
                df["running_mean_fitness"],
                label=f"{label} (mean)",
                linewidth=1.5,
            )

        if (
            not no_std
            and "running_std_fitness" in df.columns
            and "running_mean_fitness" in df.columns
        ):
            mean_vals = df["running_mean_fitness"]
            std_vals = df["running_std_fitness"]
            ax.fill_between(
                iters,
                mean_vals - std_vals,
                mean_vals + std_vals,
                alpha=0.15,
            )

        if not no_best and "frontier_fitness" in df.columns:
            ax.plot(
                iters,
                df["frontier_fitness"],
                label=f"{label} (best)",
                linewidth=1.0,
                linestyle="--",
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Fitness Trajectory ({metric})")
    ax.legend(fontsize=8)

    files_created = []
    fig.savefig(out_path / "trajectory.png", dpi=300, bbox_inches="tight")
    files_created.append("trajectory.png")

    if pdf:
        fig.savefig(out_path / "trajectory.pdf", dpi=300, bbox_inches="tight")
        files_created.append("trajectory.pdf")

    plt.close(fig)

    summary = {
        "output_dir": str(out_path),
        "metric": metric,
        "files": files_created,
    }
    click.echo(json.dumps(summary, indent=2))
