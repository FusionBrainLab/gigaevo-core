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
    no_frontier_labels: set[str] | None = None,
    sentinel_value: float | None = None,
) -> list[tuple[str, pd.DataFrame]]:
    """Fetch and prepare DataFrames for each run. Returns list of (label, df).

    Args:
        no_frontier_labels: Set of run labels for which frontier (cummax) should
            be suppressed. Useful for adversarial Improver populations where
            fitness is non-monotonic.
        sentinel_value: Exact fitness value used for invalid programs (e.g. -1.0).
            Rows matching this value are removed before computing statistics.
    """
    from gigaevo.utils.dataframes import prepare_iteration_dataframe
    from gigaevo.utils.redis import fetch_evolution_dataframe

    results: list[tuple[str, pd.DataFrame]] = []
    for rc in run_configs:
        config = _build_redis_config(rc, redis_host, redis_port)
        raw_df = asyncio.run(fetch_evolution_dataframe(config, add_stage_results=False))
        if raw_df.empty:
            continue
        label = config.display_label()
        skip_frontier = no_frontier_labels is not None and label in no_frontier_labels
        prepared = prepare_iteration_dataframe(
            raw_df,
            fitness_col=f"metric_{metric}",
            compute_frontier=not skip_frontier,
            sentinel_value=sentinel_value,
        )
        if prepared.empty:
            continue
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


def _aggregate_per_iteration(
    df: pd.DataFrame,
    iteration_col: str,
) -> pd.DataFrame:
    """Aggregate multiple programs per iteration into one row per iteration.

    Groups by iteration and computes:
    - running_mean_fitness: mean across programs in the same iteration
    - running_std_fitness: RMS of per-program stds (preserves variance scale)
    - frontier_fitness: last value (frontier is already cumulative)

    This reduces point-to-point noise and matches the old tools/comparison.py
    behavior that produced smooth, clean curves.
    """
    import numpy as np

    agg_spec: dict = {
        "running_mean_fitness": "mean",
        "running_std_fitness": lambda x: np.sqrt((x**2).mean()),
    }
    if "frontier_fitness" in df.columns:
        agg_spec["frontier_fitness"] = "last"

    grouped = df.groupby(iteration_col).agg(agg_spec).reset_index()
    grouped = grouped.sort_values(iteration_col)
    return grouped


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
@click.option(
    "--no-frontier",
    is_flag=True,
    default=False,
    help="Suppress frontier (cummax) line for ALL runs.",
)
@click.option(
    "--no-frontier-for",
    type=str,
    default=None,
    help="Comma-separated labels to suppress frontier for (e.g. 'D1,D2').",
)
@click.option(
    "--annotate-frontier",
    is_flag=True,
    default=False,
    help="Annotate significant frontier jumps on the plot.",
)
@click.option(
    "--max-annotations",
    type=int,
    default=5,
    help="Max number of frontier annotations per run (default: 5).",
)
@click.option(
    "--paper",
    is_flag=True,
    default=False,
    help="Use publication-quality styling (larger fonts, 300 DPI, colorblind-safe).",
)
@click.option(
    "--sentinel",
    type=float,
    default=None,
    help="Sentinel fitness value for invalid programs (e.g. -1.0). Filtered before plotting.",
)
@click.pass_context
def comparison(
    ctx: click.Context,
    output_dir: str,
    smoothing: str,
    window: int,
    show: bool,
    metric: str,
    no_frontier: bool,
    no_frontier_for: str | None,
    annotate_frontier: bool,
    max_annotations: int,
    paper: bool,
    sentinel: float | None,
) -> None:
    """Plot fitness comparison across runs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gigaevo.utils.plotting import annotate_frontier_points

    # Build set of labels to suppress frontier for
    no_frontier_labels: set[str] | None = None
    if no_frontier:
        no_frontier_labels = {"__ALL__"}  # sentinel: suppress for all
    elif no_frontier_for:
        no_frontier_labels = {s.strip() for s in no_frontier_for.split(",")}

    run_configs = RunResolver.resolve(
        ctx.obj["experiment"],
        ctx.obj["runs"],
        ctx.obj["redis_host"],
        ctx.obj["redis_port"],
    )

    # When --no-frontier applies to ALL runs, pass all labels; otherwise pass specific ones
    actual_no_frontier: set[str] | None = None
    if no_frontier_labels and "__ALL__" in no_frontier_labels:
        # Resolve all labels first, then suppress frontier for all
        all_labels = {
            _build_redis_config(
                rc, ctx.obj["redis_host"], ctx.obj["redis_port"]
            ).display_label()
            for rc in run_configs
        }
        actual_no_frontier = all_labels
    else:
        actual_no_frontier = no_frontier_labels

    prepared_dfs = _fetch_run_data(
        run_configs,
        ctx.obj["redis_host"],
        ctx.obj["redis_port"],
        metric=metric,
        no_frontier_labels=actual_no_frontier,
        sentinel_value=sentinel,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Publication styling
    if paper:
        plt.rcParams.update(
            {
                "font.size": 14,
                "axes.titlesize": 16,
                "axes.labelsize": 14,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 11,
                "lines.linewidth": 2.0,
                "figure.dpi": 300,
            }
        )

    figsize = (8.0, 5.0) if paper else (7.0, 4.5)
    fig, ax = plt.subplots(figsize=figsize)

    # Colorblind-safe palette (Okabe-Ito) for paper, default matplotlib otherwise
    if paper:
        colors = [
            "#0072B2",  # blue
            "#D55E00",  # vermilion
            "#009E73",  # green
            "#CC79A7",  # pink
            "#F0E442",  # yellow
            "#56B4E9",  # sky blue
            "#E69F00",  # orange
            "#000000",  # black
        ]
    else:
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

    iteration_col = "iteration"

    run_labels = []
    for i, (label, df) in enumerate(prepared_dfs):
        color = colors[i % len(colors)]

        # Aggregate to one point per iteration (matches old tools/comparison.py)
        agg = _aggregate_per_iteration(df, iteration_col)
        iters = agg[iteration_col]

        mean_vals = agg["running_mean_fitness"]
        if smoothing != "none" and window > 1:
            mean_vals = _smooth_series(mean_vals, window, smoothing)  # type: ignore[arg-type]

        lw_mean = 2.0 if paper else 1.5
        ax.plot(
            iters, mean_vals, label=f"{label} (mean)", color=color, linewidth=lw_mean
        )

        if "running_std_fitness" in agg.columns:
            std_vals = agg["running_std_fitness"]
            if smoothing != "none" and window > 1:
                std_vals = _smooth_series(std_vals, window, smoothing)  # type: ignore[arg-type]
            import numpy as np

            std_vals = np.maximum(std_vals, 0)
            ax.fill_between(
                iters,
                mean_vals - std_vals,
                mean_vals + std_vals,
                alpha=0.15,
                color=color,
            )

        if "frontier_fitness" in agg.columns:
            lw_best = 1.5 if paper else 1.0
            ax.plot(
                iters,
                agg["frontier_fitness"],
                label=f"{label} (best)",
                color=color,
                linewidth=lw_best,
                linestyle="--",
            )

            if annotate_frontier:
                annotate_frontier_points(
                    ax,
                    iters.values,
                    agg["frontier_fitness"].values,
                    minimize=False,
                    max_annotations=max_annotations,
                    color=color,
                )

        run_labels.append(label)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric.replace("_", " ").title())
    if not paper:
        ax.set_title(f"Evolution Runs Comparison ({metric})")
    ax.legend(fontsize=11 if paper else 8)

    stem = "evolution_runs_comparison"
    dpi = 300 if paper else 300
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            out_path / f"{stem}.{ext}",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )

    if show:
        plt.show()
    plt.close(fig)

    summary = {
        "output_dir": str(out_path),
        "runs": run_labels,
        "smoothing": smoothing,
        "metric": metric,
        "no_frontier": no_frontier,
        "no_frontier_for": no_frontier_for,
        "annotate_frontier": annotate_frontier,
        "paper": paper,
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
@click.option(
    "--sentinel",
    type=float,
    default=None,
    help="Sentinel fitness value for invalid programs (e.g. -1.0). Filtered before plotting.",
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
    sentinel: float | None,
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
        sentinel_value=sentinel,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    iteration_col = "iteration"

    for label, df in prepared_dfs:
        agg = _aggregate_per_iteration(df, iteration_col)
        iters = agg[iteration_col]

        if not no_mean and "running_mean_fitness" in agg.columns:
            ax.plot(
                iters,
                agg["running_mean_fitness"],
                label=f"{label} (mean)",
                linewidth=1.5,
            )

        if (
            not no_std
            and "running_std_fitness" in agg.columns
            and "running_mean_fitness" in agg.columns
        ):
            mean_vals = agg["running_mean_fitness"]
            std_vals = agg["running_std_fitness"]
            ax.fill_between(
                iters,
                mean_vals - std_vals,
                mean_vals + std_vals,
                alpha=0.15,
            )

        if not no_best and "frontier_fitness" in agg.columns:
            ax.plot(
                iters,
                agg["frontier_fitness"],
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


@plot.command("arms-race")
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Output directory for plot files.",
)
@click.option("--metric", default="fitness", help="Metric to plot.")
@click.option(
    "--paired",
    type=str,
    required=True,
    help="Pair spec: 'G_label:D_label' (e.g. 'C1_A:C1_B'). Repeatable via comma.",
)
@click.option(
    "--show-max",
    is_flag=True,
    default=False,
    help="Plot max(G, D) best-overall across each pair.",
)
@click.option(
    "--smoothing",
    type=click.Choice(["none", "ema", "rolling", "gaussian", "lowess", "boxcar"]),
    default="ema",
    help="Smoothing method for mean/std series (default: ema).",
)
@click.option(
    "--window",
    type=int,
    default=10,
    help="Smoothing window size (default: 10).",
)
@click.option(
    "--bands/--no-bands",
    default=True,
    help="Shade +/- 1 std around the mean (default: on).",
)
@click.option(
    "--annotate-frontier",
    is_flag=True,
    default=False,
    help="Annotate significant frontier jumps on the Constructor panel.",
)
@click.option(
    "--max-annotations",
    type=int,
    default=3,
    help="Max frontier annotations per constructor run (default: 3).",
)
@click.option(
    "--paper",
    is_flag=True,
    default=False,
    help="Use publication-quality styling.",
)
@click.option("--show", is_flag=True, default=False, help="Show plot interactively.")
@click.option(
    "--sentinel",
    type=float,
    default=None,
    help="Sentinel fitness value for invalid programs (e.g. -1.0). Filtered before plotting.",
)
@click.pass_context
def arms_race(
    ctx: click.Context,
    output_dir: str,
    metric: str,
    paired: str,
    show_max: bool,
    smoothing: str,
    window: int,
    bands: bool,
    annotate_frontier: bool,
    max_annotations: int,
    paper: bool,
    show: bool,
    sentinel: float | None,
) -> None:
    """Dual-panel arms race plot for adversarial co-evolution.

    Plots Constructor (G) and Improver (D) fitness on stacked panels with shared
    X-axis. Optionally overlays max(G, D) — the best-overall across both populations.

    Example:
        gigaevo plot arms-race -r ... --paired C1_A:C1_B --show-max -o plots/
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from gigaevo.utils.plotting import annotate_frontier_points

    # Parse pair specs
    pairs: list[tuple[str, str]] = []
    for pair_str in paired.split(","):
        parts = pair_str.strip().split(":")
        if len(parts) != 2:
            raise click.BadParameter(
                f"Pair '{pair_str}' must be 'G_label:D_label'", param_hint="--paired"
            )
        pairs.append((parts[0].strip(), parts[1].strip()))

    run_configs = RunResolver.resolve(
        ctx.obj["experiment"],
        ctx.obj["runs"],
        ctx.obj["redis_host"],
        ctx.obj["redis_port"],
    )

    # Suppress frontier for D runs (non-monotonic)
    d_labels = {d for _, d in pairs}
    prepared_dfs = _fetch_run_data(
        run_configs,
        ctx.obj["redis_host"],
        ctx.obj["redis_port"],
        metric=metric,
        no_frontier_labels=d_labels,
        sentinel_value=sentinel,
    )

    # Index by label
    df_by_label: dict[str, pd.DataFrame] = {label: df for label, df in prepared_dfs}

    if paper:
        plt.rcParams.update(
            {
                "font.size": 14,
                "axes.titlesize": 16,
                "axes.labelsize": 14,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 11,
                "lines.linewidth": 2.0,
                "figure.dpi": 300,
            }
        )

    # Colorblind-safe palette
    colors = (
        ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00"]
        if paper
        else ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    iteration_col = "iteration"

    fig, (ax_g, ax_d) = plt.subplots(
        2,
        1,
        figsize=(8.0, 8.0) if paper else (7.0, 7.0),
        sharex=True,
    )

    files_created = []
    pair_labels = []

    def _plot_panel(ax, agg, label, color, *, show_frontier: bool) -> None:
        """Plot mean (smoothed) + optional +/- 1 std band + optional frontier line."""
        iters = agg[iteration_col]
        mean_vals = agg["running_mean_fitness"]
        if smoothing != "none" and window > 1:
            mean_vals = _smooth_series(mean_vals, window, smoothing)  # type: ignore[arg-type]
        ax.plot(
            iters,
            mean_vals,
            label=f"{label} (mean)",
            color=color,
            linewidth=2.0 if paper else 1.5,
        )
        if bands and "running_std_fitness" in agg.columns:
            std_vals = agg["running_std_fitness"]
            if smoothing != "none" and window > 1:
                std_vals = _smooth_series(std_vals, window, smoothing)  # type: ignore[arg-type]
            std_vals = np.maximum(std_vals, 0)
            ax.fill_between(
                iters,
                mean_vals - std_vals,
                mean_vals + std_vals,
                alpha=0.15,
                color=color,
            )
        if show_frontier and "frontier_fitness" in agg.columns:
            ax.plot(
                iters,
                agg["frontier_fitness"],
                label=f"{label} (best)",
                color=color,
                linewidth=1.5 if paper else 1.0,
                linestyle="--",
            )
            if annotate_frontier:
                annotate_frontier_points(
                    ax,
                    iters.values,
                    agg["frontier_fitness"].values,
                    minimize=False,
                    max_annotations=max_annotations,
                    color=color,
                )

    for i, (g_label, d_label) in enumerate(pairs):
        color = colors[i % len(colors)]

        # Constructor (G) panel — top (frontier is monotonic, show it)
        if g_label in df_by_label:
            g_agg = _aggregate_per_iteration(df_by_label[g_label], iteration_col)
            _plot_panel(ax_g, g_agg, g_label, color, show_frontier=True)

        # Improver (D) panel — bottom (no frontier, non-monotonic)
        if d_label in df_by_label:
            d_agg = _aggregate_per_iteration(df_by_label[d_label], iteration_col)
            _plot_panel(ax_d, d_agg, d_label, color, show_frontier=False)

        # max(G, D) overlay on G panel
        if show_max and g_label in df_by_label and d_label in df_by_label:
            g_agg = _aggregate_per_iteration(df_by_label[g_label], iteration_col)
            d_agg = _aggregate_per_iteration(df_by_label[d_label], iteration_col)

            # Align on common iterations
            g_indexed = g_agg.set_index(iteration_col)
            d_indexed = d_agg.set_index(iteration_col)

            g_col = (
                "frontier_fitness"
                if "frontier_fitness" in g_indexed.columns
                else "running_mean_fitness"
            )
            common_iters = g_indexed.index.intersection(d_indexed.index)
            if len(common_iters) > 0:
                g_vals = g_indexed.loc[common_iters, g_col]
                d_vals = d_indexed.loc[common_iters, "running_mean_fitness"]
                max_vals = np.maximum(g_vals.values, d_vals.values)
                ax_g.plot(
                    common_iters,
                    max_vals,
                    label=f"max({g_label},{d_label})",
                    color=color,
                    linewidth=2.5 if paper else 2.0,
                    linestyle=":",
                    alpha=0.8,
                )

        pair_labels.append(f"{g_label}:{d_label}")

    ax_g.set_ylabel(f"Constructor {metric.replace('_', ' ').title()}")
    ax_d.set_ylabel(f"Improver {metric.replace('_', ' ').title()}")
    ax_d.set_xlabel("Iteration")

    if not paper:
        ax_g.set_title("Arms Race — Constructor (G)")
        ax_d.set_title("Arms Race — Improver (D)")

    ax_g.legend(fontsize=11 if paper else 8)
    ax_d.legend(fontsize=11 if paper else 8)

    stem = "arms_race"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            out_path / f"{stem}.{ext}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        files_created.append(f"{stem}.{ext}")

    if show:
        plt.show()
    plt.close(fig)

    summary = {
        "output_dir": str(out_path),
        "pairs": pair_labels,
        "metric": metric,
        "show_max": show_max,
        "paper": paper,
        "files": files_created,
    }
    click.echo(json.dumps(summary, indent=2))
