"""Plot subcommand -- fitness trajectory visualization."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click
import matplotlib
import redis as redis_lib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load_metrics_yaml(problem_dir: str) -> dict[str, dict]:
    """Load metrics.yaml from a problem directory, return {metric_name: spec_dict}."""
    path = Path(problem_dir) / "metrics.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return {}
    return data.get("specs", {})


def _higher_is_better(specs: dict[str, dict], metric: str) -> bool:
    """Determine if higher values are better for a given metric."""
    spec = specs.get(metric, {})
    if not isinstance(spec, dict):
        return True
    return spec.get("higher_is_better", True)


def _fetch_trajectory_data(
    host: str, port: int, db: int, prefix: str, metric: str
) -> tuple[list[dict], list[dict]]:
    """Fetch gen-mean and frontier entries from Redis.

    Returns (gen_mean_entries, frontier_entries) as lists of parsed JSON dicts.
    """
    r = redis_lib.Redis(host=host, port=port, db=db)
    try:
        gen_mean_entries: list[dict] = []
        raw_mean = r.lrange(
            f"{prefix}:metrics:history:program_metrics:valid_gen_{metric}_mean", 0, -1
        )
        for raw in raw_mean:
            try:
                gen_mean_entries.append(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                pass

        frontier_entries: list[dict] = []
        raw_frontier = r.lrange(
            f"{prefix}:metrics:history:program_metrics:valid_frontier_{metric}", 0, -1
        )
        for raw in raw_frontier:
            try:
                frontier_entries.append(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                pass

        return gen_mean_entries, frontier_entries
    finally:
        r.close()


def _compute_per_iteration_stats(
    gen_mean_entries: list[dict],
    frontier_entries: list[dict],
    higher_is_better: bool,
) -> tuple[list[int], list[float], list[float], list[float]]:
    """Compute per-iteration best, mean, and std.

    Returns (iterations, best_values, mean_values, std_values).
    """
    gen_vals: dict[int, list[float]] = {}
    for entry in gen_mean_entries:
        g = entry.get("s")
        v = entry.get("v")
        if g is None or v is None:
            continue
        try:
            gen_vals.setdefault(int(g), []).append(float(v))
        except (ValueError, TypeError):
            pass

    frontier_points: list[tuple[int, float]] = []
    for entry in frontier_entries:
        s = entry.get("s")
        v = entry.get("v")
        if s is not None and v is not None:
            try:
                frontier_points.append((int(s), float(v)))
            except (ValueError, TypeError):
                pass
    frontier_points.sort(key=lambda x: x[0])

    sorted_gens = sorted(gen_vals.keys())
    if not sorted_gens:
        return [], [], [], []

    frontier_at_gen: dict[int, float] = {}
    running_best: float | None = None
    fi = 0
    for gen in sorted_gens:
        while fi < len(frontier_points) and frontier_points[fi][0] <= gen:
            v = frontier_points[fi][1]
            if running_best is None:
                running_best = v
            elif higher_is_better and v > running_best:
                running_best = v
            elif not higher_is_better and v < running_best:
                running_best = v
            fi += 1
        if running_best is not None:
            frontier_at_gen[gen] = running_best

    iterations: list[int] = []
    best_values: list[float] = []
    mean_values: list[float] = []
    std_values: list[float] = []

    for gen in sorted_gens:
        vals = gen_vals[gen]
        mean_val = vals[-1]

        import statistics

        std_val = statistics.stdev(vals) if len(vals) > 1 else 0.0

        best_val = frontier_at_gen.get(gen)
        if best_val is None:
            continue

        iterations.append(gen)
        best_values.append(best_val)
        mean_values.append(mean_val)
        std_values.append(std_val)

    return iterations, best_values, mean_values, std_values


@click.command()
@click.option("--prefix", required=True, help="Redis key prefix for the run")
@click.option("--db", required=True, type=int, help="Redis database number")
@click.option("--redis-host", default="localhost", help="Redis server hostname")
@click.option("--redis-port", default=6379, type=int, help="Redis server port")
@click.option("--metric", default="fitness", help="Which metric to plot")
@click.option("-o", "--output", default=None, help="Custom output file path")
@click.option("--pdf", is_flag=True, help="Output PDF instead of PNG")
@click.option("--no-best", is_flag=True, help="Hide best fitness line")
@click.option("--no-mean", is_flag=True, help="Hide mean fitness line")
@click.option("--no-std", is_flag=True, help="Hide std deviation band")
@click.option("--problem-dir", default=None, help="Path for metrics.yaml")
def plot(
    prefix: str,
    db: int,
    redis_host: str,
    redis_port: int,
    metric: str,
    output: str | None,
    pdf: bool,
    no_best: bool,
    no_mean: bool,
    no_std: bool,
    problem_dir: str | None,
) -> None:
    """Plot fitness trajectory from a run as PNG or PDF."""
    try:
        higher = True
        if problem_dir:
            specs = _load_metrics_yaml(problem_dir)
            higher = _higher_is_better(specs, metric)

        gen_mean_entries, frontier_entries = _fetch_trajectory_data(
            redis_host, redis_port, db, prefix, metric
        )

        if not gen_mean_entries and not frontier_entries:
            click.echo(
                json.dumps({"error": f"No trajectory data for prefix={prefix}, metric={metric}"}),
                err=True,
            )
            sys.exit(1)

        iterations, best_values, mean_values, std_values = _compute_per_iteration_stats(
            gen_mean_entries, frontier_entries, higher
        )

        if not iterations:
            click.echo(
                json.dumps({"error": "No valid iteration data after processing"}),
                err=True,
            )
            sys.exit(1)

        fig, ax = plt.subplots(figsize=(10, 6))

        if not no_best:
            ax.plot(
                iterations,
                best_values,
                linewidth=2.0,
                color="#1f77b4",
                label=f"Best {metric}",
                zorder=3,
            )

        if not no_mean:
            ax.plot(
                iterations,
                mean_values,
                linewidth=1.5,
                color="#ff7f0e",
                linestyle="--",
                label=f"Mean {metric}",
                zorder=2,
            )

        if not no_std and std_values:
            import numpy as np

            mean_arr = np.array(mean_values)
            std_arr = np.array(std_values)
            ax.fill_between(
                iterations,
                mean_arr - std_arr,
                mean_arr + std_arr,
                alpha=0.2,
                color="#ff7f0e",
                label="Std dev",
                zorder=1,
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"{prefix} - {metric} trajectory")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()

        ext = "pdf" if pdf else "png"
        if output:
            output_path = output
        else:
            safe_prefix = prefix.replace("/", "_").replace("\\", "_")
            output_path = f"{safe_prefix}_{metric}_trajectory.{ext}"

        output_path = os.path.abspath(output_path)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        click.echo(
            json.dumps({
                "plot": output_path,
                "format": ext,
                "iterations": len(iterations),
            })
        )
    except Exception as exc:
        click.echo(json.dumps({"error": str(exc)}), err=True)
        sys.exit(1)
