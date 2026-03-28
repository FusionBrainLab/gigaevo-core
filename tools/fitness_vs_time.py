#!/usr/bin/env python3
"""Plot fitness vs wall-clock time for multiple evolution runs.

Reads timestamped fitness history from Redis and plots fitness trajectories
against wall-clock time (hours from start). Useful for comparing throughput
between generational and steady-state engines.

Usage:
    # From experiment.yaml (auto-discovers runs):
    python tools/fitness_vs_time.py --experiment hover/steady-state-validation

    # Manual runs:
    python tools/fitness_vs_time.py \
        --run "chains/hover/full@6:S1 (generational)" \
        --run "chains/hover/full@8:S3 (steady-state)" \
        --output results/fitness_vs_time.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import redis


def fetch_fitness_history(
    host: str, port: int, db: int, prefix: str, metric: str = "valid_frontier_fitness"
) -> list[tuple[float, float]]:
    """Fetch (wall_time, fitness) pairs from Redis history.

    Returns list of (unix_timestamp, fitness_value) sorted by time.
    """
    r = redis.Redis(host=host, port=port, db=db, decode_responses=True)
    key = f"{prefix}:metrics:history:program_metrics:{metric}"
    entries = r.lrange(key, 0, -1)
    if not entries:
        return []

    points: list[tuple[float, float]] = []
    for raw in entries:
        try:
            e = json.loads(raw)
            t = float(e.get("t", 0))
            v = float(e.get("v", 0))
            if t > 0:
                points.append((t, v))
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    points.sort(key=lambda p: p[0])
    return points


def load_runs_from_experiment(experiment: str) -> list[dict]:
    """Load run specs from experiment.yaml."""
    from tools.experiment.manifest import load_manifest

    m = load_manifest(experiment)
    runs = []
    for run in m.runs:
        label = f"{run.label} ({run.condition})"
        runs.append(
            {
                "prefix": run.prefix,
                "db": run.db,
                "label": label,
                "condition": run.condition,
            }
        )
    return runs


def parse_run_spec(spec: str) -> dict:
    """Parse 'prefix@db:label' into dict."""
    if ":" in spec:
        addr, label = spec.rsplit(":", 1)
    else:
        addr, label = spec, spec
    prefix, db_str = addr.split("@")
    return {"prefix": prefix, "db": int(db_str), "label": label, "condition": ""}


def plot_fitness_vs_time(
    run_data: list[tuple[str, list[tuple[float, float]], str]],
    output: str,
    title: str = "Fitness vs Wall-Clock Time",
    metric_label: str = "Soft Fractional Retrieval Coverage",
):
    """Plot fitness trajectories against wall-clock time.

    Args:
        run_data: list of (label, [(unix_time, fitness), ...], condition)
        output: path to save the plot
        title: plot title
        metric_label: y-axis label
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Color by condition
    condition_colors = {"control": "#1f77b4", "treatment": "#ff7f0e"}
    condition_styles = {"control": "-", "treatment": "--"}

    for label, points, condition in run_data:
        if not points:
            continue
        t0 = points[0][0]
        hours = [(t - t0) / 3600 for t, _ in points]
        fitness = [v * 100 for _, v in points]  # Convert to percentage

        color = condition_colors.get(condition, "#333333")
        style = condition_styles.get(condition, "-")
        ax.plot(hours, fitness, style, label=label, color=color, linewidth=2, alpha=0.8)
        # Mark final point
        if hours:
            ax.scatter([hours[-1]], [fitness[-1]], color=color, s=50, zorder=5)

    ax.set_xlabel("Wall-Clock Time (hours)", fontsize=12)
    ax.set_ylabel(f"{metric_label} (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


def fetch_pool_stats(host: str, port: int, pool_name: str) -> dict[str, dict]:
    """Fetch current endpoint pool stats from Redis DB 15."""
    r = redis.Redis(host=host, port=port, db=15, decode_responses=True)
    inflight = r.hgetall(f"llm_pool:{pool_name}:inflight")
    # Find all stats keys for this pool
    stats_keys = r.keys(f"llm_pool:{pool_name}:stats:*")
    endpoint_stats = {}
    for sk in stats_keys:
        data = r.hgetall(sk)
        # Match stats key back to endpoint via inflight hash
        for ep in inflight:
            from hashlib import sha256

            if sha256(ep.encode()).hexdigest()[:12] in sk:
                endpoint_stats[ep] = {
                    "inflight": int(inflight.get(ep, 0)),
                    "requests": int(data.get("requests", 0)),
                    "errors": int(data.get("errors", 0)),
                    "total_latency_ms": float(data.get("total_latency_ms", 0)),
                }
                break
    return endpoint_stats


def print_pool_summary(host: str, port: int):
    """Print current mutation and chain server load."""
    for pool_name, label in [
        ("mutation", "Mutation LLM"),
        ("chain_hover", "Chain LLM"),
    ]:
        stats = fetch_pool_stats(host, port, pool_name)
        if not stats:
            continue
        print(f"\n  {label} Pool:")
        total_req = sum(s["requests"] for s in stats.values())
        total_err = sum(s["errors"] for s in stats.values())
        for ep, s in sorted(stats.items()):
            avg_ms = s["total_latency_ms"] / s["requests"] if s["requests"] else 0
            print(
                f"    {ep}: {s['inflight']} inflight, "
                f"{s['requests']} reqs, {s['errors']} errs, "
                f"avg {avg_ms / 1000:.1f}s"
            )
        print(f"    Total: {total_req} requests, {total_err} errors")


def main():
    parser = argparse.ArgumentParser(description="Plot fitness vs wall-clock time")
    parser.add_argument("--experiment", help="Load runs from experiment.yaml")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Manual run spec: prefix@db:label (repeatable)",
    )
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument(
        "--metric", default="valid_frontier_fitness", help="Redis metric key"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: experiments/<exp>/fitness_vs_time.png)",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--pool-stats", action="store_true", help="Print LLM pool stats"
    )
    args = parser.parse_args()

    # Collect run specs
    runs = []
    if args.experiment:
        runs = load_runs_from_experiment(args.experiment)
    for spec in args.run:
        runs.append(parse_run_spec(spec))

    if not runs:
        parser.error("Specify --experiment or at least one --run")

    # Fetch data
    run_data = []
    for run in runs:
        points = fetch_fitness_history(
            args.redis_host, args.redis_port, run["db"], run["prefix"], args.metric
        )
        gen_count = len(points)
        fitness_str = f"{points[-1][1] * 100:.1f}%" if points else "N/A"
        print(f"  {run['label']}: {gen_count} data points, latest={fitness_str}")
        run_data.append((run["label"], points, run["condition"]))

    # Determine output path
    output = args.output
    if output is None and args.experiment:
        output = f"experiments/{args.experiment}/fitness_vs_time.png"
    elif output is None:
        output = "fitness_vs_time.png"

    title = args.title or (
        f"Fitness vs Time — {args.experiment}" if args.experiment else "Fitness vs Time"
    )

    plot_fitness_vs_time(run_data, output, title=title)

    if args.pool_stats:
        print_pool_summary(args.redis_host, args.redis_port)


if __name__ == "__main__":
    main()
