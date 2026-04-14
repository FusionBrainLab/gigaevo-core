"""Trajectory subcommand -- gen-by-gen fitness trajectory."""

from __future__ import annotations

import json

import click
import redis as redis_lib

from gigaevo.cli.run_resolver import RunResolver


def _fetch_trajectory(
    r: redis_lib.Redis,
    prefix: str,
    metric: str,
) -> list[dict]:
    """Fetch per-gen trajectory data from Redis. Returns list of row dicts."""
    frontier_key = f"{prefix}:metrics:history:program_metrics:valid_frontier_{metric}"
    mean_key = f"{prefix}:metrics:history:program_metrics:valid_gen_{metric}_mean"

    frontier_raw = r.lrange(frontier_key, 0, -1)
    mean_raw = r.lrange(mean_key, 0, -1)

    frontier_by_gen: dict[int, float] = {}
    for raw in frontier_raw:
        try:
            entry = json.loads(raw)
            frontier_by_gen[int(entry["s"])] = float(entry["v"])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass

    mean_by_gen: dict[int, float] = {}
    for raw in mean_raw:
        try:
            entry = json.loads(raw)
            mean_by_gen[int(entry["s"])] = float(entry["v"])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass

    all_gens = sorted(set(frontier_by_gen.keys()) | set(mean_by_gen.keys()))
    running_best: float | None = None
    rows: list[dict] = []
    for gen in all_gens:
        best = frontier_by_gen.get(gen)
        if best is not None:
            if running_best is None or best > running_best:
                running_best = best
        mean = mean_by_gen.get(gen)
        rows.append(
            {
                "Gen": gen,
                "Best": running_best,
                "Mean": mean,
            }
        )
    return rows


@click.command()
@click.option("--tail", type=int, default=None, help="Show only last N generations.")
@click.option(
    "--metric",
    multiple=True,
    default=None,
    help="Metric(s) to display. Repeatable. Auto-discovers from metrics.yaml in --experiment mode.",
)
@click.pass_context
def trajectory(ctx: click.Context, tail: int | None, metric: tuple[str, ...]) -> None:
    """Show gen-by-gen fitness trajectory."""
    formatter = ctx.obj["formatter"]
    experiment = ctx.obj["experiment"]
    runs = ctx.obj["runs"]
    redis_host = ctx.obj["redis_host"]
    redis_port = ctx.obj["redis_port"]

    run_configs = RunResolver.resolve(
        experiment=experiment,
        runs=runs,
        redis_host=redis_host,
        redis_port=redis_port,
    )

    # Auto-discover metrics from run_configs when none explicitly specified
    if not metric:
        seen: set[str] = set()
        discovered: list[str] = []
        for rc in run_configs:
            for m in rc.metric_names:
                if m not in seen:
                    seen.add(m)
                    discovered.append(m)
        metrics_to_show = discovered if discovered else ["fitness"]
    else:
        metrics_to_show = list(metric)

    redis_factory = ctx.obj.get("redis_factory")
    all_rows: list[dict] = []

    for rc in run_configs:
        spec = rc.run_spec
        if redis_factory:
            r = redis_factory(spec.db)
        else:
            r = redis_lib.Redis(
                host=redis_host, port=redis_port, db=spec.db, decode_responses=True
            )
        try:
            for m in metrics_to_show:
                rows = _fetch_trajectory(r, spec.prefix, m)
                for row in rows:
                    if len(run_configs) > 1:
                        row["Label"] = spec.label
                    if len(metrics_to_show) > 1:
                        row["Metric"] = m
                all_rows.extend(rows)
        finally:
            r.close()

    if tail is not None and tail > 0:
        all_rows = all_rows[-tail:]

    columns = ["Gen", "Best", "Mean"]
    if len(run_configs) > 1:
        columns = ["Label"] + columns
    if len(metrics_to_show) > 1:
        columns = ["Metric"] + columns

    formatter.echo(all_rows, columns=columns, title="Trajectory")
