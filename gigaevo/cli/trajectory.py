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
@click.option("--metric", default="fitness", help="Metric to display.")
@click.pass_context
def trajectory(ctx: click.Context, tail: int | None, metric: str) -> None:
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
            rows = _fetch_trajectory(r, spec.prefix, metric)
            if len(run_configs) > 1:
                for row in rows:
                    row["Label"] = spec.label
            all_rows.extend(rows)
        finally:
            r.close()

    if tail is not None and tail > 0:
        all_rows = all_rows[-tail:]

    columns = ["Gen", "Best", "Mean"]
    if len(run_configs) > 1:
        columns = ["Label"] + columns

    formatter.echo(all_rows, columns=columns, title="Trajectory")
