"""Collect subcommand -- fetch top programs and summary stats from Redis."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

import click
import redis as redis_lib
import yaml

from gigaevo.database.redis_program_storage import (
    RedisProgramStorage,
    RedisProgramStorageConfig,
)


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


async def _fetch_programs(host: str, port: int, db: int, prefix: str) -> list:
    """Fetch all programs from Redis using RedisProgramStorage."""
    url = f"redis://{host}:{port}/{db}"
    storage = RedisProgramStorage(
        RedisProgramStorageConfig(
            redis_url=url,
            key_prefix=prefix,
            max_connections=50,
            connection_pool_timeout=30.0,
            health_check_interval=60,
            read_only=True,
        )
    )
    try:
        return await storage.get_all()
    finally:
        await storage.close()


def _collect_data(
    prefix: str,
    db: int,
    host: str,
    port: int,
    top_n: int,
    metric: str,
    higher_is_better: bool,
) -> dict:
    """Collect experiment data from Redis and return structured result."""
    programs = asyncio.run(_fetch_programs(host, port, db, prefix))

    valid_programs = [
        p for p in programs if metric in p.metrics and p.metrics[metric] is not None
    ]

    valid_programs.sort(
        key=lambda p: p.metrics[metric],
        reverse=higher_is_better,
    )

    top_programs_list = []
    for p in valid_programs[:top_n]:
        entry: dict = {
            "id": p.id,
            "fitness": p.metrics.get(metric),
            "metrics": p.metrics,
            "generation": p.generation,
            "state": p.state.value,
        }
        top_programs_list.append(entry)

    r = redis_lib.Redis(host=host, port=port, db=db)
    try:
        r.scard(f"{prefix}:status:DONE")
        discarded_count = r.scard(f"{prefix}:status:DISCARDED")

        fitness_trajectory: list[float] = []
        hist_key = f"{prefix}:metrics:history:program_metrics:valid_frontier_{metric}"
        raw_entries = r.lrange(hist_key, 0, -1)
        for raw in raw_entries:
            try:
                entry_data = json.loads(raw)
                val = entry_data.get("v")
                if val is not None:
                    fitness_trajectory.append(float(val))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        run_state_raw = r.hgetall(f"{prefix}:run_state")
        run_state: dict[str, str] = {}
        for k, v in run_state_raw.items():
            try:
                run_state[k.decode() if isinstance(k, bytes) else str(k)] = (
                    v.decode() if isinstance(v, bytes) else str(v)
                )
            except (UnicodeDecodeError, AttributeError):
                pass

        gen = None
        raw_gen = run_state.get("engine:total_generations")
        if raw_gen:
            try:
                gen = int(raw_gen)
            except (ValueError, TypeError):
                pass
    finally:
        r.close()

    best_fitness: float | None = None
    if valid_programs:
        best_fitness = valid_programs[0].metrics.get(metric)

    return {
        "best_fitness": best_fitness,
        "valid_programs": len(valid_programs),
        "generations": gen,
        "top_programs": top_programs_list,
        "discarded_programs": discarded_count,
        "total_programs": len(programs),
        "fitness_trajectory": fitness_trajectory,
        "run_state": run_state,
        "metric": metric,
        "higher_is_better": higher_is_better,
    }


@click.command()
@click.option("--prefix", required=True, help="Redis key prefix for the run")
@click.option("--db", required=True, type=int, help="Redis database number")
@click.option("--redis-host", default="localhost", help="Redis server hostname")
@click.option("--redis-port", default=6379, type=int, help="Redis server port")
@click.option("--top-n", default=5, type=int, help="Number of top programs to return")
@click.option(
    "--problem-dir",
    default=None,
    help="Path to problem directory for metrics.yaml",
)
@click.option("--metric", default="fitness", help="Metric to rank programs by")
def collect(
    prefix: str,
    db: int,
    redis_host: str,
    redis_port: int,
    top_n: int,
    problem_dir: str | None,
    metric: str,
) -> None:
    """Fetch top programs and summary statistics from a run."""
    try:
        higher = True
        if problem_dir:
            specs = _load_metrics_yaml(problem_dir)
            higher = _higher_is_better(specs, metric)

        result = _collect_data(
            prefix=prefix,
            db=db,
            host=redis_host,
            port=redis_port,
            top_n=top_n,
            metric=metric,
            higher_is_better=higher,
        )
        click.echo(json.dumps(result))
    except Exception as exc:
        click.echo(json.dumps({"error": str(exc)}), err=True)
        sys.exit(1)
