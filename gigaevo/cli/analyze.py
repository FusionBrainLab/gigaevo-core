"""Analyze subcommand -- comprehensive experiment analysis."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

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


def _build_top_section(
    programs: list,
    metric: str,
    higher: bool,
    top_n: int,
) -> list[dict]:
    """Build the top-N programs section."""
    valid = [
        p for p in programs
        if metric in p.metrics and p.metrics[metric] is not None
    ]
    valid.sort(key=lambda p: p.metrics[metric], reverse=higher)

    result: list[dict] = []
    for p in valid[:top_n]:
        entry: dict = {
            "id": p.id,
            "fitness": p.metrics.get(metric),
            "metrics": p.metrics,
            "generation": p.generation,
            "state": p.state.value,
        }
        result.append(entry)
    return result


def _build_stats_section(
    programs: list,
    metric: str,
    higher: bool,
    host: str,
    port: int,
    db: int,
    prefix: str,
) -> dict:
    """Build the stats section with summary statistics."""
    valid = [
        p for p in programs
        if metric in p.metrics and p.metrics[metric] is not None
    ]
    fitness_values = [p.metrics[metric] for p in valid]

    r = redis_lib.Redis(host=host, port=port, db=db)
    try:
        gen = None
        raw_gen = r.hget(f"{prefix}:run_state", "engine:total_generations")
        if raw_gen:
            try:
                gen = int(raw_gen)
            except (ValueError, TypeError):
                pass

        done_count = r.scard(f"{prefix}:status:DONE")
        discarded_count = r.scard(f"{prefix}:status:DISCARDED")
    finally:
        r.close()

    best_fitness: float | None = None
    mean_fitness: float | None = None
    std_fitness: float | None = None

    if fitness_values:
        if higher:
            best_fitness = max(fitness_values)
        else:
            best_fitness = min(fitness_values)
        mean_fitness = sum(fitness_values) / len(fitness_values)
        if len(fitness_values) > 1:
            import statistics
            std_fitness = statistics.stdev(fitness_values)
        else:
            std_fitness = 0.0

    return {
        "generations": gen,
        "valid_programs": len(valid),
        "discarded_programs": discarded_count,
        "total_programs": len(programs),
        "best_fitness": best_fitness,
        "mean_fitness": mean_fitness,
        "std_fitness": std_fitness,
    }


def _build_convergence_section(
    host: str, port: int, db: int, prefix: str, metric: str, higher: bool
) -> dict:
    """Build the convergence analysis section."""
    r = redis_lib.Redis(host=host, port=port, db=db)
    try:
        frontier_entries: list[dict] = []
        raw_frontier = r.lrange(
            f"{prefix}:metrics:history:program_metrics:valid_frontier_{metric}", 0, -1
        )
        for raw in raw_frontier:
            try:
                frontier_entries.append(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                pass
    finally:
        r.close()

    if not frontier_entries:
        return {
            "generation_of_best": None,
            "improvement_rate": None,
            "plateau_detected": False,
            "plateau_length": 0,
        }

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

    if not frontier_points:
        return {
            "generation_of_best": None,
            "improvement_rate": None,
            "plateau_detected": False,
            "plateau_length": 0,
        }

    improvement_points: list[tuple[int, float]] = []
    running_best: float | None = None
    for s, v in frontier_points:
        if running_best is None:
            improvement_points.append((s, v))
            running_best = v
        elif higher and v > running_best:
            improvement_points.append((s, v))
            running_best = v
        elif not higher and v < running_best:
            improvement_points.append((s, v))
            running_best = v

    last_gen, last_val = improvement_points[-1]
    first_gen, first_val = improvement_points[0]

    gen_span = last_gen - first_gen
    improvement_rate: float | None = None
    if gen_span > 0:
        improvement_rate = (last_val - first_val) / gen_span

    final_gen = frontier_points[-1][0]
    plateau_length = final_gen - last_gen
    plateau_detected = plateau_length > max(10, final_gen * 0.2)

    return {
        "generation_of_best": last_gen,
        "improvement_rate": improvement_rate,
        "plateau_detected": plateau_detected,
        "plateau_length": plateau_length,
    }


def _run_analysis(
    prefix: str,
    db: int,
    host: str,
    port: int,
    top_n: int,
    metric: str,
    higher: bool,
    section: str | None,
) -> dict:
    """Run full analysis and return result dict."""
    programs = asyncio.run(_fetch_programs(host, port, db, prefix))

    if section == "top":
        return {"top": _build_top_section(programs, metric, higher, top_n)}
    elif section == "stats":
        return {"stats": _build_stats_section(programs, metric, higher, host, port, db, prefix)}
    elif section == "convergence":
        return {"convergence": _build_convergence_section(host, port, db, prefix, metric, higher)}

    return {
        "top": _build_top_section(programs, metric, higher, top_n),
        "stats": _build_stats_section(programs, metric, higher, host, port, db, prefix),
        "convergence": _build_convergence_section(host, port, db, prefix, metric, higher),
    }


def _parse_compare_spec(spec: str) -> tuple[str, int]:
    """Parse a compare spec 'prefix@db' into (prefix, db)."""
    if "@" not in spec:
        raise ValueError(f"Compare spec must be prefix@db, got: {spec!r}")
    prefix, db_str = spec.rsplit("@", 1)
    return prefix, int(db_str)


@click.command()
@click.option("--prefix", required=True, help="Redis key prefix for the run")
@click.option("--db", required=True, type=int, help="Redis database number")
@click.option("--redis-host", default="localhost", help="Redis server hostname")
@click.option("--redis-port", default=6379, type=int, help="Redis server port")
@click.option("--top-n", default=5, type=int, help="Number of top programs")
@click.option("--metric", default="fitness", help="Which metric to analyze")
@click.option("--problem-dir", default=None, help="Path for metrics.yaml")
@click.option(
    "--section",
    default=None,
    type=click.Choice(["top", "stats", "convergence"]),
    help="Extract single section",
)
@click.option(
    "--compare",
    multiple=True,
    help='Cross-run comparison specs as "prefix@db"',
)
def analyze(
    prefix: str,
    db: int,
    redis_host: str,
    redis_port: int,
    top_n: int,
    metric: str,
    problem_dir: str | None,
    section: str | None,
    compare: tuple[str, ...],
) -> None:
    """Comprehensive experiment analysis with top/stats/convergence sections."""
    try:
        higher = True
        if problem_dir:
            specs = _load_metrics_yaml(problem_dir)
            higher = _higher_is_better(specs, metric)

        if compare:
            runs_results: list[dict] = []

            primary = _run_analysis(
                prefix, db, redis_host, redis_port, top_n, metric, higher, section
            )
            primary["name"] = f"{prefix}@{db}"
            runs_results.append(primary)

            for spec in compare:
                try:
                    cmp_prefix, cmp_db = _parse_compare_spec(spec)
                    cmp_result = _run_analysis(
                        cmp_prefix, cmp_db, redis_host, redis_port, top_n, metric, higher, section
                    )
                    cmp_result["name"] = spec
                    runs_results.append(cmp_result)
                except Exception as exc:
                    runs_results.append({"name": spec, "error": str(exc)})

            best_run: str | None = None
            best_fitness: float | None = None
            for run in runs_results:
                stats = run.get("stats", {})
                bf = stats.get("best_fitness")
                if bf is not None:
                    if best_fitness is None:
                        best_fitness = bf
                        best_run = run.get("name")
                    elif higher and bf > best_fitness:
                        best_fitness = bf
                        best_run = run.get("name")
                    elif not higher and bf < best_fitness:
                        best_fitness = bf
                        best_run = run.get("name")

            comparison_info: dict = {"best_run": best_run}
            if len(runs_results) >= 2:
                fitnesses = [
                    r.get("stats", {}).get("best_fitness")
                    for r in runs_results
                    if r.get("stats", {}).get("best_fitness") is not None
                ]
                if len(fitnesses) >= 2:
                    comparison_info["metric_diff"] = max(fitnesses) - min(fitnesses)

            result = {"runs": runs_results, "comparison": comparison_info}
        else:
            result = _run_analysis(
                prefix, db, redis_host, redis_port, top_n, metric, higher, section
            )

        click.echo(json.dumps(result))
    except Exception as exc:
        click.echo(json.dumps({"error": str(exc)}), err=True)
        sys.exit(1)
