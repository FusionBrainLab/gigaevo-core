"""Status subcommand -- query Redis for current run status."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import click
import redis as redis_lib
import yaml


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


def _get_metric_names(specs: dict[str, dict]) -> list[str]:
    """Return metric names from specs, primary first, excluding is_valid."""
    primary: list[str] = []
    secondary: list[str] = []
    for name, spec in specs.items():
        if name == "is_valid":
            continue
        if isinstance(spec, dict) and spec.get("is_primary", False):
            primary.append(name)
        else:
            secondary.append(name)
    return primary + secondary


def _get_run_status(
    prefix: str,
    db: int,
    metric_names: list[str] | None = None,
    host: str = "localhost",
    port: int = 6379,
) -> dict:
    """Query Redis for current run status.

    Returns dict with status fields including gen, metrics, program counts,
    invalidity rate, and validator duration stats.
    """
    if metric_names is None:
        metric_names = ["fitness"]

    r = redis_lib.Redis(host=host, port=port, db=db)
    try:
        total_keys = r.dbsize()

        gen = None
        raw_gen = r.hget(f"{prefix}:run_state", "engine:total_generations")
        if raw_gen:
            try:
                gen = int(raw_gen)
            except (ValueError, TypeError):
                pass

        metrics: dict[str, float | None] = {}
        for name in metric_names:
            hist_key = f"{prefix}:metrics:history:program_metrics:valid_frontier_{name}"
            raw = r.lindex(hist_key, -1)
            if raw:
                try:
                    metrics[name] = json.loads(raw)["v"]
                except (KeyError, json.JSONDecodeError, TypeError):
                    metrics[name] = None
            else:
                metrics[name] = None

        total_programs: int | None = None
        valid_programs: int | None = None
        invalid_rate: float | None = None
        raw_total = r.lindex(
            f"{prefix}:metrics:history:program_metrics:programs_total_count", -1
        )
        raw_valid = r.lindex(
            f"{prefix}:metrics:history:program_metrics:programs_valid_count", -1
        )
        if raw_total and raw_valid:
            try:
                total_programs = int(json.loads(raw_total)["v"])
                valid_programs = int(json.loads(raw_valid)["v"])
                if total_programs > 0:
                    invalid_rate = (total_programs - valid_programs) / total_programs
            except (KeyError, json.JSONDecodeError, ValueError, TypeError):
                pass

        validator_mean_s: float | None = None
        validator_max_s: float | None = None
        dur_key = (
            f"{prefix}:metrics:history:dag_runner:dag:internals:"
            "CallValidatorFunction:stage_duration"
        )
        if r.type(dur_key) == b"list":
            recent = r.lrange(dur_key, -20, -1)
            durations: list[float] = []
            for raw_d in recent:
                try:
                    v = json.loads(raw_d)["v"]
                    if v is not None:
                        durations.append(float(v))
                except (KeyError, json.JSONDecodeError, ValueError, TypeError):
                    pass
            if durations:
                validator_mean_s = sum(durations) / len(durations)
                validator_max_s = max(durations)

        done_count = r.scard(f"{prefix}:status:DONE")
        queued_count = r.scard(f"{prefix}:status:QUEUED")
        running_count = r.scard(f"{prefix}:status:RUNNING")
        discarded_count = r.scard(f"{prefix}:status:DISCARDED")

        return {
            "gen": gen,
            "metrics": metrics,
            "best_fitness": metrics.get("fitness"),
            "total_keys": total_keys,
            "total_programs": total_programs,
            "valid_programs": valid_programs,
            "invalid_rate": invalid_rate,
            "validator_mean_s": validator_mean_s,
            "validator_max_s": validator_max_s,
            "done": done_count,
            "queued": queued_count,
            "running": running_count,
            "discarded": discarded_count,
            "error": None,
        }
    except Exception as exc:
        return {
            "gen": None,
            "metrics": {},
            "best_fitness": None,
            "total_keys": None,
            "total_programs": None,
            "valid_programs": None,
            "invalid_rate": None,
            "validator_mean_s": None,
            "validator_max_s": None,
            "done": None,
            "queued": None,
            "running": None,
            "discarded": None,
            "error": str(exc),
        }
    finally:
        r.close()


@click.command()
@click.option("--prefix", required=True, help="Redis key prefix for the run")
@click.option("--db", required=True, type=int, help="Redis database number")
@click.option("--redis-host", default="localhost", help="Redis server hostname")
@click.option("--redis-port", default=6379, type=int, help="Redis server port")
@click.option(
    "--problem-dir",
    default=None,
    help="Path to problem directory for metrics.yaml lookup",
)
def status(
    prefix: str,
    db: int,
    redis_host: str,
    redis_port: int,
    problem_dir: str | None,
) -> None:
    """Query Redis for current run status and output JSON."""
    try:
        metric_names: list[str] = ["fitness"]
        if problem_dir:
            specs = _load_metrics_yaml(problem_dir)
            names = _get_metric_names(specs)
            if names:
                metric_names = names

        result = _get_run_status(
            prefix=prefix,
            db=db,
            metric_names=metric_names,
            host=redis_host,
            port=redis_port,
        )
        click.echo(json.dumps(result))
    except Exception as exc:
        click.echo(json.dumps({"error": str(exc)}), err=True)
        sys.exit(1)
