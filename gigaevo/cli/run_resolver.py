"""RunResolver: bridges CLI --experiment/--run flags to monitoring RunConfig."""

from __future__ import annotations

from pathlib import Path

import click
import yaml

from gigaevo.monitoring.experiment_monitor import RunConfig
from gigaevo.monitoring.run_spec import RunSpec


def _load_manifest(experiment: str):
    """Lazy-load experiment manifest to avoid import at CLI startup."""
    from gigaevo.monitoring.manifest import load_manifest

    return load_manifest(experiment)


def _load_metric_names(problem_name: str) -> list[str]:
    """Load metric names from problems/{problem_name}/metrics.yaml.

    Returns primary metrics first, excluding is_valid. Falls back to ["fitness"].
    """
    path = Path("problems") / problem_name / "metrics.yaml"
    if not path.exists():
        return ["fitness"]
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return ["fitness"]
    specs = data.get("specs", {})
    if not specs:
        return ["fitness"]

    primary: list[str] = []
    secondary: list[str] = []
    for name, spec in specs.items():
        if name == "is_valid":
            continue
        if isinstance(spec, dict) and spec.get("is_primary", False):
            primary.append(name)
        else:
            secondary.append(name)
    result = primary + secondary
    return result if result else ["fitness"]


class RunResolver:
    """Resolve CLI flags into list[RunConfig] for the monitoring library."""

    @staticmethod
    def resolve(
        experiment: str | None,
        runs: list[str] | tuple[str, ...],
        redis_host: str,
        redis_port: int,
    ) -> list[RunConfig]:
        """Resolve --experiment or --run flags into RunConfig objects.

        Raises click.UsageError if neither or both are provided.
        """
        has_experiment = experiment is not None and experiment != ""
        has_runs = len(runs) > 0

        if has_experiment and has_runs:
            raise click.UsageError("Use --experiment or --run, not both")

        if not has_experiment and not has_runs:
            raise click.UsageError("Provide --experiment or at least one --run")

        if has_runs:
            return RunResolver._resolve_from_runs(runs, redis_host, redis_port)
        assert experiment is not None  # guaranteed by checks above
        return RunResolver._resolve_from_experiment(experiment)

    @staticmethod
    def _resolve_from_runs(
        runs: list[str] | tuple[str, ...],
        redis_host: str = "localhost",
        redis_port: int = 6379,
    ) -> list[RunConfig]:
        configs = []
        for raw in runs:
            spec = RunSpec.parse(raw)
            if spec.needs_prefix:
                spec = RunResolver._autodiscover_prefix(spec, redis_host, redis_port)
            configs.append(RunConfig(run_spec=spec))
        return configs

    @staticmethod
    def _autodiscover_prefix(
        spec: RunSpec, redis_host: str, redis_port: int
    ) -> RunSpec:
        """Resolve a prefix-less RunSpec by finding the instance lock in Redis."""
        from gigaevo.cli.inspect_cmd import discover_prefixes

        prefixes = discover_prefixes(redis_host, redis_port, spec.db)
        if len(prefixes) == 0:
            raise click.UsageError(f"No experiment prefix found in Redis DB {spec.db}")
        if len(prefixes) > 1:
            raise click.UsageError(
                f"Multiple prefixes in DB {spec.db}: {', '.join(prefixes)}. "
                f"Specify explicitly with prefix@{spec.db}"
            )
        prefix = prefixes[0]
        return RunSpec(
            prefix=prefix,
            db=spec.db,
            label=f"{prefix}@{spec.db}",
        )

    @staticmethod
    def _resolve_from_experiment(experiment: str) -> list[RunConfig]:
        manifest = _load_manifest(experiment)
        configs = []
        for run in manifest.runs:
            spec = RunSpec(prefix=run.prefix, db=run.db, label=run.label)
            metric_names = _load_metric_names(run.problem_name)
            configs.append(
                RunConfig(
                    run_spec=spec,
                    metric_names=metric_names,
                    pid=run.pid,
                )
            )
        return configs
