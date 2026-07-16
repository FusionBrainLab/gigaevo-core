"""RunResolver: bridges CLI --experiment/--run flags to monitoring RunConfig."""

from __future__ import annotations

from pathlib import Path, PurePath
from typing import TYPE_CHECKING

import click

from gigaevo.monitoring.run_spec import RunSpec

if TYPE_CHECKING:
    from gigaevo.database.program_storage import ProgramStorage
    from gigaevo.monitoring.experiment_monitor import RunConfig


def build_readonly_storage(
    spec: RunSpec, redis_host: str, redis_port: int
) -> ProgramStorage:
    """Read-only ProgramStorage for a resolved RunSpec (disk or Redis)."""
    if spec.is_disk:
        from gigaevo.database.factory import build_readonly_disk_storage

        assert spec.path is not None
        return build_readonly_disk_storage(root_dir=spec.path, key_prefix=spec.prefix)
    from gigaevo.database.factory import build_readonly_redis_storage

    return build_readonly_redis_storage(
        host=redis_host, port=redis_port, db=spec.db, key_prefix=spec.prefix
    )


def reject_disk_specs(run_configs: list[RunConfig], command: str) -> None:
    """Raise UsageError when a Redis-only command receives disk-path specs."""
    disk = [rc.run_spec.label for rc in run_configs if rc.run_spec.is_disk]
    if disk:
        raise click.UsageError(
            f"`{command}` requires Redis-backed runs; disk-path specs are "
            f"not supported: {', '.join(disk)}"
        )


def _validate_unique_labels(run_configs: list[RunConfig]) -> None:
    """Require labels to identify exactly one run in multi-run commands."""
    by_label: dict[str, list[RunSpec]] = {}
    for run_config in run_configs:
        spec = run_config.run_spec
        by_label.setdefault(spec.label, []).append(spec)

    duplicates = {label: specs for label, specs in by_label.items() if len(specs) > 1}
    if not duplicates:
        return

    details = []
    for label, specs in sorted(duplicates.items()):
        targets = [
            str(Path(spec.path) / spec.prefix)
            if spec.is_disk and spec.path is not None
            else f"{spec.prefix}@{spec.db}"
            for spec in specs
        ]
        details.append(f"{label!r} ({', '.join(targets)})")
    raise click.UsageError(
        "Run labels must be unique; duplicate label(s): "
        f"{'; '.join(details)}. Add a distinct :LABEL to each -r spec."
    )


def _load_manifest(experiment: str):
    """Lazy-load experiment manifest to avoid import at CLI startup."""
    from gigaevo.experiment.manifest import load_manifest

    return load_manifest(experiment)


def _load_metric_names(problem_name: str) -> list[str]:
    """Load metric names from problems/{problem_name}/metrics.yaml.

    Returns primary metrics first, excluding is_valid. Falls back to ["fitness"].
    """
    from gigaevo.cli.log_paths import problem_metrics_path

    path = problem_metrics_path(problem_name)
    if not path.exists():
        return ["fitness"]
    import yaml

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
            configs = RunResolver._resolve_from_runs(runs, redis_host, redis_port)
        else:
            assert experiment is not None  # guaranteed by checks above
            configs = RunResolver._resolve_from_experiment(experiment)
        _validate_unique_labels(configs)
        return configs

    @staticmethod
    def _resolve_from_runs(
        runs: list[str] | tuple[str, ...],
        redis_host: str = "localhost",
        redis_port: int = 6379,
    ) -> list[RunConfig]:
        from gigaevo.monitoring.experiment_monitor import RunConfig

        configs = []
        for raw in runs:
            try:
                spec = RunSpec.parse(raw)
            except ValueError as exc:
                raise click.BadParameter(str(exc), param_hint="--run/-r") from exc
            if spec.is_disk:
                spec = RunResolver._resolve_disk(spec)
            elif spec.needs_prefix:
                spec = RunResolver._autodiscover_prefix(spec, redis_host, redis_port)
            configs.append(RunConfig(run_spec=spec))
        return configs

    @staticmethod
    def _resolve_disk(spec: RunSpec) -> RunSpec:
        """Resolve a disk-path RunSpec by locating the storage prefix directory.

        Accepts either the storage root (containing one prefix directory)
        or the prefix directory itself (containing ``programs/``). A Hydra
        output directory containing ``storage/`` is accepted as a convenience.
        """
        assert spec.path is not None
        target = Path(spec.path).expanduser()
        if not target.is_dir():
            raise click.UsageError(f"Disk storage path is not a directory: {target}")
        if (target / "storage").is_dir():
            target = target / "storage"
        if (target / "programs").is_dir():
            base = target
        else:
            candidates = sorted(
                d for d in target.iterdir() if (d / "programs").is_dir()
            )
            if not candidates:
                raise click.UsageError(
                    f"No program storage found under {target} "
                    f"(expected a <prefix>/programs/ directory)"
                )
            if len(candidates) > 1:
                names = ", ".join(d.name for d in candidates)
                raise click.UsageError(
                    f"Multiple storage prefixes under {target}: {names}. "
                    f"Point directly at one prefix directory."
                )
            base = candidates[0]
        auto_label = spec.label == PurePath(spec.path).name
        return RunSpec(
            prefix=base.name,
            db=-1,
            label=base.name if auto_label else spec.label,
            path=str(base.parent),
        )

    @staticmethod
    def _autodiscover_prefix(
        spec: RunSpec, redis_host: str, redis_port: int
    ) -> RunSpec:
        """Resolve a prefix-less RunSpec from recognized keys in Redis."""
        from redis.exceptions import RedisError

        from gigaevo.cli.inspect_cmd import discover_prefixes

        try:
            prefixes = discover_prefixes(redis_host, redis_port, spec.db)
        except RedisError as exc:
            raise click.ClickException(
                f"Cannot inspect Redis DB {spec.db} at {redis_host}:{redis_port}: {exc}"
            ) from exc
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
        from gigaevo.monitoring.experiment_monitor import RunConfig

        manifest = _load_manifest(experiment)
        configs = []
        for run in manifest.contract.runs:
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
