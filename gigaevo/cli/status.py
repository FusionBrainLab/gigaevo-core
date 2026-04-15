"""Status subcommand -- show current run status from Redis."""

from __future__ import annotations

from pathlib import Path

import click
import yaml

from gigaevo.cli.output_formatter import OutputFormatter
from gigaevo.cli.run_resolver import RunResolver
from gigaevo.monitoring.experiment_monitor import ExperimentMonitor, RunConfig
from gigaevo.monitoring.snapshot import RunSnapshot


def _load_metric_specs(experiment: str | None) -> dict[str, dict]:
    """Load metrics.yaml specs for all problems in an experiment manifest.

    Merges specs from all unique problem_names. Falls back to empty dict
    if no metrics.yaml found or not in --experiment mode.
    """
    if not experiment:
        return {}
    try:
        from gigaevo.cli.run_resolver import _load_manifest

        manifest = _load_manifest(experiment)
        all_specs: dict[str, dict] = {}
        seen: set[str] = set()
        for run in manifest.contract.runs:
            if run.problem_name not in seen:
                seen.add(run.problem_name)
                path = Path("problems") / run.problem_name / "metrics.yaml"
                if path.exists():
                    with open(path) as f:
                        data = yaml.safe_load(f)
                    if isinstance(data, dict):
                        all_specs.update(data.get("specs", {}))
        return all_specs
    except Exception:
        return {}


def _format_metric_value(value: float | None, name: str, specs: dict[str, dict]) -> str:
    """Format a metric value according to its metrics.yaml spec.

    - None: display as "?"
    - sentinel_value: display as "N/A"
    - upper_bound == 1.0: display as percentage (value * 100)
    - decimals: control decimal places
    - Default: 3 decimal places
    """
    if value is None:
        return "?"
    spec = specs.get(name, {})
    sentinel = spec.get("sentinel_value")
    if sentinel is not None and value == sentinel:
        return "N/A"
    decimals = spec.get("decimals", 3)
    upper_bound = spec.get("upper_bound")
    if upper_bound is not None and upper_bound == 1.0:
        pct_decimals = max(0, decimals - 2)
        return f"{value * 100:.{pct_decimals}f}%"
    return f"{value:.{decimals}f}"


def _snapshot_to_row(
    snapshot: RunSnapshot, metric_specs: dict[str, dict] | None = None
) -> dict:
    """Convert a RunSnapshot to a flat dict for OutputFormatter."""
    row: dict = {
        "Label": snapshot.run_spec.label,
        "DB": snapshot.run_spec.db,
        "Gen": snapshot.generation,
    }

    specs = metric_specs or {}
    for name, value in snapshot.metrics.items():
        col_name = name.replace("_", " ").title()
        row[col_name] = _format_metric_value(value, name, specs)

    invalid_rate = snapshot.invalid_rate
    if invalid_rate is not None:
        row["Invalid%"] = f"{invalid_rate:.0%}"
    else:
        row["Invalid%"] = None

    if snapshot.validator_mean_s is not None and snapshot.validator_max_s is not None:
        row["Val dur(s)"] = (
            f"{snapshot.validator_mean_s:.0f}/{snapshot.validator_max_s:.0f}"
        )
    else:
        row["Val dur(s)"] = None

    row["Keys"] = snapshot.total_keys

    if snapshot.pid is not None:
        row["PID"] = snapshot.pid
        row["Status"] = "ALIVE" if snapshot.pid_alive else "DEAD"
    else:
        row["PID"] = None
        row["Status"] = None

    if snapshot.error is not None:
        row["Error"] = snapshot.error

    return row


def _build_columns(rows: list[dict]) -> list[str]:
    """Determine column order from available data."""
    base = ["Label", "DB", "Gen"]
    metric_cols = []
    for row in rows:
        for key in row:
            if key not in base and key not in (
                "Invalid%",
                "Val dur(s)",
                "Keys",
                "PID",
                "Status",
                "Error",
            ):
                if key not in metric_cols:
                    metric_cols.append(key)
    tail = ["Invalid%", "Val dur(s)", "Keys"]
    if any(row.get("PID") is not None for row in rows):
        tail.extend(["PID", "Status"])
    if any("Error" in row for row in rows):
        tail.append("Error")
    return base + metric_cols + tail


@click.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show current run status from Redis."""
    formatter: OutputFormatter = ctx.obj["formatter"]
    experiment: str | None = ctx.obj["experiment"]
    runs: tuple[str, ...] = ctx.obj["runs"]
    redis_host: str = ctx.obj["redis_host"]
    redis_port: int = ctx.obj["redis_port"]

    run_configs: list[RunConfig] = RunResolver.resolve(
        experiment=experiment,
        runs=runs,
        redis_host=redis_host,
        redis_port=redis_port,
    )

    metric_specs = _load_metric_specs(experiment)

    redis_factory = ctx.obj.get("redis_factory")
    monitor = ExperimentMonitor(
        redis_host=redis_host,
        redis_port=redis_port,
        redis_factory=redis_factory,
    )
    snapshots: list[RunSnapshot] = monitor.collect(run_configs)

    rows = [_snapshot_to_row(s, metric_specs) for s in snapshots]
    columns = _build_columns(rows)
    formatter.echo(rows, columns=columns, title="Run Status")
