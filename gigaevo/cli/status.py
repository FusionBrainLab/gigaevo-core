"""Status subcommand -- show current run status from Redis."""

from __future__ import annotations

import click

from gigaevo.cli.output_formatter import OutputFormatter
from gigaevo.cli.run_resolver import RunResolver
from gigaevo.monitoring.experiment_monitor import ExperimentMonitor, RunConfig
from gigaevo.monitoring.snapshot import RunSnapshot


def _snapshot_to_row(snapshot: RunSnapshot) -> dict:
    """Convert a RunSnapshot to a flat dict for OutputFormatter."""
    row: dict = {
        "Label": snapshot.run_spec.label,
        "DB": snapshot.run_spec.db,
        "Gen": snapshot.generation,
    }

    for name, value in snapshot.metrics.items():
        col_name = name.replace("_", " ").title()
        row[col_name] = value

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

    redis_factory = ctx.obj.get("redis_factory")
    monitor = ExperimentMonitor(
        redis_host=redis_host,
        redis_port=redis_port,
        redis_factory=redis_factory,
    )
    snapshots: list[RunSnapshot] = monitor.collect(run_configs)

    rows = [_snapshot_to_row(s) for s in snapshots]
    columns = _build_columns(rows)
    formatter.echo(rows, columns=columns, title="Run Status")
