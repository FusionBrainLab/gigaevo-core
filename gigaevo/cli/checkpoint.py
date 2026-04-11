"""Checkpoint subcommand -- composite status + notify in one shot."""

from __future__ import annotations

import click

from gigaevo.cli.run_resolver import RunResolver
from gigaevo.monitoring.experiment_monitor import ExperimentMonitor


def _snapshot_to_row(snap) -> dict:
    """Convert a RunSnapshot to a display row."""
    row: dict = {
        "Label": snap.run_spec.label,
        "DB": snap.run_spec.db,
        "Gen": snap.generation,
    }
    if snap.metrics:
        for key, val in snap.metrics.items():
            col_name = key.replace("_", " ").title()
            row[col_name] = val
    row["Invalid%"] = snap.invalid_rate
    row["Total"] = snap.total_programs
    row["Valid"] = snap.valid_programs
    return row


def _build_columns(rows: list[dict]) -> list[str]:
    """Build column list from row keys, preserving order."""
    cols: list[str] = []
    for row in rows:
        for key in row:
            if key not in cols:
                cols.append(key)
    return cols


@click.command()
@click.option(
    "--no-notify",
    is_flag=True,
    default=False,
    help="Skip notification dispatch (status only).",
)
@click.option(
    "--no-plots",
    is_flag=True,
    default=False,
    help="Skip plot generation.",
)
@click.pass_context
def checkpoint(ctx: click.Context, no_notify: bool, no_plots: bool) -> None:
    """Run a checkpoint: collect status, generate plots, dispatch notifications."""
    formatter = ctx.obj["formatter"]
    experiment = ctx.obj["experiment"]
    runs = ctx.obj["runs"]
    redis_host = ctx.obj["redis_host"]
    redis_port = ctx.obj["redis_port"]

    if not experiment and not runs:
        click.echo(
            "Error: Checkpoint requires --experiment or --run flag.", err=True
        )
        ctx.exit(1)
        return

    run_configs = RunResolver.resolve(
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
    snapshots = monitor.collect(run_configs)

    # Display status
    rows = [_snapshot_to_row(s) for s in snapshots]
    columns = _build_columns(rows)
    formatter.echo(rows, columns=columns, title="Checkpoint Status")

    if no_notify:
        return

    # Dispatch notifications
    from gigaevo.monitoring.dispatcher import NotificationDispatcher
    from gigaevo.monitoring.notifications import StatusUpdate

    update = StatusUpdate(
        experiment_name=experiment or "ad-hoc",
        snapshots=snapshots,
    )

    dispatcher = NotificationDispatcher([])
    import asyncio

    asyncio.run(dispatcher.dispatch(update))
