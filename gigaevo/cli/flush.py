"""Flush subcommand -- kill workers and flush Redis databases."""

from __future__ import annotations

import time

import click

from gigaevo.cli.flush_ops import (
    find_exec_runner_pids,
    flush_db,
    kill_run_writers,
    kill_workers,
)


@click.command()
@click.option(
    "--db",
    multiple=True,
    required=True,
    type=str,
    help="Redis DB numbers to flush. Repeat (--db 1 --db 2) or comma-separate (--db 1,2,3).",
)
@click.option(
    "--confirm",
    is_flag=True,
    default=False,
    help="Actually execute. Without this flag, dry-run only.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Explicit dry-run mode (same as omitting --confirm).",
)
@click.option(
    "--no-kill-workers",
    is_flag=True,
    default=False,
    help="Skip killing exec_runner workers.",
)
@click.pass_context
def flush(
    ctx: click.Context,
    db: tuple[str, ...],
    confirm: bool,
    dry_run: bool,
    no_kill_workers: bool,
) -> None:
    """Kill workers and flush Redis databases.

    Dry-run by default -- pass --confirm to execute.
    """
    redis_host = ctx.obj["redis_host"]
    redis_port = ctx.obj["redis_port"]

    # Parse: support --db 1 --db 2, --db 1,2,3, or --db '1 2 3'
    raw: list[int] = []
    for entry in db:
        for part in entry.replace(",", " ").split():
            try:
                raw.append(int(part))
            except ValueError:
                click.echo(f"Error: '{part}' is not a valid DB number", err=True)
                ctx.exit(1)
                return

    # Validate DB range
    for d in raw:
        if not 0 <= d <= 15:
            click.echo(f"Error: DB number {d} out of range (0-15)", err=True)
            ctx.exit(1)
            return

    dbs = raw
    is_dry_run = not confirm or dry_run

    if is_dry_run:
        click.echo("[flush] DRY-RUN mode -- pass --confirm to execute\n")
    else:
        click.echo(f"[flush] DESTRUCTIVE OPERATION: Flushing Redis DBs {dbs}\n")

    # Step 1: Kill workers
    if not no_kill_workers:
        kill_run_writers(dbs, is_dry_run)
        pids = find_exec_runner_pids(dbs)
        kill_workers(pids, is_dry_run)
        if not is_dry_run and pids:
            time.sleep(2)
    else:
        click.echo("[workers] Skipping exec_runner cleanup (--no-kill-workers)")

    # Step 2: Flush each DB
    all_ok = True
    for d in dbs:
        ok = flush_db(d, redis_host, redis_port, is_dry_run)
        if not ok:
            all_ok = False

    # Step 3: Summary
    if is_dry_run:
        click.echo("\n[summary] Dry-run complete. Run with --confirm to execute.")
    elif all_ok:
        click.echo("\n[summary] All DBs flushed successfully.")
    else:
        click.echo("\n[summary] Some DBs may not be clean -- check output above.")
        ctx.exit(1)
