"""Flush subcommand -- kill workers and flush Redis databases."""

from __future__ import annotations

import time
from typing import Any

import click

from gigaevo.cli.flush_ops import (
    find_exec_runner_pids,
    flush_db,
    kill_run_writers,
    kill_workers,
)


class _VarDbOption(click.Option):
    """--db option that gobbles trailing non-flag args so --db 1 2 3 works.

    Supports all three calling styles:
      --db 1 2 3 4           (space-separated — grabs trailing non-flag args)
      --db 1,2,3,4           (comma-separated in one token)
      --db 1 --db 2 --db 3   (repeated flags, backward-compatible)
    """

    def add_to_parser(self, parser: Any, ctx: click.Context) -> None:  # type: ignore[override]
        super().add_to_parser(parser, ctx)
        for name in self.opts:
            opt = parser._long_opt.get(name) or parser._short_opt.get(name)
            if opt is None:
                continue
            orig_process = opt.process

            def _process(
                value: str,
                state: Any,
                _orig: Any = orig_process,
            ) -> None:
                # Gobble additional non-flag tokens into this --db group
                while state.rargs and not state.rargs[0].startswith("-"):
                    value = value + "," + state.rargs.pop(0)
                _orig(value, state)

            opt.process = _process


@click.command()
@click.option(
    "--db",
    cls=_VarDbOption,
    multiple=True,
    required=True,
    type=str,
    help=(
        "Redis DB numbers to flush. "
        "Space-separated (--db 1 2 3), comma-separated (--db 1,2,3), "
        "or repeated (--db 1 --db 2)."
    ),
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
@click.option(
    "--kill-only",
    is_flag=True,
    default=False,
    help="Kill workers only, skip Redis flush.",
)
@click.option(
    "--kill-orphans",
    is_flag=True,
    default=False,
    help=(
        "Also kill exec_runner workers whose parent has exited. These workers "
        "cannot be attributed to a target DB, so this is off by default."
    ),
)
@click.pass_context
def flush(
    ctx: click.Context,
    db: tuple[str, ...],
    confirm: bool,
    dry_run: bool,
    no_kill_workers: bool,
    kill_only: bool,
    kill_orphans: bool,
) -> None:
    """Kill stale workers and flush Redis databases.

    DESTRUCTIVE — dry-run by default. Pass `--confirm` to actually execute.

    For each target DB, the command kills any `run.py` writer processes,
    then its child `exec_runner.py` worker processes, then calls `FLUSHDB` on
    each Redis DB. Orphan workers are only included with `--kill-orphans`
    because their DB ownership is unknowable. Warns for 5 s
    before flushing a non-empty DB so you can abort with Ctrl+C if you
    forgot to archive.

    DBs must be non-negative and may be passed as
    `--db 1 2 3`, `--db 1,2,3`, or `--db 1 --db 2 --db 3`. Use
    `--kill-only` to stop workers without flushing, or
    `--no-kill-workers` to flush without touching processes.
    """
    redis_host = ctx.obj["redis_host"]
    redis_port = ctx.obj["redis_port"]

    if kill_only and no_kill_workers:
        raise click.UsageError(
            "--kill-only and --no-kill-workers cannot be used together"
        )

    # Parse: each db entry may be comma-separated (from gobbled space args or literal commas)
    raw: list[int] = []
    for entry in db:
        for part in entry.replace(",", " ").split():
            try:
                raw.append(int(part))
            except ValueError:
                click.echo(f"Error: '{part}' is not a valid DB number", err=True)
                ctx.exit(1)
                return

    if not raw:
        raise click.BadParameter("provide at least one DB number", param_hint="--db")

    # Validate DB range and avoid performing destructive work twice.
    for d in raw:
        if d < 0:
            raise click.BadParameter(
                f"DB number must be non-negative (got {d})", param_hint="--db"
            )

    dbs = list(dict.fromkeys(raw))
    is_dry_run = not confirm or dry_run

    if kill_only:
        click.echo("[flush] KILL-ONLY mode -- workers only, no DB flush\n")
    elif is_dry_run:
        click.echo("[flush] DRY-RUN mode -- pass --confirm to execute\n")
    else:
        click.echo(f"[flush] DESTRUCTIVE OPERATION: Flushing Redis DBs {dbs}\n")

    # Step 1: Kill workers
    if not no_kill_workers:
        # Resolve child PIDs while their run.py parents still carry the target
        # DB in their command line. Once a writer exits, that attribution is
        # lost and its children look like unrelated orphans.
        pids = find_exec_runner_pids(dbs, include_orphans=kill_orphans)
        writer_pids = kill_run_writers(dbs, is_dry_run)
        kill_workers(pids, is_dry_run)
        if not is_dry_run and (writer_pids or pids):
            time.sleep(2)
    else:
        click.echo("[workers] Skipping exec_runner cleanup (--no-kill-workers)")

    # Step 2: Flush each DB (skip if --kill-only)
    all_ok = True
    if not kill_only:
        for d in dbs:
            ok = flush_db(d, redis_host, redis_port, is_dry_run)
            if not ok:
                all_ok = False

    # Step 3: Summary
    if kill_only:
        click.echo("\n[summary] Workers cleanup complete.")
    elif is_dry_run and all_ok:
        click.echo("\n[summary] Dry-run complete. Run with --confirm to execute.")
    elif all_ok:
        click.echo("\n[summary] All DBs flushed successfully.")
    else:
        click.echo("\n[summary] Some DBs may not be clean -- check output above.")
        ctx.exit(1)
