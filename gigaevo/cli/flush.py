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
