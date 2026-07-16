"""Inspect subcommand -- discover experiment prefixes in a Redis DB."""

from __future__ import annotations

import click
import redis

_LOCK_SUFFIX = ":__instance_lock__"
_RUN_STATE_SUFFIX = ":run_state"
_PROGRAM_MARKER = ":program:"


def discover_prefixes(
    redis_host: str,
    redis_port: int,
    db: int,
) -> list[str]:
    """Return prefixes inferred from lock, run-state, and program keys."""
    r = redis.Redis(host=redis_host, port=redis_port, db=db, decode_responses=True)
    try:
        prefixes = {
            key.removesuffix(_LOCK_SUFFIX)
            for key in r.scan_iter(match=f"*{_LOCK_SUFFIX}", count=1000)
        }
        prefixes.update(
            key.removesuffix(_RUN_STATE_SUFFIX)
            for key in r.scan_iter(match=f"*{_RUN_STATE_SUFFIX}", count=1000)
        )
        prefixes.update(
            key.rsplit(_PROGRAM_MARKER, 1)[0]
            for key in r.scan_iter(match=f"*{_PROGRAM_MARKER}*", count=1000)
            if _PROGRAM_MARKER in key
        )
    finally:
        r.close()
    return sorted(prefixes)


@click.command("inspect")
@click.option(
    "--db",
    multiple=True,
    required=True,
    type=int,
    help="Redis DB number(s) to inspect (repeat for multiple).",
)
@click.pass_context
def inspect(ctx: click.Context, db: tuple[int, ...]) -> None:
    """Discover which experiment prefix(es) live in a Redis DB.

    Scans non-blockingly for instance-lock, run-state, and program keys, then
    prints each detected prefix paired with its DB. This also discovers cleanly
    stopped runs whose instance lock has already been released.

    Pass `--db N` repeatedly to inspect multiple DBs.
    """
    redis_host: str = ctx.obj["redis_host"]
    redis_port: int = ctx.obj["redis_port"]

    for d in db:
        if not 0 <= d <= 15:
            click.echo(f"Error: DB number {d} out of range (0-15)", err=True)
            ctx.exit(1)
            return

    for d in db:
        prefixes = discover_prefixes(redis_host, redis_port, d)
        if prefixes:
            for p in sorted(prefixes):
                click.echo(f"db={d}  prefix={p}")
        else:
            click.echo(f"db={d}  (empty or no recognized keys)")
