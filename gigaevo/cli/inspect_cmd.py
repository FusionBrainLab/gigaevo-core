"""Inspect subcommand -- discover experiment prefixes in a Redis DB."""

from __future__ import annotations

import click
import redis

_LOCK_SUFFIX = ":__instance_lock__"


def discover_prefixes(
    redis_host: str,
    redis_port: int,
    db: int,
) -> list[str]:
    """Return experiment prefixes in a Redis DB by finding instance lock keys."""
    r = redis.Redis(host=redis_host, port=redis_port, db=db, decode_responses=True)
    try:
        keys = r.keys(f"*{_LOCK_SUFFIX}")
    finally:
        r.close()
    return sorted(k.removesuffix(_LOCK_SUFFIX) for k in keys)


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

    Scans for instance-lock keys (`<prefix>:__instance_lock__`) and prints
    each detected prefix paired with its DB. Useful when you inherit a
    Redis DB and need to find the right `-r prefix@db[:label]` spec.

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
