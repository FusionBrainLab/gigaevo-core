"""Flush subcommand stub -- implemented in plan 04-04."""

from __future__ import annotations

import click


@click.command()
@click.pass_context
def flush(ctx: click.Context) -> None:
    """Kill workers and flush Redis databases."""
    click.echo("Not yet implemented")
