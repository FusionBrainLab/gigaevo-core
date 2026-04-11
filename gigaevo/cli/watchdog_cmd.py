"""Watchdog subcommand stub -- implemented in plan 04-04."""

from __future__ import annotations

import click


@click.command()
@click.pass_context
def watchdog(ctx: click.Context) -> None:
    """Start or manage the experiment watchdog."""
    click.echo("Not yet implemented")
