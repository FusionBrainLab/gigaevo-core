"""Export sub-group stub -- implemented in plan 04-03."""

from __future__ import annotations

import click


@click.group()
def export() -> None:
    """Export evolution data to CSV."""


@export.command("csv")
@click.pass_context
def csv_cmd(ctx: click.Context) -> None:
    """Export full evolution data to CSV."""
    click.echo("Not yet implemented")


@export.command("frontier")
@click.pass_context
def frontier(ctx: click.Context) -> None:
    """Export frontier-only CSV."""
    click.echo("Not yet implemented")
