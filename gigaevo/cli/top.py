"""Top programs subcommand stub -- implemented in plan 04-02."""

from __future__ import annotations

import click


@click.command()
@click.pass_context
def top(ctx: click.Context) -> None:
    """Inspect top programs by fitness."""
    click.echo("Not yet implemented")
