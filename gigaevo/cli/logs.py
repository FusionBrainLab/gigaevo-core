"""Logs subcommand stub -- implemented in plan 04-02."""

from __future__ import annotations

import click


@click.command()
@click.pass_context
def logs(ctx: click.Context) -> None:
    """Tail experiment log files."""
    click.echo("Not yet implemented")
