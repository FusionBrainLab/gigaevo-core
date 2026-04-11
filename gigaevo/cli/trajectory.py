"""Trajectory subcommand stub -- implemented in plan 04-02."""

from __future__ import annotations

import click


@click.command()
@click.pass_context
def trajectory(ctx: click.Context) -> None:
    """Show gen-by-gen fitness trajectory."""
    click.echo("Not yet implemented")
