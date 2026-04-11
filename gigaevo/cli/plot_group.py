"""Plot sub-group stub -- implemented in plan 04-03."""

from __future__ import annotations

import click


@click.group()
def plot() -> None:
    """Generate plots from evolution runs."""


@plot.command("comparison")
@click.pass_context
def comparison(ctx: click.Context) -> None:
    """Plot fitness comparison across runs."""
    click.echo("Not yet implemented")


@plot.command("trajectory")
@click.pass_context
def trajectory(ctx: click.Context) -> None:
    """Plot fitness trajectory for a run."""
    click.echo("Not yet implemented")
