"""Analyze subcommand -- comprehensive experiment analysis (stub for Task 2)."""
from __future__ import annotations

import click


@click.command()
@click.option("--prefix", required=True, help="Redis key prefix for the run")
@click.option("--db", required=True, type=int, help="Redis database number")
@click.option("--redis-host", default="localhost", help="Redis server hostname")
@click.option("--redis-port", default=6379, type=int, help="Redis server port")
@click.option("--top-n", default=5, type=int, help="Number of top programs")
@click.option("--metric", default="fitness", help="Which metric to analyze")
@click.option("--problem-dir", default=None, help="Path for metrics.yaml")
@click.option(
    "--section",
    default=None,
    type=click.Choice(["top", "stats", "convergence"]),
    help="Extract single section",
)
@click.option(
    "--compare",
    multiple=True,
    help='Cross-run comparison specs as "prefix@db"',
)
def analyze(
    prefix: str,
    db: int,
    redis_host: str,
    redis_port: int,
    top_n: int,
    metric: str,
    problem_dir: str | None,
    section: str | None,
    compare: tuple[str, ...],
) -> None:
    """Comprehensive experiment analysis (stub -- replaced by Task 2)."""
    click.echo("Analyze command stub -- will be implemented in Task 2")
