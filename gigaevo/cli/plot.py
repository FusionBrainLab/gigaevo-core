"""Plot subcommand -- fitness trajectory visualization (stub for Task 2)."""
from __future__ import annotations

import click


@click.command()
@click.option("--prefix", required=True, help="Redis key prefix for the run")
@click.option("--db", required=True, type=int, help="Redis database number")
@click.option("--redis-host", default="localhost", help="Redis server hostname")
@click.option("--redis-port", default=6379, type=int, help="Redis server port")
@click.option("--metric", default="fitness", help="Which metric to plot")
@click.option("-o", "--output", default=None, help="Custom output file path")
@click.option("--pdf", is_flag=True, help="Output PDF instead of PNG")
@click.option("--no-best", is_flag=True, help="Hide best fitness line")
@click.option("--no-mean", is_flag=True, help="Hide mean fitness line")
@click.option("--no-std", is_flag=True, help="Hide std deviation band")
@click.option("--problem-dir", default=None, help="Path for metrics.yaml")
def plot(
    prefix: str,
    db: int,
    redis_host: str,
    redis_port: int,
    metric: str,
    output: str | None,
    pdf: bool,
    no_best: bool,
    no_mean: bool,
    no_std: bool,
    problem_dir: str | None,
) -> None:
    """Plot fitness trajectory from a run (stub -- replaced by Task 2)."""
    click.echo("Plot command stub -- will be implemented in Task 2")
