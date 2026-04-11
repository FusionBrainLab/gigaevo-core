"""Logs subcommand -- discover and tail experiment log files."""

from __future__ import annotations

from pathlib import Path
import subprocess

import click


def _discover_log_file(experiment: str | None) -> Path | None:
    """Try to discover the nohup log file for an experiment."""
    if experiment is None:
        return None
    exp_dir = Path("experiments") / experiment
    if not exp_dir.exists():
        return None
    patterns = ["nohup_*.log", "nohup*.log", "*.log"]
    for pattern in patterns:
        matches = sorted(
            exp_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if matches:
            return matches[0]
    return None


@click.command()
@click.option(
    "--file",
    "log_file",
    type=click.Path(),
    default=None,
    help="Explicit log file path.",
)
@click.option(
    "-f", "--follow", is_flag=True, default=False, help="Follow log output (tail -f)."
)
@click.option(
    "-n", "--tail", "tail_n", type=int, default=50, help="Number of lines to show."
)
@click.pass_context
def logs(ctx: click.Context, log_file: str | None, follow: bool, tail_n: int) -> None:
    """Tail experiment log files."""
    experiment = ctx.obj.get("experiment")

    if log_file:
        path = Path(log_file)
    else:
        path = _discover_log_file(experiment)

    if path is None or not path.exists():
        click.echo("Error: Log file not found. Use --file or --experiment.", err=True)
        ctx.exit(1)
        return

    if follow:
        try:
            subprocess.run(["tail", "-f", str(path)], check=False)
        except KeyboardInterrupt:
            pass
    else:
        try:
            result = subprocess.run(
                ["tail", "-n", str(tail_n), str(path)],
                capture_output=True,
                text=True,
                check=False,
            )
            click.echo(result.stdout)
        except Exception as exc:
            click.echo(f"Error reading log: {exc}", err=True)
            ctx.exit(1)
