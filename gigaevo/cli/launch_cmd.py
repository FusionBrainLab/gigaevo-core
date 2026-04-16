"""CLI command: gigaevo -e <exp> launch."""

from __future__ import annotations

import click

from gigaevo.experiment.launch import run_launch


@click.command("launch")
@click.option("--dry-run", is_flag=True, help="Validate and claim DBs, but don't exec.")
@click.option(
    "--skip-preflight",
    is_flag=True,
    help="Skip preflight checks (for re-launches where preflight already passed).",
)
@click.pass_context
def launch(ctx: click.Context, dry_run: bool, skip_preflight: bool) -> None:
    """Launch an implemented experiment: preflight, exec, set running, spawn watchdog."""
    experiment = ctx.obj.get("experiment")
    if not experiment:
        click.echo("Error: launch requires --experiment / -e flag.", err=True)
        ctx.exit(1)
        return

    result = run_launch(experiment, dry_run=dry_run, skip_preflight=skip_preflight)

    if result.ok:
        if dry_run:
            click.echo(
                f"Dry run complete for {result.experiment} "
                f"(last step: {result.last_completed_step.name}). "
                f"No runs started."
            )
        else:
            pids = ", ".join(f"{k}={v}" for k, v in result.run_pids.items())
            click.echo(
                f"Launched {result.experiment}: status={result.status}, "
                f"PIDs=[{pids}], watchdog={result.watchdog_pid}"
            )
    else:
        click.echo(f"Launch failed: {result.error}", err=True)
        ctx.exit(1)
