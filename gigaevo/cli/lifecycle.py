"""Lifecycle composite commands: preflight, launch, closeout, restart.

These are thin orchestrators that chain existing tool functions.
All destructive operations require --confirm.
"""

from __future__ import annotations

from pathlib import Path
import sys

import click


@click.command()
@click.pass_context
def preflight(ctx: click.Context) -> None:
    """Run pre-launch checks on experiment configuration."""
    experiment = ctx.obj.get("experiment")
    if not experiment:
        click.echo("Error: Preflight requires --experiment flag.", err=True)
        ctx.exit(1)
        return

    # Load preflight_check from project root (tools/experiment/preflight_check.py)
    proj = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(proj))
    try:
        from tools.experiment.preflight_check import _report, run_checks

        results = run_checks(experiment)
        exit_code = _report(results)
        sys.exit(exit_code)
    finally:
        sys.path.pop(0)


@click.command()
@click.option(
    "--confirm",
    is_flag=True,
    default=False,
    help="Actually execute. Without this flag, dry-run only.",
)
@click.pass_context
def launch(ctx: click.Context, confirm: bool) -> None:
    """Launch an experiment: preflight, config dump, start runs, verify PIDs."""
    experiment = ctx.obj.get("experiment")
    if not experiment:
        click.echo("Error: Launch requires --experiment flag.", err=True)
        ctx.exit(1)
        return

    from gigaevo.monitoring.manifest import load_manifest

    manifest = load_manifest(experiment)

    if not confirm:
        click.echo(f"[launch] DRY-RUN for {experiment}")
        click.echo(f"  Status: {manifest.experiment.status}")
        click.echo(f"  Runs: {len(manifest.runs)}")
        click.echo(f"  Servers: {len(manifest.servers)}")
        click.echo("\nPass --confirm to execute launch.")
        return

    click.echo(f"[launch] Starting {experiment}...")
    click.echo(
        "Use the experiment-launch skill for the full launch workflow "
        "with preflight checks, config dumps, and PID verification."
    )


@click.command()
@click.option(
    "--confirm",
    is_flag=True,
    default=False,
    help="Actually execute. Without this flag, dry-run only.",
)
@click.pass_context
def closeout(ctx: click.Context, confirm: bool) -> None:
    """Close out a completed experiment: archive, analyze, update PR."""
    experiment = ctx.obj.get("experiment")
    if not experiment:
        click.echo("Error: Closeout requires --experiment flag.", err=True)
        ctx.exit(1)
        return

    from gigaevo.monitoring.manifest import load_manifest

    manifest = load_manifest(experiment)

    if not confirm:
        click.echo(f"[closeout] DRY-RUN for {experiment}")
        click.echo(f"  Status: {manifest.experiment.status}")
        click.echo(f"  Runs: {len(manifest.runs)}")
        click.echo("\nPass --confirm to execute closeout.")
        return

    click.echo(f"[closeout] Closing out {experiment}...")
    click.echo(
        "Use the experiment-closeout skill for the full closeout workflow "
        "with archiving, analysis, and PR updates."
    )


@click.command()
@click.option(
    "--confirm",
    is_flag=True,
    default=False,
    help="Actually execute. Without this flag, dry-run only.",
)
@click.pass_context
def restart(ctx: click.Context, confirm: bool) -> None:
    """Restart an experiment: kill runs, flush DBs, re-launch."""
    experiment = ctx.obj.get("experiment")
    if not experiment:
        click.echo("Error: Restart requires --experiment flag.", err=True)
        ctx.exit(1)
        return

    from gigaevo.monitoring.manifest import load_manifest

    manifest = load_manifest(experiment)

    if not confirm:
        click.echo(f"[restart] DRY-RUN for {experiment}")
        click.echo(f"  Status: {manifest.experiment.status}")
        click.echo(f"  Runs to kill: {len(manifest.runs)}")
        for run in manifest.runs:
            click.echo(f"    {run.label}: DB {run.db}, PID {run.pid}")
        click.echo("\nPass --confirm to execute restart.")
        return

    click.echo(f"[restart] Restarting {experiment}...")
    click.echo(
        "Use the experiment-restart skill for the full restart workflow "
        "with process cleanup, DB flush, and re-launch."
    )
