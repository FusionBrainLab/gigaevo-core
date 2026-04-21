"""CLI command: gigaevo -e <exp> launch."""

from __future__ import annotations

import click

from gigaevo.experiment.launch import _generate_launch_script, run_launch


@click.command("launch")
@click.option("--dry-run", is_flag=True, help="Validate and claim DBs, but don't exec.")
@click.option(
    "--skip-preflight",
    is_flag=True,
    help="Skip preflight checks (for re-launches where preflight already passed).",
)
@click.option(
    "--generate-script",
    is_flag=True,
    help="Generate experiments/<exp>/launch.sh from experiment.yaml and exit (no preflight, no DB claim, no exec).",
)
@click.pass_context
def launch(
    ctx: click.Context, dry_run: bool, skip_preflight: bool, generate_script: bool
) -> None:
    """Launch an implemented experiment end-to-end.

    Default flow: run preflight (LLM endpoints, DB claims, pin checks) →
    generate `launch.sh` → exec all runs via `nohup` → set status to
    `running` → spawn watchdog.

    \b
    Flags select a partial flow:
      --dry-run          Preflight + DB claim only; do not exec. Writes
                         LAUNCH_PREVIEW.md so the researcher can verify
                         resolved config.
      --skip-preflight   Assume preflight already passed (for re-launches
                         after fixing a transient issue).
      --generate-script  Write `experiments/<exp>/launch.sh` and exit.
                         No preflight, no DB claim, no exec. Useful when
                         you want to inspect / hand-edit / re-run `bash
                         launch.sh` manually.
    """
    experiment = ctx.obj.get("experiment")
    if not experiment:
        click.echo("Error: launch requires --experiment / -e flag.", err=True)
        ctx.exit(1)
        return

    if generate_script and dry_run:
        click.echo(
            "Error: --generate-script and --dry-run are mutually exclusive.", err=True
        )
        ctx.exit(1)
        return

    if generate_script:
        out_path = _generate_launch_script(experiment)
        click.echo(f"Generated launch.sh at {out_path}")
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
