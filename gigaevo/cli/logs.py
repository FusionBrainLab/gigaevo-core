"""Logs subcommand -- tail per-run experiment log files.

Resolution priority:
  1. Explicit --file path (bypasses manifest entirely).
  2. Positional run labels resolved via manifest to run_<label>.log.
  3. No args + --experiment → list mode (table of candidate run logs).
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import click


def _load_manifest(experiment: str):
    """Lazy-load experiment manifest (avoid import at CLI startup)."""
    from gigaevo.experiment.manifest import load_manifest

    return load_manifest(experiment)


def _experiment_dir(experiment: str) -> Path:
    return Path("experiments") / experiment


def _log_path_for_label(experiment: str, label: str) -> Path:
    return _experiment_dir(experiment) / f"run_{label}.log"


def _list_run_logs(ctx: click.Context, experiment: str) -> int:
    """Render a table of run logs from the manifest with size/mtime/exists."""
    manifest = _load_manifest(experiment)
    exp_dir = _experiment_dir(experiment)
    rows: list[dict] = []
    for run in manifest.contract.runs:
        path = exp_dir / f"run_{run.label}.log"
        exists = path.exists()
        rows.append(
            {
                "label": run.label,
                "db": run.db,
                "log_file": str(path),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else 0,
                "mtime": int(path.stat().st_mtime) if exists else 0,
            }
        )
    formatter = ctx.obj["formatter"]
    formatter.echo(
        rows,
        columns=["label", "db", "log_file", "exists", "size_bytes", "mtime"],
    )
    return 0


def _tail(paths: list[Path], follow: bool, tail_n: int) -> int:
    cmd: list[str] = ["tail"]
    if follow:
        cmd.append("-f")
    cmd.extend(["-n", str(tail_n)])
    cmd.extend(str(p) for p in paths)
    if follow:
        try:
            subprocess.run(cmd, check=False)
        except KeyboardInterrupt:
            return 0
        return 0
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    click.echo(result.stdout)
    return result.returncode or 0


@click.command()
@click.argument("labels", nargs=-1)
@click.option(
    "--file",
    "log_file",
    type=click.Path(),
    default=None,
    help="Explicit log file path (bypasses manifest).",
)
@click.option(
    "-f",
    "--follow",
    is_flag=True,
    default=False,
    help="Live tail (tail -f).",
)
@click.option(
    "-n",
    "--tail",
    "tail_n",
    type=int,
    default=50,
    help="Number of lines to show (ignored with -f streaming past history).",
)
@click.pass_context
def logs(
    ctx: click.Context,
    labels: tuple[str, ...],
    log_file: str | None,
    follow: bool,
    tail_n: int,
) -> None:
    """Tail experiment run log files.

    Usage:

      gigaevo -e <exp> logs                 List all run logs (sizes, mtimes).
      gigaevo -e <exp> logs <label>         Tail run_<label>.log.
      gigaevo -e <exp> logs <label> -f      Live tail.
      gigaevo -e <exp> logs <a> <b> -f      Tail multiple runs (multiplexed).
      gigaevo logs --file <path>            Tail an arbitrary file.
    """
    experiment = ctx.obj.get("experiment")

    # 1. Explicit --file.
    if log_file:
        path = Path(log_file)
        if not path.exists():
            click.echo(f"Error: log file not found: {path}", err=True)
            ctx.exit(1)
            return
        ctx.exit(_tail([path], follow=follow, tail_n=tail_n))
        return

    # 2/3. Require --experiment for label resolution or list mode.
    if not experiment:
        click.echo(
            "Error: provide --experiment (-e) or --file. "
            "Example: gigaevo -e task/name logs [LABEL ...]",
            err=True,
        )
        ctx.exit(1)
        return

    if not labels:
        # 3. List mode.
        ctx.exit(_list_run_logs(ctx, experiment))
        return

    # 2. Positional labels.
    manifest = _load_manifest(experiment)
    known = {run.label for run in manifest.contract.runs}
    unknown = [label for label in labels if label not in known]
    if unknown:
        click.echo(
            f"Error: unknown run label(s): {', '.join(unknown)}. "
            f"Known: {', '.join(sorted(known))}",
            err=True,
        )
        ctx.exit(1)
        return

    paths = [_log_path_for_label(experiment, label) for label in labels]
    missing = [p for p in paths if not p.exists()]
    if missing:
        click.echo(
            f"Error: log file(s) not found: {', '.join(str(p) for p in missing)}",
            err=True,
        )
        ctx.exit(1)
        return

    ctx.exit(_tail(paths, follow=follow, tail_n=tail_n))
