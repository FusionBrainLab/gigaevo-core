"""Profiler subcommand -- parse a runner log into text + HTML flow profile.

Two artifacts are emitted per source log:

* ``<out>/profile_<label>.txt`` -- plain-text pipeline summary (counts,
  refresh queue stats, per-program timeline).
* ``<out>/profile_<label>.html`` -- interactive Plotly dashboard (per-program
  lifecycle bars, stage sub-bars, refresh + re-eval bands, decision bars).

Resolution priority (highest first):

1. ``--file <path>`` -- profile that log alone (no manifest required).
2. Positional labels with ``-e/--experiment`` -- resolve to
   ``experiments/<exp>/run_<label>.log``.
3. No labels + ``-e`` -- profile every run in the manifest.
"""

from __future__ import annotations

from pathlib import Path

import click

from gigaevo.monitoring.flow_profiler import (
    DEFAULT_LAST_N_ROWS,
    compute_saturation,
    compute_utilization,
    format_summary_text,
    parse_log,
    render_full_html,
)


def _load_manifest(experiment: str):
    """Lazy-load experiment manifest to avoid import at CLI startup."""
    from gigaevo.experiment.manifest import load_manifest

    return load_manifest(experiment)


def _experiment_dir(experiment: str) -> Path:
    return Path("experiments") / experiment


def _log_path_for_label(experiment: str, label: str) -> Path:
    return _experiment_dir(experiment) / f"run_{label}.log"


def _profile_one(
    log_path: Path,
    out_dir: Path,
    label: str,
    *,
    emit_text: bool,
    emit_html: bool,
    subtitle: str = "",
    last_n: int | None = DEFAULT_LAST_N_ROWS,
) -> None:
    """Parse one log and write the requested artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    programs, refreshes, llm_events, backpressure = parse_log(log_path)
    utilization = compute_utilization(programs, refreshes, llm_events)
    saturation = compute_saturation(backpressure)

    if emit_text:
        txt_path = out_dir / f"profile_{label}.txt"
        body = format_summary_text(
            programs,
            refreshes,
            log_path=log_path,
            utilization=utilization,
            saturation=saturation,
        )
        txt_path.write_text(body)
        click.echo(f"wrote {txt_path}")

    if emit_html:
        html_path = out_dir / f"profile_{label}.html"
        html = render_full_html(
            programs,
            refreshes,
            title=f"flow profile · {label}",
            subtitle=subtitle or str(log_path),
            div_id=f"gigaevo-flow-{label}",
            utilization=utilization,
            backpressure=backpressure,
            saturation=saturation,
            last_n=last_n,
        )
        html_path.write_text(html)
        click.echo(f"wrote {html_path}")


@click.command()
@click.argument("labels", nargs=-1)
@click.option(
    "--file",
    "log_file",
    type=click.Path(),
    default=None,
    help=(
        "Explicit log file path (bypasses manifest). The output label is "
        "derived from the file stem."
    ),
)
@click.option(
    "--out-dir",
    type=click.Path(),
    default=None,
    help=(
        "Output directory for profile artifacts. Defaults to "
        "experiments/<exp>/profiler/ in -e mode, or ./ when using --file."
    ),
)
@click.option(
    "--text-only",
    is_flag=True,
    default=False,
    help="Only emit the .txt summary (skip HTML).",
)
@click.option(
    "--html-only",
    is_flag=True,
    default=False,
    help="Only emit the .html dashboard (skip text summary).",
)
@click.option(
    "--last-n",
    type=int,
    default=DEFAULT_LAST_N_ROWS,
    show_default=True,
    help=(
        "Initial y-axis window: show only the last N programs on page "
        "open. Toolbar buttons let viewers widen or show all. Use 0 (or "
        "any non-positive value) to disable clipping."
    ),
)
@click.pass_context
def profiler(
    ctx: click.Context,
    labels: tuple[str, ...],
    log_file: str | None,
    out_dir: str | None,
    text_only: bool,
    html_only: bool,
    last_n: int,
) -> None:
    """Profile evolution runner logs into text summary + HTML dashboard.

    \b
    Usage:
      gigaevo -e <exp> profiler                Profile every run in the manifest.
      gigaevo -e <exp> profiler <label>        Profile one run by label.
      gigaevo -e <exp> profiler <a> <b>        Profile multiple runs.
      gigaevo profiler --file <path>           Profile an arbitrary log.

    Output: two files per run, named ``profile_<label>.{txt,html}``, placed
    under ``experiments/<exp>/profiler/`` by default (override with
    ``--out-dir``).
    """
    if text_only and html_only:
        raise click.UsageError("--text-only and --html-only are mutually exclusive")
    emit_text = not html_only
    emit_html = not text_only
    last_n_arg: int | None = last_n if last_n and last_n > 0 else None

    experiment: str | None = ctx.obj.get("experiment")

    # 1. Explicit --file
    if log_file:
        path = Path(log_file)
        if not path.exists():
            click.echo(f"Error: log file not found: {path}", err=True)
            ctx.exit(1)
            return
        label = path.stem
        if label.startswith("run_"):
            label = label[len("run_") :]
        target = Path(out_dir) if out_dir else Path(".")
        _profile_one(
            path,
            target,
            label,
            emit_text=emit_text,
            emit_html=emit_html,
            last_n=last_n_arg,
        )
        ctx.exit(0)
        return

    # 2/3. Need --experiment
    if not experiment:
        click.echo(
            "Error: provide --experiment (-e) or --file. "
            "Example: gigaevo -e task/name profiler [LABEL ...]",
            err=True,
        )
        ctx.exit(1)
        return

    manifest = _load_manifest(experiment)
    known = {run.label for run in manifest.contract.runs}
    target_labels = list(labels) if labels else sorted(known)
    unknown = [label for label in target_labels if label not in known]
    if unknown:
        click.echo(
            f"Error: unknown run label(s): {', '.join(unknown)}. "
            f"Known: {', '.join(sorted(known))}",
            err=True,
        )
        ctx.exit(1)
        return

    paths = [(label, _log_path_for_label(experiment, label)) for label in target_labels]
    missing = [str(p) for _, p in paths if not p.exists()]
    if missing:
        click.echo(
            f"Error: log file(s) not found: {', '.join(missing)}",
            err=True,
        )
        ctx.exit(1)
        return

    default_out = _experiment_dir(experiment) / "profiler"
    target_dir = Path(out_dir) if out_dir else default_out
    for label, path in paths:
        _profile_one(
            path,
            target_dir,
            label,
            emit_text=emit_text,
            emit_html=emit_html,
            subtitle=f"{experiment} / {label}",
            last_n=last_n_arg,
        )

    ctx.exit(0)
