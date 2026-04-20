"""Top programs subcommand -- inspect best programs by fitness."""

from __future__ import annotations

import json
from pathlib import Path

import click
import redis as redis_lib

from gigaevo.cli.output_formatter import OutputFormatter
from gigaevo.cli.run_resolver import RunResolver


def _fetch_top_programs(
    r: redis_lib.Redis,
    prefix: str,
    metric: str,
    n: int,
    minimize: bool = False,
) -> list[dict]:
    """Fetch top N programs from Redis by metric value."""
    program_keys = r.keys(f"{prefix}:program:*")
    programs = []
    for key in program_keys:
        raw = r.get(key)
        if raw is None:
            continue
        try:
            prog = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        metrics = prog.get("metrics", {})
        val = metrics.get(metric)
        if val is None:
            continue
        programs.append(
            {
                "id": prog.get("id", "?"),
                "generation": prog.get("generation")
                or prog.get("lineage", {}).get("generation"),
                metric: val,
                "state": prog.get("state", "?"),
                "code": prog.get("code", ""),
            }
        )

    programs.sort(key=lambda p: p.get(metric, 0), reverse=not minimize)
    return programs[:n]


@click.command()
@click.option("-n", "--top-n", type=int, default=5, help="Number of top programs.")
@click.option("--metric", default="fitness", help="Metric to rank by.")
@click.option("--minimize", is_flag=True, default=False, help="Lower is better.")
@click.option(
    "--code", "show_code", is_flag=True, default=False, help="Show source code."
)
@click.option(
    "--save-dir", type=click.Path(), default=None, help="Save programs to files."
)
@click.option(
    "-f",
    "--format",
    "format_name",
    type=click.Choice(["table", "json", "csv", "markdown"], case_sensitive=False),
    default=None,
    help=(
        "Output format override (table|json|csv|markdown). Passed AFTER "
        "the subcommand — overrides the global `-f/--format` flag when "
        "given."
    ),
)
@click.pass_context
def top(
    ctx: click.Context,
    top_n: int,
    metric: str,
    minimize: bool,
    show_code: bool,
    save_dir: str | None,
    format_name: str | None,
) -> None:
    """Inspect top-N programs ranked by a metric.

    Auto-detection: in `-e/--experiment` mode, `--metric` defaults to the
    manifest's `problem.metric_name` (e.g. `actual_fitness`). Otherwise
    defaults to `fitness`. Pass `--minimize` to rank ascending.
    """
    if top_n < 1:
        raise click.BadParameter(
            f"-n/--top-n must be >= 1 (got {top_n})", param_hint="-n/--top-n"
        )
    formatter = ctx.obj["formatter"]
    if format_name is not None:
        formatter = OutputFormatter(format_name=format_name)
        ctx.obj["formatter"] = formatter
    experiment = ctx.obj["experiment"]
    runs = ctx.obj["runs"]
    redis_host = ctx.obj["redis_host"]
    redis_port = ctx.obj["redis_port"]

    # Use manifest's problem.metric_name as default when in --experiment mode
    # and user didn't explicitly pass --metric (still has Click default "fitness")
    if metric == "fitness" and experiment:
        try:
            from gigaevo.experiment.manifest import load_manifest

            manifest = load_manifest(experiment)
            if manifest.contract.problem.metric_name:
                metric = manifest.contract.problem.metric_name
        except Exception:
            pass  # fall back to "fitness"

    run_configs = RunResolver.resolve(
        experiment=experiment,
        runs=runs,
        redis_host=redis_host,
        redis_port=redis_port,
    )

    redis_factory = ctx.obj.get("redis_factory")
    all_programs: list[dict] = []

    for rc in run_configs:
        spec = rc.run_spec
        if redis_factory:
            r = redis_factory(spec.db)
        else:
            r = redis_lib.Redis(
                host=redis_host, port=redis_port, db=spec.db, decode_responses=True
            )
        try:
            progs = _fetch_top_programs(r, spec.prefix, metric, top_n, minimize)
            for p in progs:
                p["label"] = spec.label
            all_programs.extend(progs)
        finally:
            r.close()

    all_programs.sort(key=lambda p: p.get(metric, 0), reverse=not minimize)
    all_programs = all_programs[:top_n]

    rows: list[dict[str, object]] = []
    for p in all_programs:
        row = {
            "Rank": len(rows) + 1,
            "ID": p["id"][:12],
            "Label": p.get("label", ""),
            "Gen": p.get("generation"),
            metric.title(): p.get(metric),
            "State": p.get("state"),
        }
        rows.append(row)

    columns = ["Rank", "ID", "Label", "Gen", metric.title(), "State"]
    formatter.echo(rows, columns=columns, title=f"Top {top_n} Programs")

    if show_code:
        for p in all_programs:
            click.echo(f"\n--- {p['id'][:12]} ({metric}={p.get(metric)}) ---")
            click.echo(p.get("code", "(no code)"))

    if save_dir:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(all_programs):
            path = out / f"top_{i + 1}_{p['id'][:12]}.py"
            path.write_text(p.get("code", ""))
        click.echo(f"Saved {len(all_programs)} programs to {save_dir}")
