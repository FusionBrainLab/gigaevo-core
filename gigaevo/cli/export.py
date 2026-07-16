"""Export sub-group: csv and frontier CSV export commands.

Selection semantics:
  * No positional labels → operate on all runs resolved from --experiment/--run.
  * Positional labels → filter resolved runs to only those labels (unknown → error).
  * 1 run in scope → write to the exact -o path, emit flat JSON summary.
  * >1 run in scope → fan out to `<stem>_<label><suffix>`, emit JSON list.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING

import click

from gigaevo.cli.run_resolver import RunResolver

if TYPE_CHECKING:
    import pandas as pd


def _fetch_dataframe(run_config, redis_host: str, redis_port: int) -> pd.DataFrame:
    """Fetch evolution DataFrame for a single run (disk or Redis)."""
    from gigaevo.cli.run_resolver import build_readonly_storage
    from gigaevo.utils.dataframes import fetch_evolution_dataframe

    storage = build_readonly_storage(run_config.run_spec, redis_host, redis_port)

    async def _fetch():
        async with storage:
            return await fetch_evolution_dataframe(storage, add_stage_results=False)

    return asyncio.run(_fetch())


def _serialize_complex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Serialize dict/list columns as JSON strings for CSV output."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(1)
            if len(sample) > 0 and isinstance(sample.iloc[0], (dict, list)):
                df[col] = df[col].apply(
                    lambda x: (
                        json.dumps(x, default=str) if isinstance(x, (dict, list)) else x
                    )
                )
    return df


def _resolve_runs(ctx: click.Context, labels: tuple[str, ...]):
    """Resolve -e/-r into RunConfig list, filtered by positional labels."""
    experiment = ctx.obj["experiment"]
    runs = ctx.obj["runs"]
    redis_host = ctx.obj["redis_host"]
    redis_port = ctx.obj["redis_port"]

    run_configs = RunResolver.resolve(
        experiment=experiment,
        runs=runs,
        redis_host=redis_host,
        redis_port=redis_port,
    )

    if labels:
        known = {rc.run_spec.label for rc in run_configs}
        unknown = [label for label in labels if label not in known]
        if unknown:
            raise click.ClickException(
                f"unknown run label(s): {', '.join(unknown)}. "
                f"Known: {', '.join(sorted(known))}"
            )
        chosen = set(labels)
        run_configs = [rc for rc in run_configs if rc.run_spec.label in chosen]

    _verify_prefixes_exist(ctx, run_configs, redis_host, redis_port)

    return run_configs, redis_host, redis_port


def _verify_prefixes_exist(
    ctx: click.Context, run_configs, redis_host: str, redis_port: int
) -> None:
    """Fail fast with a friendly error if a resolved prefix has no data in Redis.

    Probes the instance-lock key OR any key under the prefix — either
    indicates the run has touched Redis. Missing → ClickException listing
    the prefixes actually present in each DB.

    Skipped entirely when `ctx.obj["redis_factory"]` is set (test affordance)
    or when Redis is unreachable — we don't want infra glitches to mask the
    real export error the user is trying to produce.
    """
    if ctx.obj.get("redis_factory") is not None:
        return

    from redis.exceptions import RedisError

    from gigaevo.cli.inspect_cmd import discover_prefixes

    missing: list[tuple[str, int, list[str]]] = []
    for rc in run_configs:
        spec = rc.run_spec
        if spec.is_disk:
            continue  # existence already verified during disk-path resolution
        try:
            available = discover_prefixes(redis_host, redis_port, spec.db)
        except RedisError:
            return  # infra unreachable → let the main fetch path raise instead
        if spec.prefix in available:
            continue
        import redis as redis_lib

        r = redis_lib.Redis(
            host=redis_host, port=redis_port, db=spec.db, decode_responses=True
        )
        try:
            if (
                next(r.scan_iter(match=f"{spec.prefix}:*", count=1000), None)
                is not None
            ):
                continue
        except RedisError:
            return
        finally:
            r.close()
        missing.append((spec.prefix, spec.db, available))

    if missing:
        lines = [
            f"  {prefix}@{db} — prefixes present in DB {db}: "
            f"{', '.join(avail) if avail else '(none)'}"
            for prefix, db, avail in missing
        ]
        raise click.ClickException("No Redis data found for:\n" + "\n".join(lines))


def _labeled_path(base: Path, label: str) -> Path:
    """Insert a filesystem-safe label between the stem and suffix."""
    safe_label = re.sub(r"[^A-Za-z0-9._@-]+", "_", label).strip("._") or "run"
    return base.with_name(f"{base.stem}_{safe_label}{base.suffix}")


def _fanout_paths(base: Path, labels: list[str]) -> dict[str, Path]:
    """Build non-colliding output paths for a multi-run export."""
    paths = {label: _labeled_path(base, label) for label in labels}
    by_path: dict[Path, list[str]] = {}
    for label, path in paths.items():
        by_path.setdefault(path, []).append(label)
    collisions = [items for items in by_path.values() if len(items) > 1]
    if collisions:
        rendered = "; ".join(
            ", ".join(repr(label) for label in group) for group in collisions
        )
        raise click.ClickException(
            "Run labels produce the same output filename after sanitization: "
            f"{rendered}. Choose distinct alphanumeric labels."
        )
    return paths


def _emit_summary(summaries: list[dict]) -> None:
    """Emit flat dict for single-run, list for multi-run."""
    payload = summaries[0] if len(summaries) == 1 else summaries
    click.echo(json.dumps(payload, indent=2))


@click.group()
def export() -> None:
    """Export evolution data to CSV."""


@export.command("csv")
@click.argument("labels", nargs=-1)
@click.option(
    "-o",
    "--output-file",
    required=True,
    type=click.Path(),
    help=(
        "Output CSV file path. With >1 run in scope, fans out to "
        "<stem>_<label><suffix>."
    ),
)
@click.pass_context
def csv_cmd(ctx: click.Context, labels: tuple[str, ...], output_file: str) -> None:
    """Export full evolution data to CSV.

    \b
    Usage:
      gigaevo -e <exp> export csv -o out.csv            Export all runs (fans out).
      gigaevo -e <exp> export csv <label> -o out.csv    Export one run.
      gigaevo -e <exp> export csv <a> <b> -o out.csv    Export selected runs.
    """
    run_configs, redis_host, redis_port = _resolve_runs(ctx, labels)
    if run_configs is None:
        return

    base = Path(output_file)
    base.parent.mkdir(parents=True, exist_ok=True)
    multi = len(run_configs) > 1
    output_paths = (
        _fanout_paths(base, [rc.run_spec.label for rc in run_configs]) if multi else {}
    )

    summaries: list[dict] = []
    for rc in run_configs:
        df = _fetch_dataframe(rc, redis_host, redis_port)
        df = _serialize_complex_columns(df)
        out_path = output_paths[rc.run_spec.label] if multi else base
        df.to_csv(out_path, index=False)
        summaries.append(
            {
                "label": rc.run_spec.label,
                "output_file": str(out_path),
                "rows": len(df),
                "columns": list(df.columns),
            }
        )

    _emit_summary(summaries)


@export.command("frontier")
@click.argument("labels", nargs=-1)
@click.option(
    "-o",
    "--output-file",
    required=True,
    type=click.Path(),
    help=(
        "Output CSV file path. With >1 run in scope, fans out to "
        "<stem>_<label><suffix>."
    ),
)
@click.option("--metric", default="fitness", help="Metric for frontier values.")
@click.option("--minimize", is_flag=True, default=False, help="Lower is better.")
@click.pass_context
def frontier(
    ctx: click.Context,
    labels: tuple[str, ...],
    output_file: str,
    metric: str,
    minimize: bool,
) -> None:
    """Export the cumulative best value by generation.

    \b
    Usage:
      gigaevo -e <exp> export frontier -o out.csv             All runs (fans out).
      gigaevo -e <exp> export frontier <label> -o out.csv     One run.
      gigaevo -e <exp> export frontier <a> <b> -o out.csv     Selected runs.
    """
    run_configs, redis_host, redis_port = _resolve_runs(ctx, labels)
    if run_configs is None:
        return

    fitness_col = f"metric_{metric}"
    base = Path(output_file)
    base.parent.mkdir(parents=True, exist_ok=True)
    multi = len(run_configs) > 1
    output_paths = (
        _fanout_paths(base, [rc.run_spec.label for rc in run_configs]) if multi else {}
    )

    summaries: list[dict] = []
    for rc in run_configs:
        df = _fetch_dataframe(rc, redis_host, redis_port)
        if fitness_col not in df.columns:
            click.echo(
                f"Error: column {fitness_col} not found (run {rc.run_spec.label})",
                err=True,
            )
            ctx.exit(1)
            return

        gen_col = "generation" if "generation" in df.columns else "iteration"
        grouped = df.groupby(gen_col)[fitness_col]
        frontier_df = (grouped.min() if minimize else grouped.max()).reset_index()
        frontier_df.columns = ["gen", "best_val"]
        frontier_df = frontier_df.sort_values("gen").reset_index(drop=True)
        if minimize:
            frontier_df["best_val"] = frontier_df["best_val"].cummin()
        else:
            frontier_df["best_val"] = frontier_df["best_val"].cummax()

        out_path = output_paths[rc.run_spec.label] if multi else base
        frontier_df.to_csv(out_path, index=False)
        best_value = (
            None if frontier_df.empty else float(frontier_df["best_val"].iloc[-1])
        )
        summaries.append(
            {
                "label": rc.run_spec.label,
                "output_file": str(out_path),
                "generations": len(frontier_df),
                "best_value": best_value,
                "direction": "minimize" if minimize else "maximize",
            }
        )

    _emit_summary(summaries)
