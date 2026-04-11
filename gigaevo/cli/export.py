"""Export sub-group: csv and frontier CSV export commands."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click
import pandas as pd

from gigaevo.cli.run_resolver import RunResolver


def _build_redis_config(run_config, redis_host: str, redis_port: int):
    """Build a RedisRunConfig from a monitoring RunConfig."""
    from gigaevo.utils.redis import RedisRunConfig

    spec = run_config.run_spec
    return RedisRunConfig(
        redis_host=redis_host,
        redis_port=redis_port,
        redis_db=spec.db,
        redis_prefix=spec.prefix,
        label=spec.label,
    )


def _fetch_dataframe(run_config, redis_host: str, redis_port: int) -> pd.DataFrame:
    """Fetch evolution DataFrame for a single run."""
    from tools.utils import fetch_evolution_dataframe

    config = _build_redis_config(run_config, redis_host, redis_port)
    return asyncio.run(fetch_evolution_dataframe(config, add_stage_results=False))


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


@click.group()
def export() -> None:
    """Export evolution data to CSV."""


@export.command("csv")
@click.option(
    "-o",
    "--output-file",
    required=True,
    type=click.Path(),
    help="Output CSV file path.",
)
@click.pass_context
def csv_cmd(ctx: click.Context, output_file: str) -> None:
    """Export full evolution data to CSV."""
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

    rc = run_configs[0]
    df = _fetch_dataframe(rc, redis_host, redis_port)
    df = _serialize_complex_columns(df)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    summary = {
        "output_file": str(out_path),
        "rows": len(df),
        "columns": list(df.columns),
    }
    click.echo(json.dumps(summary, indent=2))


@export.command("frontier")
@click.option(
    "-o",
    "--output-file",
    required=True,
    type=click.Path(),
    help="Output CSV file path.",
)
@click.option("--metric", default="fitness", help="Metric for frontier values.")
@click.pass_context
def frontier(ctx: click.Context, output_file: str, metric: str) -> None:
    """Export frontier-only CSV with gen and best_val columns."""
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

    rc = run_configs[0]
    df = _fetch_dataframe(rc, redis_host, redis_port)

    fitness_col = f"metric_{metric}"
    if fitness_col not in df.columns:
        click.echo(f"Error: column {fitness_col} not found", err=True)
        ctx.exit(1)
        return

    gen_col = "generation"
    if gen_col not in df.columns:
        gen_col = "metadata_iteration"

    frontier_df = df.groupby(gen_col)[fitness_col].max().reset_index()
    frontier_df.columns = ["gen", "best_val"]
    frontier_df = frontier_df.sort_values("gen").reset_index(drop=True)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frontier_df.to_csv(out_path, index=False)

    summary = {
        "output_file": str(out_path),
        "generations": len(frontier_df),
        "best_value": float(frontier_df["best_val"].max()),
    }
    click.echo(json.dumps(summary, indent=2))
