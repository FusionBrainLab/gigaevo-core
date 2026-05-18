import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.append("../gigaevo-core-internal")

import pandas as pd

from tools.status import parse_run_arg
from tools.utils import (
    RedisRunConfig,
    fetch_evolution_dataframe,
    prepare_iteration_dataframe,
)


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` via tmpfile + os.replace.

    Avoids leaving a half-written CSV when two writers target the same
    output or when the process is interrupted mid-flush.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _serialize_complex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Serialize dict/list columns to JSON strings for safe CSV roundtripping.

    Pandas' default `str()` conversion produces Python repr (single quotes,
    True/False, etc.) which cannot be reliably parsed back.  `json.dumps`
    escapes newlines, quotes, and backslashes so the value stays in a single
    CSV cell and can be restored with `json.loads`.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
            df[col] = df[col].apply(
                lambda x: (
                    json.dumps(x, ensure_ascii=False)
                    if isinstance(x, (dict, list))
                    else x
                )
            )
    return df


async def export_redis_run_to_csv(
    config: RedisRunConfig,
    output_file: str | Path,
    *,
    add_stage_results: bool = False,
) -> Path:
    output_path = Path(output_file)

    df: pd.DataFrame = await fetch_evolution_dataframe(
        config, add_stage_results=add_stage_results
    )
    df = _serialize_complex_columns(df)
    _atomic_write_csv(df, output_path)
    return output_path


async def main():
    parser = argparse.ArgumentParser(
        description="Export Redis evolution data to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Single-argument run spec
  PYTHONPATH=. python tools/redis2pd.py --run chains/hotpotqa/static@4:O --output-file /tmp/o.csv

  # Frontier-only CSV (gen,best_val) for 05_results.md tables
  PYTHONPATH=. python tools/redis2pd.py --run chains/hotpotqa/static@4:O \\
      --frontier-csv --output-file /tmp/frontier_o.csv

  # Split prefix/db form (used by archive_run.sh)
  PYTHONPATH=. python tools/redis2pd.py --redis-db 4 --redis-prefix chains/hotpotqa/static \\
      --output-file /tmp/o.csv
""",
    )
    # Combined run-spec argument
    parser.add_argument(
        "--run",
        metavar="PREFIX@DB[:LABEL]",
        help="Run spec: prefix@db or prefix@db:label (takes precedence over --redis-db/--redis-prefix)",
    )
    # Split-form run arguments (consumed when --run is not provided)
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--redis-db", type=int, help="Redis database (paired with --redis-prefix)")
    parser.add_argument(
        "--redis-prefix", type=str, help="Redis prefix (paired with --redis-db)"
    )
    parser.add_argument(
        "--output-file", type=str, required=True, help="Output CSV file path"
    )
    parser.add_argument(
        "--frontier-csv",
        action="store_true",
        help=(
            "Emit a compact gen,best_val CSV (frontier only) instead of the full program history. "
            "Useful for 05_results.md tables and comparison.py input."
        ),
    )
    args = parser.parse_args()

    # Resolve run config: --run takes precedence over the split --redis-db / --redis-prefix pair
    if args.run:
        prefix, db, label = parse_run_arg(args.run)
        config = RedisRunConfig(
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            redis_db=db,
            redis_prefix=prefix,
            label=label,
        )
    elif args.redis_db is not None and args.redis_prefix is not None:
        config = RedisRunConfig(
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            redis_db=args.redis_db,
            redis_prefix=args.redis_prefix,
            label=args.output_file,
        )
    else:
        parser.error(
            "Provide either --run PREFIX@DB[:LABEL] or both --redis-db and --redis-prefix"
        )

    df: pd.DataFrame = await fetch_evolution_dataframe(config, add_stage_results=False)

    if df.empty:
        print(f"No data found for {config.display_label()}")
        return

    output_path = Path(args.output_file)
    if args.frontier_csv:
        prepared = prepare_iteration_dataframe(df)
        if prepared.empty:
            print("No valid iteration/fitness data after filtering")
            return
        # One row per gen: take the last frontier_fitness per iteration
        iteration_col = "metadata_iteration"
        frontier_col = "frontier_fitness"
        frontier_df = (
            prepared.groupby(iteration_col, as_index=False)[frontier_col]
            .last()
            .sort_values(iteration_col)
            .rename(columns={iteration_col: "gen", frontier_col: "best_val"})
        )
        _atomic_write_csv(frontier_df, output_path)
        print(f"Frontier CSV: {len(frontier_df)} gens → {output_path}")
    else:
        df = _serialize_complex_columns(df)
        _atomic_write_csv(df, output_path)
        print(f"Full history: {len(df)} programs → {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
