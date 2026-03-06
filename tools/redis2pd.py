import argparse
import asyncio

import pandas as pd

from tools.status import parse_run_arg
from tools.utils import (
    RedisRunConfig,
    fetch_evolution_dataframe,
    prepare_iteration_dataframe,
)


async def main():
    parser = argparse.ArgumentParser(
        description="Export Redis evolution data to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # New format (preferred)
  PYTHONPATH=. python tools/redis2pd.py --run chains/hotpotqa/static@4:O --output-file /tmp/o.csv

  # Frontier-only CSV (gen,best_val) for 05_results.md tables
  PYTHONPATH=. python tools/redis2pd.py --run chains/hotpotqa/static@4:O \\
      --frontier-csv --output-file /tmp/frontier_o.csv

  # Legacy format (still works, used by archive_run.sh)
  PYTHONPATH=. python tools/redis2pd.py --redis-db 4 --redis-prefix chains/hotpotqa/static \\
      --output-file /tmp/o.csv
""",
    )
    # New unified format
    parser.add_argument(
        "--run",
        metavar="PREFIX@DB[:LABEL]",
        help="Run spec: prefix@db or prefix@db:label (takes precedence over --redis-db/--redis-prefix)",
    )
    # Legacy args (still supported for archive_run.sh compatibility)
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument(
        "--redis-db", type=int, help="Redis database [legacy; prefer --run]"
    )
    parser.add_argument(
        "--redis-prefix", type=str, help="Redis prefix [legacy; prefer --run]"
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

    # Resolve run config: --run takes precedence over legacy --redis-db / --redis-prefix
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
        frontier_df.to_csv(args.output_file, index=False)
        print(f"Frontier CSV: {len(frontier_df)} gens → {args.output_file}")
    else:
        df.to_csv(args.output_file, index=False)
        print(f"Full history: {len(df)} programs → {args.output_file}")


if __name__ == "__main__":
    asyncio.run(main())
