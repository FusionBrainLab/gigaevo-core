#!/usr/bin/env python3
"""
Dump the top-N programs from a running (or completed) evolution run.

Connects to Redis read-only, sorts programs by fitness, and prints a
summary table plus optionally the full source code of each program.

Example usage:
    # Quick overview — top 5 programs
    PYTHONPATH=. python tools/top_programs.py --run chains/hotpotqa/static@5

    # Top 10 with full code
    PYTHONPATH=. python tools/top_programs.py --run chains/hotpotqa/static@5 -n 10 --code

    # Save codes to files
    PYTHONPATH=. python tools/top_programs.py --run chains/hotpotqa/static@5 -n 3 --save-dir top_programs/

    # Minimization problem
    PYTHONPATH=. python tools/top_programs.py --run heilbron@0 --minimize

    # Filter to DONE programs only, show all metrics
    PYTHONPATH=. python tools/top_programs.py --run chains/hotpotqa/static@5 --state done --all-metrics

    # JSON output for scripting
    PYTHONPATH=. python tools/top_programs.py --run chains/hotpotqa/static@5 -n 3 --json

Run format: prefix@db[:label]  (same as comparison.py)
"""

import argparse
import asyncio
import json
from pathlib import Path
import sys
import textwrap

from loguru import logger

from gigaevo.database.redis_program_storage import (
    RedisProgramStorage,
    RedisProgramStorageConfig,
)
from gigaevo.programs.program import Program
from tools.utils import RedisRunConfig


def _parse_run_arg(arg: str, default_host: str, default_port: int) -> RedisRunConfig:
    """Parse --run argument of the form prefix@db[:label]."""
    label: str | None = None
    if ":" in arg:
        prefix_db, label = arg.split(":", 1)
    else:
        prefix_db = arg

    if "@" not in prefix_db:
        raise ValueError("--run format must be prefix@db[:label]")
    prefix, db_str = prefix_db.split("@", 1)
    db = int(db_str)
    return RedisRunConfig(
        redis_host=default_host,
        redis_port=default_port,
        redis_db=db,
        redis_prefix=prefix,
        label=label,
    )


async def fetch_programs(config: RedisRunConfig) -> list[Program]:
    """Fetch all programs from Redis (read-only)."""
    storage = RedisProgramStorage(
        RedisProgramStorageConfig(
            redis_url=config.url(),
            key_prefix=config.redis_prefix,
            max_connections=50,
            connection_pool_timeout=30.0,
            health_check_interval=60,
            read_only=True,
        )
    )
    try:
        return await storage.get_all()
    finally:
        await storage.close()


def rank_programs(
    programs: list[Program],
    *,
    metric: str = "fitness",
    minimize: bool = False,
    state_filter: str | None = None,
    top_n: int = 5,
) -> list[Program]:
    """Filter and rank programs by metric value."""
    # Filter by state if requested
    if state_filter:
        programs = [p for p in programs if p.state.value == state_filter]

    # Filter to programs that have the requested metric
    programs = [
        p for p in programs if metric in p.metrics and p.metrics[metric] is not None
    ]

    # Sort
    programs.sort(key=lambda p: p.metrics[metric], reverse=not minimize)

    return programs[:top_n]


def format_table(
    programs: list[Program],
    *,
    metric: str = "fitness",
    all_metrics: bool = False,
) -> str:
    """Format programs as a readable table."""
    if not programs:
        return "  (no programs found)"

    lines = []

    # Header
    header_parts = [
        f"{'#':>3}",
        f"{'ID':>10}",
        f"{'State':>10}",
        f"{'Gen':>4}",
        f"{metric:>12}",
    ]
    if all_metrics:
        # Collect all metric keys across programs
        all_keys = set()
        for p in programs:
            all_keys.update(p.metrics.keys())
        all_keys.discard(metric)
        extra_keys = sorted(all_keys)
        for k in extra_keys:
            header_parts.append(f"{k:>12}")
    header_parts.extend([f"{'Parents':>8}", f"{'Children':>9}", f"{'Created':>20}"])

    lines.append("  ".join(header_parts))
    lines.append("  ".join("-" * len(h) for h in header_parts))

    for i, p in enumerate(programs, 1):
        row = [
            f"{i:>3}",
            f"{p.id[:10]:>10}",
            f"{p.state.value:>10}",
            f"{p.generation:>4}",
            f"{p.metrics.get(metric, float('nan')):>12.6g}",
        ]
        if all_metrics:
            for k in extra_keys:
                val = p.metrics.get(k)
                if val is not None:
                    row.append(f"{val:>12.6g}")
                else:
                    row.append(f"{'—':>12}")
        row.extend(
            [
                f"{len(p.lineage.parents):>8}",
                f"{len(p.lineage.children):>9}",
                f"{p.created_at.strftime('%Y-%m-%d %H:%M:%S'):>20}",
            ]
        )
        lines.append("  ".join(row))

    return "\n".join(lines)


def format_program_detail(rank: int, p: Program, *, metric: str = "fitness") -> str:
    """Format a single program with full code."""
    sep = "=" * 80
    lines = [
        sep,
        f"  #{rank}  {p.id}",
        f"  State: {p.state.value}  |  Generation: {p.generation}  |  "
        f"{metric}: {p.metrics.get(metric, 'N/A')}",
        f"  All metrics: {p.metrics}",
        f"  Parents: {p.lineage.parents}",
        f"  Mutation: {p.lineage.mutation or '(root)'}",
        f"  Created: {p.created_at}",
        sep,
        p.code,
        "",
    ]
    return "\n".join(lines)


def format_json(programs: list[Program], *, metric: str = "fitness") -> str:
    """Format programs as JSON for scripting."""
    entries = []
    for i, p in enumerate(programs, 1):
        entries.append(
            {
                "rank": i,
                "id": p.id,
                "state": p.state.value,
                "generation": p.generation,
                "metrics": p.metrics,
                "parents": p.lineage.parents,
                "mutation": p.lineage.mutation,
                "created_at": p.created_at.isoformat(),
                "code": p.code,
            }
        )
    return json.dumps(entries, indent=2)


def save_programs(
    programs: list[Program], save_dir: Path, *, metric: str = "fitness"
) -> None:
    """Save each program's code to a separate file."""
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, p in enumerate(programs, 1):
        fitness_val = p.metrics.get(metric, 0)
        filename = f"rank{i:02d}_{fitness_val:.4f}_{p.id[:8]}.py"
        filepath = save_dir / filename
        filepath.write_text(p.code)
        logger.info(f"Saved #{i} to {filepath}")

    # Also save a summary JSON
    summary_path = save_dir / "summary.json"
    summary_path.write_text(format_json(programs, metric=metric))
    logger.info(f"Saved summary to {summary_path}")


async def main():
    parser = argparse.ArgumentParser(
        description="Dump top-N programs from an evolution run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              %(prog)s --run chains/hotpotqa/static@5
              %(prog)s --run chains/hotpotqa/static@5 -n 10 --code
              %(prog)s --run chains/hotpotqa/static@5 -n 3 --save-dir top_programs/
              %(prog)s --run heilbron@0 --minimize --metric fitness
        """),
    )

    # Connection
    parser.add_argument(
        "--run",
        required=True,
        help="Run spec: prefix@db[:label] (same format as comparison.py)",
    )
    parser.add_argument(
        "--redis-host", default="localhost", help="Redis host (default: localhost)"
    )
    parser.add_argument(
        "--redis-port", type=int, default=6379, help="Redis port (default: 6379)"
    )

    # Selection
    parser.add_argument(
        "-n",
        "--top-n",
        type=int,
        default=5,
        help="Number of top programs to show (default: 5)",
    )
    parser.add_argument(
        "--metric",
        default="fitness",
        help="Metric to rank by (default: fitness)",
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Lower metric is better (default: maximize)",
    )
    parser.add_argument(
        "--state",
        default=None,
        choices=["queued", "running", "done", "discarded"],
        help="Filter to programs in this state only",
    )

    # Output format
    parser.add_argument(
        "--code",
        action="store_true",
        help="Print full source code for each program",
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="Show all metrics in the table (not just the ranking metric)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output as JSON (for scripting)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Save each program's code to this directory",
    )

    args = parser.parse_args()

    # Parse run config
    config = _parse_run_arg(args.run, args.redis_host, args.redis_port)

    # Fetch programs
    logger.info(f"Fetching programs from {config.url()} prefix={config.redis_prefix!r}")
    all_programs = await fetch_programs(config)
    logger.info(f"Loaded {len(all_programs)} total programs")

    # State breakdown
    from collections import Counter

    state_counts = Counter(p.state.value for p in all_programs)
    logger.info(f"State breakdown: {dict(state_counts)}")

    # Rank
    ranked = rank_programs(
        all_programs,
        metric=args.metric,
        minimize=args.minimize,
        state_filter=args.state,
        top_n=args.top_n,
    )

    if not ranked:
        logger.warning("No programs found matching criteria")
        sys.exit(1)

    direction = "min" if args.minimize else "max"
    print(
        f"\nTop {len(ranked)} programs by {args.metric} ({direction}imize), "
        f"from {config.display_label()}:\n"
    )

    # Output
    if args.json_output:
        print(format_json(ranked, metric=args.metric))
    else:
        print(format_table(ranked, metric=args.metric, all_metrics=args.all_metrics))

        if args.code:
            print()
            for i, p in enumerate(ranked, 1):
                print(format_program_detail(i, p, metric=args.metric))

    # Save to files if requested
    if args.save_dir:
        save_programs(ranked, Path(args.save_dir), metric=args.metric)

    return ranked


if __name__ == "__main__":
    asyncio.run(main())
