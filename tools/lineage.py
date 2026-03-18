#!/usr/bin/env python3
"""
Trace evolutionary ancestry of a program back to its root seed.

Reads all programs from Redis and walks the parent chain for the
requested program(s). Useful for Phase 5 "Lessons Learned" — which
mutations drove the best result?

Example usage:
    # Trace best program by fitness
    PYTHONPATH=. python tools/lineage.py --run chains/hotpotqa/static@4:O --top-n 1

    # Trace specific program by ID prefix
    PYTHONPATH=. python tools/lineage.py --run chains/hotpotqa/static@4:O --program abc12345

    # Top 3 programs, each traced back up to 5 ancestor hops
    PYTHONPATH=. python tools/lineage.py --run chains/hotpotqa/static@4:O --top-n 3 --depth 5
"""

import argparse
import asyncio
from pathlib import Path
import sys

PROJ = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ))

from gigaevo.database.redis_program_storage import (  # noqa: E402
    RedisProgramStorage,
    RedisProgramStorageConfig,
)
from gigaevo.programs.program import Program  # noqa: E402
from tools.status import parse_run_arg  # noqa: E402


async def _fetch_programs(url: str, prefix: str) -> list[Program]:
    storage = RedisProgramStorage(
        RedisProgramStorageConfig(
            redis_url=url,
            key_prefix=prefix,
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


def _build_id_map(programs: list[Program]) -> dict[str, Program]:
    return {p.id: p for p in programs}


def _resolve_program(id_prefix: str, id_map: dict[str, Program]) -> Program:
    """Resolve a program by exact or prefix match. Raises if ambiguous or not found."""
    exact = id_map.get(id_prefix)
    if exact is not None:
        return exact
    matches = [p for pid, p in id_map.items() if pid.startswith(id_prefix)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise ValueError(f"No program with ID prefix {id_prefix!r} found.")
    ids = [p.id[:12] for p in matches]
    raise ValueError(
        f"Ambiguous ID prefix {id_prefix!r} — {len(matches)} matches: {ids}"
    )


def _get_fitness(p: Program, metric: str) -> float | None:
    return p.metrics.get(metric)


def _get_generation(p: Program) -> int:
    return p.lineage.generation


def _format_fitness(p: Program, metric: str) -> str:
    v = _get_fitness(p, metric)
    if v is None:
        return "fitness=N/A"
    return f"fitness={v * 100:.1f}%"


def _walk_lineage(
    start: Program,
    id_map: dict[str, Program],
    depth: int | None,
    metric: str,
) -> list[Program]:
    """Walk ancestor chain from start to root. Returns chain [start, parent, grandparent, ...]."""
    chain: list[Program] = [start]
    current = start
    hops = 0
    while True:
        if depth is not None and hops >= depth:
            break
        parents = current.lineage.parents
        if not parents:
            break  # root reached
        # Handle multi-parent case: follow first parent, note merge
        parent_id = parents[0]
        parent = id_map.get(parent_id)
        if parent is None:
            break  # parent not in Redis (e.g. external seed)
        chain.append(parent)
        current = parent
        hops += 1
    return chain


def _print_lineage(
    chain: list[Program],
    label: str,
    metric: str,
) -> None:
    """Print a lineage chain."""
    tip = chain[0]
    tip_id = tip.id[:8]
    tip_gen = _get_generation(tip)
    tip_fit = _format_fitness(tip, metric)
    print(f"Lineage of program {tip_id} (run {label}, gen {tip_gen}, {tip_fit})")
    print()

    for p in chain:
        pid = p.id[:8]
        gen = _get_generation(p)
        fit = _format_fitness(p, metric)

        mut = p.lineage.mutation or ""
        parents = p.lineage.parents
        extra_parents = parents[1:] if len(parents) > 1 else []

        if p.lineage.is_root():
            # Seed — try to include a short hint from the mutation description or name
            seed_hint = p.name or "(no name)"
            print(f"  gen {gen:>3}  {pid}  {fit}  [SEED — {seed_hint}]")
        else:
            mut_display = f"mutation: {mut}" if mut else "(no mutation label)"
            merge_note = ""
            if extra_parents:
                merge_note = f"  [+{len(extra_parents)} more parent(s)]"
            print(f"  gen {gen:>3}  {pid}  {fit}  {mut_display}{merge_note}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace evolutionary ancestry of a GigaEvo program",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Reads all programs from Redis and walks the parent chain. Fetches all programs
in memory — use on a run that is complete or paused; avoid on very large runs
if memory is constrained.

--top-n and --program are mutually exclusive.
""",
    )
    parser.add_argument(
        "--run",
        required=True,
        metavar="PREFIX@DB[:LABEL]",
        help="Run spec: prefix@db or prefix@db:label",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--top-n",
        type=int,
        default=1,
        metavar="N",
        help="Trace top-N programs by fitness (default: 1)",
    )
    group.add_argument(
        "--program",
        metavar="ID_PREFIX",
        help="Trace specific program by ID or ID prefix (unambiguous)",
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        metavar="N",
        help="Max ancestor hops (default: unlimited, traces to root)",
    )
    parser.add_argument(
        "--metric",
        default="fitness",
        help="Fitness metric to rank and display by (default: fitness)",
    )
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    args = parser.parse_args()

    prefix, db, label = parse_run_arg(args.run)
    url = f"redis://{args.redis_host}:{args.redis_port}/{db}"

    programs = asyncio.run(_fetch_programs(url, prefix))
    if not programs:
        print(f"No programs found for {label} (prefix={prefix}, db={db})")
        sys.exit(1)

    id_map = _build_id_map(programs)

    # Determine which programs to trace
    if args.program:
        try:
            start_programs = [_resolve_program(args.program, id_map)]
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Top-N by metric
        scored = [p for p in programs if _get_fitness(p, args.metric) is not None]
        scored.sort(key=lambda p: _get_fitness(p, args.metric), reverse=True)
        start_programs = scored[: args.top_n]

    if not start_programs:
        print(f"No programs with metric '{args.metric}' found.", file=sys.stderr)
        sys.exit(1)

    for i, start in enumerate(start_programs):
        if i > 0:
            print("---")
        chain = _walk_lineage(start, id_map, args.depth, args.metric)
        _print_lineage(chain, label, args.metric)
        if args.depth is not None and len(chain) > args.depth:
            print(f"  ... (truncated at depth {args.depth})")
        print()


if __name__ == "__main__":
    main()
