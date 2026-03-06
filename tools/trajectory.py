#!/usr/bin/env python3
"""
Gen-by-gen trajectory dump for a GigaEvo evolution run.

Reads Redis metrics keys directly — lightweight, no full program fetch.
Use at checkpoints during a run or for Phase 5 analysis.

Example usage:
    PYTHONPATH=. python tools/trajectory.py --run chains/hotpotqa/static@4:O
    PYTHONPATH=. python tools/trajectory.py --run chains/hotpotqa/static@4:O --tail 10
"""

import argparse
import json
import sys

import redis as redis_lib

from tools.status import parse_run_arg


def _read_list(r, key: str) -> list[dict]:
    """Read all entries from a Redis list key as parsed JSON dicts."""
    raw_entries = r.lrange(key, 0, -1)
    result = []
    for raw in raw_entries:
        try:
            result.append(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            pass
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Print gen-by-gen trajectory for a GigaEvo evolution run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Output columns:
  Gen      -- MAP-Elites generation number
  best     -- frontier fitness (best valid program seen so far)
  mean     -- mean fitness of valid programs in this gen
  n_valid  -- number of valid programs evaluated in this gen

Summary lines:
  Last improvement -- gen where frontier last jumped, magnitude
  Acceptance rate  -- number of gens (in last 10) where frontier improved / total valid
                      programs in those gens. Numerator is generation-level (0-1 per gen);
                      denominator is program-level (n_valid per gen summed). This is a
                      per-valid-program improvement rate, not a per-mutation rate (invalid
                      programs excluded from denominator).
""",
    )
    parser.add_argument(
        "--run",
        required=True,
        metavar="PREFIX@DB[:LABEL]",
        help="Run spec: prefix@db or prefix@db:label",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=None,
        metavar="N",
        help="Show only the last N gens (default: all)",
    )
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    args = parser.parse_args()

    prefix, db, label = parse_run_arg(args.run)
    r = redis_lib.Redis(host=args.redis_host, port=args.redis_port, db=db)

    try:
        # Per-gen mean fitness: one entry per valid program, s=generation, v=running_mean
        gen_mean_entries = _read_list(
            r, f"{prefix}:metrics:history:program_metrics:valid_gen_fitness_mean"
        )
        # Frontier improvements: one entry per improvement, s=iteration(≈gen), v=frontier_value
        frontier_entries = _read_list(
            r, f"{prefix}:metrics:history:program_metrics:valid_frontier_fitness"
        )
    finally:
        r.close()

    if not gen_mean_entries:
        print(f"No data found for {label} (prefix={prefix}, db={db})")
        sys.exit(1)

    # Build per-gen info from gen_mean_entries.
    # Each valid program in gen g writes ONE entry with s=g, v=running_mean_at_that_point.
    # The last entry with s=g is the final mean for gen g.
    # Count of entries with s=g == n_valid for gen g.
    gen_vals: dict[
        int, list[float]
    ] = {}  # gen -> list of running-mean values (in order)
    for entry in gen_mean_entries:
        g = entry.get("s")
        v = entry.get("v")
        if g is None or v is None:
            continue
        gen_vals.setdefault(int(g), []).append(float(v))

    gen_info: dict[int, dict] = {}
    for g, vals in gen_vals.items():
        gen_info[g] = {
            "mean": vals[-1],  # final running mean = mean of all valid programs in gen
            "n_valid": len(vals),
        }

    # Build frontier_at_gen: for each gen, the best frontier value seen up to that gen.
    # frontier_entries are ordered by time; s=iteration(≈gen), v=fitness.
    frontier_points: list[tuple[int, float]] = []
    for entry in frontier_entries:
        s = entry.get("s")
        v = entry.get("v")
        if s is not None and v is not None:
            frontier_points.append((int(s), float(v)))
    frontier_points.sort(key=lambda x: x[0])

    sorted_gens = sorted(gen_info.keys())
    frontier_at_gen: dict[int, float] = {}
    running_best: float | None = None
    fi = 0
    for gen in sorted_gens:
        while fi < len(frontier_points) and frontier_points[fi][0] <= gen:
            v = frontier_points[fi][1]
            if running_best is None or v > running_best:
                running_best = v
            fi += 1
        if running_best is not None:
            frontier_at_gen[gen] = running_best

    # Apply --tail
    if args.tail is not None and args.tail < len(sorted_gens):
        sorted_gens = sorted_gens[-args.tail :]

    if not sorted_gens:
        print(f"No generation data found for {label}")
        sys.exit(1)

    # Print table
    max_gen = sorted_gens[-1]
    gen_width = max(len(str(max_gen)), 3)

    print(f"Trajectory: {label}  (prefix={prefix}, db={db})")
    print()
    for gen in sorted_gens:
        info = gen_info[gen]
        best_val = frontier_at_gen.get(gen)
        mean_val = info["mean"]
        n_valid = info["n_valid"]
        best_str = f"{best_val * 100:.1f}%" if best_val is not None else "     ?"
        mean_str = f"{mean_val * 100:.1f}%"
        print(
            f"Gen {gen:{gen_width}d}: best={best_str:>6}  mean={mean_str:>6}  n_valid={n_valid:>3}"
        )

    print()

    # Last improvement summary
    # Walk frontier_points to find monotonic improvements
    improvement_points: list[tuple[int, float]] = []
    running_max: float | None = None
    for s, v in frontier_points:
        if running_max is None or v > running_max:
            improvement_points.append((s, v))
            running_max = v

    if len(improvement_points) >= 2:
        last_s, last_v = improvement_points[-1]
        prev_s, prev_v = improvement_points[-2]
        delta = (last_v - prev_v) * 100
        print(
            f"  Last improvement: gen {last_s}"
            f" ({prev_v * 100:.1f}% \u2192 {last_v * 100:.1f}%, +{delta:.1f}pp)"
        )
    elif len(improvement_points) == 1:
        last_s, last_v = improvement_points[0]
        print(f"  Last improvement: gen {last_s} (\u2192 {last_v * 100:.1f}%)")

    # Acceptance rate over last 10 gens:
    # = frontier improvements in that window / total valid programs in that window
    all_gens = sorted(gen_info.keys())
    if len(all_gens) >= 2:
        window = 10
        recent_gens = all_gens[-window:]
        start_gen = recent_gens[0]
        end_gen = recent_gens[-1]

        improvements = 0
        total_valid = 0
        for i, gen in enumerate(recent_gens):
            total_valid += gen_info[gen]["n_valid"]
            if i > 0:
                prev_gen = recent_gens[i - 1]
                if frontier_at_gen.get(gen, 0.0) > frontier_at_gen.get(prev_gen, 0.0):
                    improvements += 1

        if total_valid > 0:
            acc_rate = improvements / total_valid
            print(
                f"  Acceptance rate (gens {start_gen}\u2013{end_gen}):"
                f" {acc_rate * 100:.1f}%"
                f" ({improvements} gens improved / {total_valid} valid programs)"
            )


if __name__ == "__main__":
    main()
