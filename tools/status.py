#!/usr/bin/env python3
"""
Show live status of one or more running (or completed) evolution runs.

Reads Redis directly — safe to run at any time, no side effects.

Example usage:
    # One run, no PID check
    PYTHONPATH=. python tools/status.py --run chains/hotpotqa/static@0:K

    # Multiple runs with PID and watchdog liveness check
    PYTHONPATH=. python tools/status.py \\
        --run chains/hotpotqa/static@0:K \\
        --run chains/hotpotqa/static_r@1:L \\
        --run chains/hotpotqa/static_r@2:M \\
        --run chains/hotpotqa/static_r@3:N \\
        --pid K:2616605 --pid L:2616606 --pid M:2616607 --pid N:2616608 \\
        --watchdog 2716169

Run format: prefix@db[:label]  (same as top_programs.py and comparison.py)
PID format: label:pid           (label must match the run label)
"""

import argparse
import json
import os
import textwrap

import redis as redis_lib


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def get_run_status(
    prefix: str, db: int, host: str = "localhost", port: int = 6379
) -> dict:
    """Query Redis for current run status.

    Returns dict with:
      gen             -- MAP-Elites iteration (generation) number
      best_val_fitness     -- best frontier fitness seen so far (val F1 for F1 runs, val EM for EM runs)
      total_keys      -- total Redis keys in this DB
      total_programs  -- total programs evaluated (valid + invalid)
      valid_programs  -- programs that passed validation
      invalid_rate    -- fraction invalid (0.0–1.0); high value = timeout/crash problem
      validator_mean_s -- mean validator stage duration in seconds (None if no data)
      validator_max_s  -- max validator stage duration in seconds (None if no data)
      error           -- exception message if query failed, else None
    """
    r = redis_lib.Redis(host=host, port=port, db=db)
    try:
        total_keys = r.dbsize()

        # Generation count: read from run_state hash persisted by EvolutionEngine.
        # Written after every generation; survives restarts (fix/redis-resume).
        gen = None
        raw_gen = r.hget(f"{prefix}:run_state", "engine:total_generations")
        if raw_gen:
            try:
                gen = int(raw_gen)
            except ValueError:
                pass

        # Best val fitness: last entry in the frontier fitness history list
        hist_key = f"{prefix}:metrics:history:program_metrics:valid_frontier_fitness"
        best_val_fitness = None
        raw = r.lindex(hist_key, -1)
        if raw:
            try:
                best_val_fitness = json.loads(raw)["v"]
            except (KeyError, json.JSONDecodeError):
                pass

        # Invalidity rate: catches timeout/crash problems early.
        # If invalid_rate > 0.5 at gen 3+, something is wrong (e.g. stage_timeout too short).
        total_programs, valid_programs, invalid_rate = None, None, None
        raw_total = r.lindex(
            f"{prefix}:metrics:history:program_metrics:programs_total_count", -1
        )
        raw_valid = r.lindex(
            f"{prefix}:metrics:history:program_metrics:programs_valid_count", -1
        )
        if raw_total and raw_valid:
            try:
                total_programs = int(json.loads(raw_total)["v"])
                valid_programs = int(json.loads(raw_valid)["v"])
                if total_programs > 0:
                    invalid_rate = (total_programs - valid_programs) / total_programs
            except (KeyError, json.JSONDecodeError, ValueError):
                pass

        # Validator stage duration: detect stage_timeout pressure.
        # Reads the last 20 successful validator durations and summarises them.
        validator_mean_s, validator_max_s = None, None
        dur_key = f"{prefix}:metrics:history:dag_runner:dag:internals:CallValidatorFunction:stage_duration"
        if r.type(dur_key) == b"list":
            recent = r.lrange(dur_key, -20, -1)
            durations = []
            for raw_d in recent:
                try:
                    v = json.loads(raw_d)["v"]
                    if v is not None:
                        durations.append(float(v))
                except (KeyError, json.JSONDecodeError, ValueError):
                    pass
            if durations:
                validator_mean_s = sum(durations) / len(durations)
                validator_max_s = max(durations)

        return {
            "gen": gen,
            "best_val_fitness": best_val_fitness,
            "total_keys": total_keys,
            "total_programs": total_programs,
            "valid_programs": valid_programs,
            "invalid_rate": invalid_rate,
            "validator_mean_s": validator_mean_s,
            "validator_max_s": validator_max_s,
            "error": None,
        }
    except Exception as e:
        return {
            "gen": None,
            "best_val_fitness": None,
            "total_keys": None,
            "total_programs": None,
            "valid_programs": None,
            "invalid_rate": None,
            "validator_mean_s": None,
            "validator_max_s": None,
            "error": str(e),
        }
    finally:
        r.close()


def parse_run_arg(arg: str) -> tuple[str, int, str]:
    """Parse 'prefix@db[:label]' → (prefix, db, label)."""
    label = None
    if ":" in arg:
        # Could be prefix@db:label or prefix:with:colons@db:label
        # Split on last occurrence of '@' then ':' on the right part
        at_idx = arg.rfind("@")
        if at_idx == -1:
            raise ValueError(f"--run must contain '@': got {arg!r}")
        prefix = arg[:at_idx]
        rest = arg[at_idx + 1 :]
        if ":" in rest:
            db_str, label = rest.split(":", 1)
        else:
            db_str = rest
    else:
        raise ValueError(f"--run must be prefix@db[:label], got {arg!r}")
    return prefix, int(db_str), label or f"{prefix}@{db_str}"


def parse_pid_arg(arg: str) -> tuple[str, int]:
    """Parse 'label:pid' → (label, pid)."""
    if ":" not in arg:
        raise ValueError(f"--pid must be label:pid, got {arg!r}")
    label, pid_str = arg.split(":", 1)
    return label, int(pid_str)


def main():
    parser = argparse.ArgumentParser(
        description="Show live status of GigaEvo evolution runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              %(prog)s --run chains/hotpotqa/static@0:K
              %(prog)s \\
                --run chains/hotpotqa/static@0:K \\
                --run chains/hotpotqa/static_r@1:L \\
                --pid K:2616605 --pid L:2616606 \\
                --watchdog 2716169
        """),
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="PREFIX@DB[:LABEL]",
        help="Run spec (repeatable). Format: prefix@db or prefix@db:label",
    )
    parser.add_argument(
        "--pid",
        action="append",
        default=[],
        metavar="LABEL:PID",
        help="PID to check for a run label (repeatable). Format: label:pid",
    )
    parser.add_argument(
        "--watchdog",
        type=int,
        default=None,
        metavar="PID",
        help="Watchdog PID to check liveness",
    )
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)

    args = parser.parse_args()

    # Parse runs
    runs = []
    for r in args.run:
        prefix, db, label = parse_run_arg(r)
        runs.append({"prefix": prefix, "db": db, "label": label})

    # Parse PIDs
    pid_map: dict[str, int] = {}
    for p in args.pid:
        label, pid = parse_pid_arg(p)
        pid_map[label] = pid

    # Header
    pid_col = bool(pid_map)
    header = (
        f"{'Run':<8} {'DB':>3}  {'Gen':>5}  {'Best Val':>10}"
        f"  {'Invalid%':>8}  {'Val dur(s)':>12}  {'Keys':>6}"
    )
    if pid_col:
        header += f"  {'PID':>10}  {'Status':<10}"
    print(header)
    print("-" * len(header))

    warnings = []

    # Rows
    for run in runs:
        status = get_run_status(
            run["prefix"], run["db"], args.redis_host, args.redis_port
        )

        gen_str = str(status["gen"]) if status["gen"] is not None else "?"
        em_str = (
            f"{status['best_val_fitness'] * 100:.1f}%"
            if status["best_val_fitness"] is not None
            else "?"
        )
        inv_str = (
            f"{status['invalid_rate'] * 100:.0f}%"
            if status["invalid_rate"] is not None
            else "?"
        )
        dur_str = (
            f"{status['validator_mean_s']:.0f}/{status['validator_max_s']:.0f}"
            if status["validator_mean_s"] is not None
            else "?"
        )
        keys_str = (
            str(status["total_keys"]) if status["total_keys"] is not None else "?"
        )

        row = (
            f"{run['label']:<8} {run['db']:>3}  {gen_str:>5}  {em_str:>10}"
            f"  {inv_str:>8}  {dur_str:>12}  {keys_str:>6}"
        )

        if pid_col:
            pid = pid_map.get(run["label"])
            if pid is not None:
                alive = pid_alive(pid)
                status_str = "✓ ALIVE" if alive else "✗ DEAD"
                row += f"  {pid:>10}  {status_str:<10}"
            else:
                row += f"  {'—':>10}  {'—':<10}"

        if status["error"]:
            row += f"  [ERROR: {status['error']}]"

        # Warn if invalidity rate is critically high (likely stage_timeout too short).
        # Normal mutation invalidity is 20–50% (bad code, runtime errors).
        # Above 75% at gen 3+ almost always means stage_timeout < actual eval time.
        gen = status["gen"] or 0
        inv = status["invalid_rate"]
        if inv is not None and inv > 0.75 and gen >= 3:
            warnings.append(
                f"  ⚠  Run {run['label']}: {inv * 100:.0f}% invalid programs at gen {gen}"
                f" — stage_timeout is likely too short for this eval workload"
            )

        print(row)

    if warnings:
        print()
        for w in warnings:
            print(w)

    # Watchdog
    if args.watchdog is not None:
        alive = pid_alive(args.watchdog)
        status_str = "✓ ALIVE" if alive else "✗ DEAD"
        print(f"\nWatchdog PID {args.watchdog}: {status_str}")


if __name__ == "__main__":
    main()
