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
    """Query Redis for current run status. Returns dict with gen, best_val_em, keys."""
    r = redis_lib.Redis(host=host, port=port, db=db)
    try:
        total_keys = r.dbsize()

        # Generation count: length of the frontier fitness history list
        hist_key = f"{prefix}:metrics:history:program_metrics:valid_frontier_fitness"
        gen = r.llen(hist_key)

        # Best val EM: last entry in the history list
        best_val_em = None
        raw = r.lindex(hist_key, -1)
        if raw:
            try:
                best_val_em = json.loads(raw)["v"]
            except (KeyError, json.JSONDecodeError):
                pass

        return {
            "gen": gen,
            "best_val_em": best_val_em,
            "total_keys": total_keys,
            "error": None,
        }
    except Exception as e:
        return {"gen": None, "best_val_em": None, "total_keys": None, "error": str(e)}
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
    header = f"{'Run':<8} {'DB':>3}  {'Gen':>5}  {'Best Val EM':>12}  {'Keys':>6}"
    if pid_col:
        header += f"  {'PID':>10}  {'Status':<10}"
    print(header)
    print("-" * len(header))

    # Rows
    for run in runs:
        status = get_run_status(
            run["prefix"], run["db"], args.redis_host, args.redis_port
        )

        gen_str = str(status["gen"]) if status["gen"] is not None else "?"
        em_str = (
            f"{status['best_val_em'] * 100:.1f}%"
            if status["best_val_em"] is not None
            else "?"
        )
        keys_str = (
            str(status["total_keys"]) if status["total_keys"] is not None else "?"
        )

        row = f"{run['label']:<8} {run['db']:>3}  {gen_str:>5}  {em_str:>12}  {keys_str:>6}"

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

        print(row)

    # Watchdog
    if args.watchdog is not None:
        alive = pid_alive(args.watchdog)
        status_str = "✓ ALIVE" if alive else "✗ DEAD"
        print(f"\nWatchdog PID {args.watchdog}: {status_str}")


if __name__ == "__main__":
    main()
