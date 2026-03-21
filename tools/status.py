#!/usr/bin/env python3
"""
Show live status of one or more running (or completed) evolution runs.

Reads Redis directly — safe to run at any time, no side effects.

Example usage:
    # From experiment manifest (recommended):
    PYTHONPATH=. python tools/status.py --experiment hover/prompt_coevolution

    # One run, no PID check
    PYTHONPATH=. python tools/status.py --run chains/hotpotqa/static@0:K

    # Multiple runs with PID and watchdog liveness check
    PYTHONPATH=. python tools/status.py \\
        --run chains/hotpotqa/static@0:K \\
        --run chains/hotpotqa/static_r@1:L \\
        --pid K:2616605 --pid L:2616606 \\
        --watchdog 2716169

Run format: prefix@db[:label]  (same as top_programs.py and comparison.py)
PID format: label:pid           (label must match the run label)
"""

import argparse
import json
import os
from pathlib import Path
import textwrap

import redis as redis_lib
import yaml

PROJ = Path(__file__).resolve().parent.parent


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def load_metrics_yaml(problem_name: str) -> dict[str, dict]:
    """Load metrics.yaml for a problem, return {metric_name: spec_dict}."""
    path = PROJ / "problems" / problem_name / "metrics.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    return data.get("specs", {})


def get_metric_names(specs: dict[str, dict]) -> list[str]:
    """Return metric names from specs, primary first, excluding is_valid."""
    primary = []
    secondary = []
    for name, spec in specs.items():
        if name == "is_valid":
            continue
        if spec.get("is_primary", False):
            primary.append(name)
        else:
            secondary.append(name)
    return primary + secondary


def get_run_status(
    prefix: str,
    db: int,
    metric_names: list[str] | None = None,
    host: str = "localhost",
    port: int = 6379,
) -> dict:
    """Query Redis for current run status.

    Returns dict with:
      gen             -- MAP-Elites iteration (generation) number
      metrics         -- {metric_name: frontier_value} for each metric
      total_keys      -- total Redis keys in this DB
      total_programs  -- total programs evaluated (valid + invalid)
      valid_programs  -- programs that passed validation
      invalid_rate    -- fraction invalid (0.0-1.0)
      validator_mean_s -- mean validator stage duration in seconds
      validator_max_s  -- max validator stage duration in seconds
      error           -- exception message if query failed, else None
    """
    if metric_names is None:
        metric_names = ["fitness"]

    r = redis_lib.Redis(host=host, port=port, db=db)
    try:
        total_keys = r.dbsize()

        # Generation count
        gen = None
        raw_gen = r.hget(f"{prefix}:run_state", "engine:total_generations")
        if raw_gen:
            try:
                gen = int(raw_gen)
            except ValueError:
                pass

        # Read frontier value for each metric
        metrics = {}
        for name in metric_names:
            hist_key = f"{prefix}:metrics:history:program_metrics:valid_frontier_{name}"
            raw = r.lindex(hist_key, -1)
            if raw:
                try:
                    metrics[name] = json.loads(raw)["v"]
                except (KeyError, json.JSONDecodeError):
                    metrics[name] = None
            else:
                metrics[name] = None

        # Invalidity rate
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

        # Validator stage duration
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
            "metrics": metrics,
            # Keep best_val_fitness for backward compat
            "best_val_fitness": metrics.get("fitness"),
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
            "metrics": {},
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
    """Parse 'prefix@db[:label]' -> (prefix, db, label)."""
    label = None
    if ":" in arg:
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
    """Parse 'label:pid' -> (label, pid)."""
    if ":" not in arg:
        raise ValueError(f"--pid must be label:pid, got {arg!r}")
    label, pid_str = arg.split(":", 1)
    return label, int(pid_str)


def runs_from_experiment(
    exp_name: str,
) -> tuple[list[dict], dict[str, int], int | None]:
    """Load runs, pid_map, watchdog_pid from experiment.yaml."""
    from tools.experiment.manifest import load_manifest

    m = load_manifest(exp_name)
    runs = []
    pid_map = {}
    for r in m.runs:
        runs.append(
            {
                "prefix": r.prefix,
                "db": r.db,
                "label": r.label,
                "problem_name": r.problem_name,
            }
        )
        if r.pid:
            pid_map[r.label] = r.pid
    watchdog_pid = m.launch.watchdog_pid if m.launch else None
    return runs, pid_map, watchdog_pid


def main():
    parser = argparse.ArgumentParser(
        description="Show live status of GigaEvo evolution runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              %(prog)s --experiment hover/prompt_coevolution
              %(prog)s --run chains/hotpotqa/static@0:K
              %(prog)s \\
                --run chains/hotpotqa/static@0:K \\
                --run chains/hotpotqa/static_r@1:L \\
                --pid K:2616605 --pid L:2616606 \\
                --watchdog 2716169
        """),
    )
    parser.add_argument(
        "--experiment",
        metavar="TASK/NAME",
        help="Load runs, PIDs, and metrics from experiment.yaml",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
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

    # Build runs list from --experiment or --run flags
    if args.experiment:
        runs, pid_map, watchdog_pid = runs_from_experiment(args.experiment)
        if args.watchdog is None:
            args.watchdog = watchdog_pid
        # Merge any extra --pid flags
        for p in args.pid:
            label, pid = parse_pid_arg(p)
            pid_map[label] = pid
    else:
        if not args.run:
            parser.error("either --experiment or --run is required")
        runs = []
        for r in args.run:
            prefix, db, label = parse_run_arg(r)
            runs.append(
                {"prefix": prefix, "db": db, "label": label, "problem_name": None}
            )
        pid_map = {}
        for p in args.pid:
            label, pid = parse_pid_arg(p)
            pid_map[label] = pid

    # Discover metrics per problem_name (cache to avoid re-reading)
    metrics_cache: dict[str | None, list[str]] = {}
    for run in runs:
        pname = run.get("problem_name")
        if pname not in metrics_cache:
            if pname:
                specs = load_metrics_yaml(pname)
                metrics_cache[pname] = get_metric_names(specs) or ["fitness"]
            else:
                metrics_cache[pname] = ["fitness"]

    # Collect all unique metric names across runs (preserving order)
    all_metric_names: list[str] = []
    seen = set()
    for run in runs:
        for m in metrics_cache.get(run.get("problem_name"), ["fitness"]):
            if m not in seen:
                all_metric_names.append(m)
                seen.add(m)

    # Query Redis
    run_statuses = []
    for run in runs:
        mnames = metrics_cache.get(run.get("problem_name"), ["fitness"])
        status = get_run_status(
            run["prefix"], run["db"], mnames, args.redis_host, args.redis_port
        )
        run_statuses.append(status)

    # Build table
    pid_col = bool(pid_map)

    # Build metric column headers with format info from specs
    # Collect specs across all problems for formatting
    all_specs: dict[str, dict] = {}
    for run in runs:
        pname = run.get("problem_name")
        if pname:
            specs = load_metrics_yaml(pname)
            for name, spec in specs.items():
                if name not in all_specs:
                    all_specs[name] = spec

    metric_cols = []
    for name in all_metric_names:
        col_name = name.replace("_", " ").title()
        spec = all_specs.get(name, {})
        # Show as percentage if upper_bound is 1.0 (fractional metric)
        is_pct = spec.get("upper_bound", 1.0) == 1.0
        decimals = spec.get("decimals", 1)
        metric_cols.append((name, col_name, max(len(col_name), 10), is_pct, decimals))

    header = f"{'Run':<8} {'DB':>3}  {'Gen':>5}"
    for _, col_name, width, _, _ in metric_cols:
        header += f"  {col_name:>{width}}"
    header += f"  {'Invalid%':>8}  {'Val dur(s)':>12}  {'Keys':>6}"
    if pid_col:
        header += f"  {'PID':>10}  {'Status':<10}"
    print(header)
    print("-" * len(header))

    warnings = []

    for run, status in zip(runs, run_statuses):
        gen_str = str(status["gen"]) if status["gen"] is not None else "?"

        row = f"{run['label']:<8} {run['db']:>3}  {gen_str:>5}"

        for name, _, width, is_pct, decimals in metric_cols:
            val = status["metrics"].get(name)
            if val is not None:
                if is_pct:
                    row += f"  {val * 100:>{width}.{decimals}f}%"
                else:
                    row += f"  {val:>{width}.{decimals}f}"
            else:
                row += f"  {'?':>{width}}"

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

        row += f"  {inv_str:>8}  {dur_str:>12}  {keys_str:>6}"

        if pid_col:
            pid = pid_map.get(run["label"])
            if pid is not None:
                alive = pid_alive(pid)
                status_str = "ALIVE" if alive else "DEAD"
                row += f"  {pid:>10}  {status_str:<10}"
            else:
                row += f"  {'--':>10}  {'--':<10}"

        if status["error"]:
            row += f"  [ERROR: {status['error']}]"

        gen = status["gen"] or 0
        inv = status["invalid_rate"]
        if inv is not None and inv > 0.75 and gen >= 3:
            warnings.append(
                f"  !!  Run {run['label']}: {inv * 100:.0f}% invalid programs at gen {gen}"
                f" -- stage_timeout is likely too short for this eval workload"
            )

        print(row)

    if warnings:
        print()
        for w in warnings:
            print(w)

    # Watchdog
    if args.watchdog is not None:
        alive = pid_alive(args.watchdog)
        status_str = "ALIVE" if alive else "DEAD"
        print(f"\nWatchdog PID {args.watchdog}: {status_str}")


if __name__ == "__main__":
    main()
