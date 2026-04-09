"""
Resource Manager — Auto-detects available GPU servers and free Redis DBs.

Used by experiment-implement Step 7 to auto-assign runs to servers and DBs.

Usage:
    # CLI: check available resources
    PYTHONPATH=. python tools/resource_manager.py --check

    # CLI: assign resources for an experiment
    PYTHONPATH=. python tools/resource_manager.py --assign --experiment hover/my-exp --n-runs 4

    # Python API
    from tools.resource_manager import find_free_resources
    resources = find_free_resources(n_runs=4)
    # → [{"server": "INTERNAL_IP", "db": 5, "label": "R1"}, ...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJ = Path(__file__).parent.parent
INFRA_PATH = PROJ / "experiments" / "infrastructure.yaml"
# DBs reserved for system use or smoke tests: skip 0 (default), 15 (benchmarks)
RESERVED_DBS = {0, 15}
ALL_DBS = set(range(16)) - RESERVED_DBS


def load_infrastructure() -> dict[str, Any]:
    return yaml.safe_load(INFRA_PATH.read_text())


def check_redis_db_free(host: str, port: int, db: int) -> bool:
    """Return True if a Redis DB has 0 keys (i.e., not in use)."""
    try:
        result = subprocess.run(
            ["redis-cli", "-h", host, "-p", str(port), "-n", str(db), "dbsize"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and result.stdout.strip() == "0"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_gpu_load(server_ip: str, ssh_user: str = "jovyan") -> float | None:
    """
    Return average GPU utilization (0-100) for a server via SSH nvidia-smi.
    Returns None if the server is unreachable or has no GPUs.
    """
    try:
        result = subprocess.run(
            [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=5",
                f"{ssh_user}@{server_ip}",
                "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        if not lines:
            return None
        loads = [float(l) for l in lines if l.isdigit()]
        return sum(loads) / len(loads) if loads else None
    except (subprocess.TimeoutExpired, ValueError, Exception):
        return None


def find_free_resources(
    n_runs: int,
    task: str | None = None,
    redis_host: str = "localhost",
    redis_port: int = 6379,
) -> list[dict[str, Any]]:
    """
    Find n_runs free (server, db) pairs.

    Returns a list of dicts:
    [{"server": "INTERNAL_IP", "db": 5, "label": "R1", "gpu_load": 23.4}, ...]

    Sorted by GPU load ascending (least loaded first).
    """
    infra = load_infrastructure()
    servers = infra.get("servers", [])

    # Filter by task if specified
    if task:
        task_servers = [s for s in servers if task in s.get("tasks", []) or "any" in s.get("tasks", [])]
        if task_servers:
            servers = task_servers

    # Get free DBs
    free_dbs = sorted(db for db in ALL_DBS if check_redis_db_free(redis_host, redis_port, db))

    if len(free_dbs) < n_runs:
        return []  # not enough free DBs

    # Get server loads
    server_loads = []
    for server in servers:
        ip = server.get("ip") or server.get("host", "")
        if not ip:
            continue
        load = check_gpu_load(ip)
        server_loads.append({"server": ip, "gpu_load": load or 100.0, "info": server})

    server_loads.sort(key=lambda x: x["gpu_load"])

    # Assign: round-robin across least-loaded servers
    assignments = []
    labels = [f"R{i+1}" for i in range(n_runs)]
    for i in range(n_runs):
        if i >= len(server_loads):
            break
        server = server_loads[i % len(server_loads)]
        assignments.append({
            "server": server["server"],
            "db": free_dbs[i],
            "label": labels[i],
            "gpu_load": server["gpu_load"],
        })

    return assignments[:n_runs]


def check_and_print() -> None:
    """CLI: print resource availability summary."""
    infra = load_infrastructure()
    servers = infra.get("servers", [])

    print("=== GigaEvo Resource Status ===\n")
    print("Servers:")
    for server in servers:
        ip = server.get("ip") or server.get("host", "")
        load = check_gpu_load(ip)
        status = f"{load:.0f}% GPU" if load is not None else "UNREACHABLE"
        name = server.get("name", ip)
        print(f"  {name:20s} {ip:18s} {status}")

    print("\nRedis DBs (localhost:6379):")
    free_dbs = []
    used_dbs = []
    for db in sorted(ALL_DBS):
        if check_redis_db_free("localhost", 6379, db):
            free_dbs.append(db)
        else:
            used_dbs.append(db)

    print(f"  Free:  {free_dbs}")
    print(f"  In use: {used_dbs}")
    print(f"\n  Available slots: {len(free_dbs)} runs can be started")


def assign_and_print(exp_name: str, n_runs: int) -> None:
    """CLI: assign resources and print as YAML for experiment.yaml integration."""
    task = exp_name.split("/")[0] if "/" in exp_name else None
    resources = find_free_resources(n_runs=n_runs, task=task)

    if not resources:
        print(f"ERROR: insufficient free resources for {n_runs} runs", file=sys.stderr)
        sys.exit(1)

    print(f"# Suggested resource assignment for {exp_name} ({n_runs} runs)")
    print("# Add to experiment.yaml servers and runs[].db fields\n")
    print("servers:")
    seen_servers = []
    for r in resources:
        if r["server"] not in seen_servers:
            seen_servers.append(r["server"])
            print(f"  - host: {r['server']}")
    print()
    print("run_db_assignments:")
    for r in resources:
        print(f"  {r['label']}: db={r['db']}  # {r['server']} ({r['gpu_load']:.0f}% GPU)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GigaEvo resource manager")
    parser.add_argument("--check", action="store_true", help="Show resource availability")
    parser.add_argument("--assign", action="store_true", help="Assign resources for an experiment")
    parser.add_argument("--experiment", type=str, help="Experiment name (task/name)")
    parser.add_argument("--n-runs", type=int, default=4, help="Number of runs to assign")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.check:
        check_and_print()
    elif args.assign:
        if not args.experiment:
            print("ERROR: --assign requires --experiment", file=sys.stderr)
            sys.exit(1)
        if args.json:
            task = args.experiment.split("/")[0] if "/" in args.experiment else None
            resources = find_free_resources(n_runs=args.n_runs, task=task)
            print(json.dumps(resources, indent=2))
        else:
            assign_and_print(args.experiment, args.n_runs)
    else:
        parser.print_help()
