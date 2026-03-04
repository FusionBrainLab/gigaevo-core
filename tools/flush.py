#!/usr/bin/env python3
"""
Safely flush Redis DBs for a new experiment launch.

Kills stale exec_runner workers first (they repopulate Redis immediately after
flush if left alive), then flushes each DB, then verifies 0 keys remain.

Dry-run by default — shows what would happen. Pass --confirm to execute.

Example usage:
    # Dry-run: show what would be flushed and which workers would be killed
    PYTHONPATH=. python tools/flush.py --db 0 1 2 3

    # Actually flush
    PYTHONPATH=. python tools/flush.py --db 0 1 2 3 --confirm

    # Flush without killing exec_runner workers (not recommended)
    PYTHONPATH=. python tools/flush.py --db 14 15 --confirm --no-kill-workers
"""

import argparse
import subprocess
import sys
import textwrap

import redis as redis_lib


def find_exec_runner_pids() -> list[int]:
    """Find all running exec_runner worker PIDs."""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
        )
        pids = []
        for line in result.stdout.splitlines():
            if "exec_runner" in line and "grep" not in line:
                parts = line.split()
                if len(parts) > 1:
                    try:
                        pids.append(int(parts[1]))
                    except ValueError:
                        pass
        return pids
    except Exception as e:
        print(f"[warn] Could not scan for exec_runner workers: {e}")
        return []


def kill_workers(pids: list[int], dry_run: bool) -> None:
    """Kill exec_runner workers."""
    if not pids:
        print("[workers] No exec_runner workers found.")
        return

    print(f"[workers] Found {len(pids)} exec_runner worker(s): {pids}")
    if dry_run:
        print(f"[workers] DRY-RUN — would kill: {pids}")
        return

    killed, failed = [], []
    for pid in pids:
        try:
            subprocess.run(["kill", str(pid)], check=True, capture_output=True)
            killed.append(pid)
        except subprocess.CalledProcessError:
            failed.append(pid)

    if killed:
        print(f"[workers] Killed: {killed}")
    if failed:
        print(f"[workers] Failed to kill (already dead?): {failed}")


def flush_db(db: int, host: str, port: int, dry_run: bool) -> bool:
    """Flush a single Redis DB. Returns True if successful (or dry-run)."""
    r = redis_lib.Redis(host=host, port=port, db=db)
    try:
        before = r.dbsize()
        if dry_run:
            print(f"[flush] DRY-RUN — DB {db}: {before} keys would be flushed")
            return True
        r.flushdb()
        after = r.dbsize()
        if after == 0:
            print(f"[flush] DB {db}: {before} keys flushed → 0 keys ✓")
            return True
        else:
            print(
                f"[flush] DB {db}: flushed but {after} keys remain — workers still running?"
            )
            return False
    except Exception as e:
        print(f"[flush] DB {db}: ERROR — {e}")
        return False
    finally:
        r.close()


def main():
    parser = argparse.ArgumentParser(
        description="Kill exec_runner workers and flush Redis DBs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Preview (dry-run, default)
              %(prog)s --db 0 1 2 3

              # Execute flush
              %(prog)s --db 0 1 2 3 --confirm

              # Flush specific DBs for P3 experiment
              %(prog)s --db 14 15 --confirm
        """),
    )
    parser.add_argument(
        "--db",
        nargs="+",
        type=int,
        required=True,
        metavar="N",
        help="Redis DB numbers to flush (space-separated)",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually kill workers and flush. Without this flag, dry-run only.",
    )
    parser.add_argument(
        "--no-kill-workers",
        action="store_true",
        help="Skip killing exec_runner workers (not recommended — they repopulate Redis)",
    )
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)

    args = parser.parse_args()
    dry_run = not args.confirm

    if dry_run:
        print("[flush] DRY-RUN mode — pass --confirm to execute\n")

    # Step 1: Kill exec_runner workers
    if not args.no_kill_workers:
        pids = find_exec_runner_pids()
        kill_workers(pids, dry_run)
    else:
        print("[workers] Skipping exec_runner cleanup (--no-kill-workers)")
    print()

    # Step 2: Flush DBs
    all_ok = True
    for db in args.db:
        ok = flush_db(db, args.redis_host, args.redis_port, dry_run)
        if not ok:
            all_ok = False
    print()

    # Step 3: Summary
    if dry_run:
        print("[summary] Dry-run complete. Run with --confirm to execute.")
    elif all_ok:
        print("[summary] All DBs flushed successfully. Ready for launch.")
    else:
        print("[summary] Some DBs may not be clean — check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
