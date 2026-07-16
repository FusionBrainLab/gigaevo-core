"""Process kill and Redis flush operations.

Ported from tools/flush.py — kills stale exec_runner workers for specified DBs
and flushes Redis databases.
"""

from __future__ import annotations

import re
import subprocess
import time

import redis as redis_lib

_REDIS_DB_RE = re.compile(r"redis\.db=(\d+)")
_RUN_SCRIPT_RE = re.compile(r"(?:^|\s)(?:\S*/)?run\.py(?:\s|$)")
_EXEC_RUNNER_RE = re.compile(r"(?:^|\s)(?:\S*/)?exec_runner\.py(?:\s|$)")


def _is_run_py_line(line: str) -> bool:
    if "grep" in line:
        return False
    return _RUN_SCRIPT_RE.search(line) is not None and "redis.db=" in line


def _extract_db(line: str) -> int | None:
    m = _REDIS_DB_RE.search(line)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _find_run_pids_for_dbs(target_dbs: list[int]) -> set[int]:
    target_set = set(target_dbs)
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        pids: set[int] = set()
        for line in result.stdout.splitlines():
            if not _is_run_py_line(line):
                continue
            db = _extract_db(line)
            if db is None or db not in target_set:
                continue
            parts = line.split()
            if len(parts) > 1:
                try:
                    pids.add(int(parts[1]))
                except ValueError:
                    pass
        return pids
    except Exception:
        return set()


def _find_all_run_pids() -> set[int]:
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        pids: set[int] = set()
        for line in result.stdout.splitlines():
            if _is_run_py_line(line):
                parts = line.split()
                if len(parts) > 1:
                    try:
                        pids.add(int(parts[1]))
                    except ValueError:
                        pass
        return pids
    except Exception:
        return set()


def find_exec_runner_pids(
    target_dbs: list[int], *, include_orphans: bool = False
) -> list[int]:
    """Find workers owned by writers for ``target_dbs``.

    Orphan workers cannot be associated with a Redis DB after their parent has
    exited, so they are excluded unless the caller explicitly opts in.
    """
    try:
        run_pids_for_target = _find_run_pids_for_dbs(target_dbs)
        all_run_pids = _find_all_run_pids() if include_orphans else set()

        result = subprocess.run(
            ["ps", "-e", "-o", "pid,ppid,cmd", "--no-headers"],
            capture_output=True,
            text=True,
        )
        matched_pids = []
        orphan_pids = []
        for line in result.stdout.splitlines():
            if _EXEC_RUNNER_RE.search(line) is None or "grep" in line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
                ppid = int(parts[1])
            except ValueError:
                continue

            if ppid in run_pids_for_target:
                matched_pids.append(pid)
            elif include_orphans and ppid not in all_run_pids:
                orphan_pids.append(pid)

        return matched_pids + orphan_pids
    except Exception as e:
        print(f"[warn] Could not scan for exec_runner workers: {e}")
        return []


def kill_workers(pids: list[int], dry_run: bool) -> None:
    if not pids:
        print("[workers] No exec_runner workers found for target DBs.")
        return

    print(f"[workers] Found {len(pids)} exec_runner worker(s) for target DBs: {pids}")
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


def kill_run_writers(target_dbs: list[int], dry_run: bool) -> list[int]:
    """Signal matching writers and return the PIDs that were found."""
    pids = sorted(_find_run_pids_for_dbs(target_dbs))
    if not pids:
        print("[writers] No run.py writer processes found for target DBs.")
        return []

    print(f"[writers] Found {len(pids)} run.py writer(s) for target DBs: {pids}")
    if dry_run:
        print(f"[writers] DRY-RUN — would kill: {pids}")
        return pids

    killed, failed = [], []
    for pid in pids:
        try:
            subprocess.run(["kill", str(pid)], check=True, capture_output=True)
            killed.append(pid)
        except subprocess.CalledProcessError:
            failed.append(pid)

    if killed:
        print(f"[writers] Killed: {killed}")
    if failed:
        print(f"[writers] Failed to kill (already dead?): {failed}")
    return pids


def warn_if_not_archived(db: int, before: int) -> None:
    if before == 0:
        return
    print(f"[warn]  DB {db}: {before} keys present.")
    print(
        "[warn]  Have you run 'bash tools/experiment/archive_run.sh --upload' for this DB?"
    )
    print(
        "[warn]  Flushing without archiving destroys all evolved programs permanently."
    )
    print("[warn]  Proceeding in 5 seconds — Ctrl+C to abort.")
    time.sleep(5)


def flush_db(db: int, host: str, port: int, dry_run: bool) -> bool:
    r = redis_lib.Redis(host=host, port=port, db=db)
    try:
        before = r.dbsize()
        if dry_run:
            print(f"[flush] DRY-RUN — DB {db}: {before} keys would be flushed")
            return True
        warn_if_not_archived(db, before)
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
