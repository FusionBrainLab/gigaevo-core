"""Unified launch orchestrator for experiments.

Sequences: gate → preflight → claim DBs → generate script → exec → record
PIDs → set status → spawn watchdog.  Returns a typed ``LaunchResult``.

The CLI (``gigaevo -e <exp> launch``) is a thin wrapper around
``run_launch()``.  Skills call the same function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import os
from pathlib import Path
import subprocess
import time

from loguru import logger

from gigaevo.experiment.launch_generator import generate as _generate_script_content
from gigaevo.experiment.manifest import (
    claim_dbs,
    experiment_dir,
    load_manifest,
    release_db_claims,
    set_status,
    update_manifest,
)

_log = logger.bind(component="launch")


class LaunchStep(IntEnum):
    NONE = 0
    GATE_CHECK = 1
    PREFLIGHT_PASSED = 2
    DBS_CLAIMED = 3
    SCRIPT_GENERATED = 4
    RUNS_LAUNCHED = 5
    STATUS_SET = 6
    WATCHDOG_SPAWNED = 7


@dataclass(frozen=True)
class LaunchResult:
    experiment: str
    status: str
    run_pids: dict[str, int] = field(default_factory=dict)
    watchdog_pid: int | None = None
    last_completed_step: LaunchStep = LaunchStep.NONE
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def run_launch(
    experiment: str,
    *,
    dry_run: bool = False,
    skip_preflight: bool = False,
) -> LaunchResult:
    """Run the full launch sequence for an experiment.

    Args:
        experiment: Experiment name (e.g. "hover/test-launch").
        dry_run: If True, stop after DB claims (no exec, no status change).
        skip_preflight: If True, skip preflight checks (for re-launches).

    Returns:
        LaunchResult with outcome. Check ``.ok`` for success.
    """
    manifest = load_manifest(experiment)

    # Step 1: Gate check
    if manifest.lifecycle.status != "implemented":
        return LaunchResult(
            experiment=experiment,
            status=manifest.lifecycle.status,
            error=(
                f"Expected status 'implemented', got '{manifest.lifecycle.status}'. "
                f"Use 'gigaevo -e {experiment} manifest reset-status implemented' to recover."
            ),
        )
    _log.info("Gate check passed: status=implemented")

    # Step 2: Preflight
    if not skip_preflight:
        failures = _run_preflight(experiment)
        if failures:
            return LaunchResult(
                experiment=experiment,
                status="implemented",
                last_completed_step=LaunchStep.GATE_CHECK,
                error=f"Preflight failed with {len(failures)} issue(s): {failures[0]}",
            )
    _log.info("Preflight passed")

    # Step 2b: LAUNCH_PREVIEW.md (best-effort — failure here does not block launch)
    _write_launch_preview(experiment)

    # Step 3: Claim DBs
    dbs = [r.db for r in manifest.contract.runs]
    claim_failures = _claim_dbs(experiment, dbs)
    if claim_failures:
        owners = ", ".join(f"DB {db} owned by {o}" for db, o in claim_failures)
        return LaunchResult(
            experiment=experiment,
            status="implemented",
            last_completed_step=LaunchStep.PREFLIGHT_PASSED,
            error=f"DB claim failed: {owners}",
        )
    _log.info("DBs claimed: {}", dbs)

    if dry_run:
        return LaunchResult(
            experiment=experiment,
            status="implemented",
            last_completed_step=LaunchStep.DBS_CLAIMED,
        )

    # Step 4+: Generate, exec, record, set status, spawn watchdog
    try:
        script_path = _generate_launch_script(experiment)
        _log.info("Launch script: {}", script_path)

        run_pids = _exec_launch_script(script_path, manifest)
        _log.info("Runs launched: {}", run_pids)

        _record_pids_and_set_running(experiment, run_pids)
        _log.info("Status set to running, PIDs recorded")

        watchdog_pid = _spawn_watchdog(experiment)
        _log.info("Watchdog spawned: PID {}", watchdog_pid)

        return LaunchResult(
            experiment=experiment,
            status="running",
            run_pids=run_pids,
            watchdog_pid=watchdog_pid,
            last_completed_step=LaunchStep.WATCHDOG_SPAWNED,
        )
    except Exception as exc:
        _log.error("Launch failed: {}", exc)
        _release_claims(dbs)
        return LaunchResult(
            experiment=experiment,
            status="implemented",
            last_completed_step=LaunchStep.DBS_CLAIMED,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Internal helpers (mockable seams)
# ---------------------------------------------------------------------------


def _run_preflight(experiment: str) -> list[str]:
    """Run preflight checks. Returns list of failure messages (empty = pass)."""
    from gigaevo.experiment.checks import run_checks

    results = run_checks(experiment)
    return [str(r) for r in results if r.is_blocking]


def _write_launch_preview(experiment: str) -> None:
    """Render LAUNCH_PREVIEW.md for reviewer Gate 2. Best-effort."""
    try:
        from gigaevo.experiment.dry_run import dry_run
        from gigaevo.experiment.launch_preview import write_launch_preview

        result = dry_run(experiment)
        out = write_launch_preview(experiment, result)
        _log.info("Wrote launch preview: {}", out)
    except Exception as exc:
        _log.warning("Launch preview skipped: {}", exc)


def _claim_dbs(experiment: str, dbs: list[int]) -> list[tuple[int, str]]:
    return claim_dbs(experiment, dbs)


def _generate_launch_script(experiment: str) -> Path:
    content = _generate_script_content(experiment)
    out_path = experiment_dir(experiment) / "launch.sh"
    out_path.write_text(content)
    out_path.chmod(0o755)
    return out_path


def _exec_launch_script(
    script_path: Path,
    manifest,
) -> dict[str, int]:
    """Execute launch.sh and extract PIDs from pids.txt."""
    exp_dir = script_path.parent
    result = subprocess.run(
        ["bash", str(script_path)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"launch.sh failed (exit {result.returncode}):\n{result.stderr[-500:]}"
        )

    pids_file = exp_dir / "pids.txt"
    if not pids_file.exists():
        raise RuntimeError("launch.sh did not create pids.txt")

    pid_values = pids_file.read_text().strip().split()
    labels = [r.label for r in manifest.contract.runs]
    if len(pid_values) != len(labels):
        raise RuntimeError(
            f"PID count mismatch: {len(pid_values)} PIDs for {len(labels)} runs"
        )

    run_pids: dict[str, int] = {}
    for label, pid_str in zip(labels, pid_values):
        pid = int(pid_str)
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            raise RuntimeError(f"Run {label} (PID {pid}) not alive after launch")
        run_pids[label] = pid

    return run_pids


def _record_pids_and_set_running(
    experiment: str,
    run_pids: dict[str, int],
) -> None:
    """Record PIDs in manifest and transition to running."""
    import datetime

    def _updater(raw: dict) -> None:
        for run in raw.get("contract", {}).get("runs", []):
            label = run.get("label")
            if label in run_pids:
                run["pid"] = run_pids[label]
        lc = raw.setdefault("lifecycle", {})
        launch = lc.setdefault("launch", {})
        launch["time"] = datetime.datetime.now(datetime.UTC).isoformat()
        launch["commit"] = _git_head()
        launch["confirmed_at"] = launch["time"]

    update_manifest(experiment, _updater)
    set_status(experiment, "running")


def _spawn_watchdog(experiment: str) -> int:
    """Spawn watchdog as a background process. Returns PID."""
    exp_dir = experiment_dir(experiment)
    log_path = exp_dir / "watchdog.log"
    with open(log_path, "a") as log_f:
        proc = subprocess.Popen(
            ["gigaevo", "-e", experiment, "watchdog"],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    time.sleep(2)
    if proc.poll() is not None:
        raise RuntimeError(
            f"Watchdog died immediately (exit {proc.returncode}). Check {log_path}"
        )

    def _record_wd_pid(raw: dict) -> None:
        raw.setdefault("control_plane", {})["watchdog_pid"] = proc.pid

    update_manifest(experiment, _record_wd_pid)
    return proc.pid


def _release_claims(dbs: list[int]) -> None:
    release_db_claims(dbs)


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"
