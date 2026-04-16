"""Tests for gigaevo.experiment.launch — unified launch orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from gigaevo.experiment.launch import LaunchResult, LaunchStep, run_launch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run(label: str, db: int) -> MagicMock:
    run = MagicMock()
    run.label = label
    run.db = db
    run.pipeline = "standard"
    run.mutation_url = "http://10.0.0.1:4000"
    run.model_name = "gpt-4o"
    return run


def _make_manifest(
    *,
    status: str = "implemented",
    runs: list | None = None,
    experiment_name: str = "hover/test-launch",
) -> MagicMock:
    m = MagicMock()
    m.lifecycle.status = status
    m.contract.runs = runs or [_make_run("A", 5), _make_run("B", 6)]
    m.contract.identity.name = experiment_name
    m.contract.servers = ["10.0.0.1"]
    m.contract.max_generations = 25
    return m


# ---------------------------------------------------------------------------
# LaunchResult
# ---------------------------------------------------------------------------


class TestLaunchResult:
    def test_successful_result_is_ok(self):
        result = LaunchResult(
            experiment="hover/test",
            status="running",
            run_pids={"A": 1234, "B": 5678},
            watchdog_pid=9999,
            last_completed_step=LaunchStep.WATCHDOG_SPAWNED,
        )
        assert result.ok
        assert result.error is None

    def test_failed_result_is_not_ok(self):
        result = LaunchResult(
            experiment="hover/test",
            status="implemented",
            run_pids={},
            last_completed_step=LaunchStep.GATE_CHECK,
            error="Preflight failed: server unreachable",
        )
        assert not result.ok
        assert "Preflight" in result.error


# ---------------------------------------------------------------------------
# Gate check
# ---------------------------------------------------------------------------


class TestGateCheck:
    def test_rejects_non_implemented_status(self):
        m = _make_manifest(status="preregistered")
        with patch("gigaevo.experiment.launch.load_manifest", return_value=m):
            result = run_launch("hover/test", dry_run=True)
        assert not result.ok
        assert result.last_completed_step == LaunchStep.NONE
        assert "implemented" in result.error.lower()

    def test_accepts_implemented_status(self):
        m = _make_manifest(status="implemented")
        with (
            patch("gigaevo.experiment.launch.load_manifest", return_value=m),
            patch("gigaevo.experiment.launch._run_preflight") as mock_pf,
            patch("gigaevo.experiment.launch._claim_dbs") as mock_claim,
        ):
            mock_pf.return_value = []
            mock_claim.return_value = []
            result = run_launch("hover/test", dry_run=True)
        assert result.last_completed_step.value >= LaunchStep.GATE_CHECK.value


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


class TestPreflight:
    def test_preflight_failure_stops_launch(self):
        m = _make_manifest()
        with (
            patch("gigaevo.experiment.launch.load_manifest", return_value=m),
            patch(
                "gigaevo.experiment.launch._run_preflight",
                return_value=["CRITICAL: server unreachable"],
            ),
        ):
            result = run_launch("hover/test", dry_run=True)
        assert not result.ok
        assert result.last_completed_step == LaunchStep.GATE_CHECK
        assert "preflight" in result.error.lower()

    def test_preflight_pass_continues(self):
        m = _make_manifest()
        with (
            patch("gigaevo.experiment.launch.load_manifest", return_value=m),
            patch("gigaevo.experiment.launch._run_preflight", return_value=[]),
            patch("gigaevo.experiment.launch._claim_dbs", return_value=[]),
        ):
            result = run_launch("hover/test", dry_run=True)
        assert result.last_completed_step.value >= LaunchStep.PREFLIGHT_PASSED.value


# ---------------------------------------------------------------------------
# DB Claims
# ---------------------------------------------------------------------------


class TestDbClaims:
    def test_claim_failure_stops_launch(self):
        m = _make_manifest()
        with (
            patch("gigaevo.experiment.launch.load_manifest", return_value=m),
            patch("gigaevo.experiment.launch._run_preflight", return_value=[]),
            patch(
                "gigaevo.experiment.launch._claim_dbs",
                return_value=[(5, "hover/other-experiment")],
            ),
        ):
            result = run_launch("hover/test", dry_run=True)
        assert not result.ok
        assert result.last_completed_step == LaunchStep.PREFLIGHT_PASSED
        assert "claim" in result.error.lower() or "DB" in result.error

    def test_claim_success_continues(self):
        m = _make_manifest()
        with (
            patch("gigaevo.experiment.launch.load_manifest", return_value=m),
            patch("gigaevo.experiment.launch._run_preflight", return_value=[]),
            patch("gigaevo.experiment.launch._claim_dbs", return_value=[]),
        ):
            result = run_launch("hover/test", dry_run=True)
        assert result.last_completed_step.value >= LaunchStep.DBS_CLAIMED.value


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_stops_after_claims(self):
        m = _make_manifest()
        with (
            patch("gigaevo.experiment.launch.load_manifest", return_value=m),
            patch("gigaevo.experiment.launch._run_preflight", return_value=[]),
            patch("gigaevo.experiment.launch._claim_dbs", return_value=[]),
        ):
            result = run_launch("hover/test", dry_run=True)
        assert result.ok
        assert result.last_completed_step == LaunchStep.DBS_CLAIMED
        assert result.run_pids == {}
        assert result.watchdog_pid is None


# ---------------------------------------------------------------------------
# Full launch (non-dry-run)
# ---------------------------------------------------------------------------


class TestFullLaunch:
    def test_full_launch_happy_path(self):
        m = _make_manifest()
        with (
            patch("gigaevo.experiment.launch.load_manifest", return_value=m),
            patch("gigaevo.experiment.launch._run_preflight", return_value=[]),
            patch("gigaevo.experiment.launch._claim_dbs", return_value=[]),
            patch("gigaevo.experiment.launch._generate_launch_script") as mock_gen,
            patch("gigaevo.experiment.launch._exec_launch_script") as mock_exec,
            patch("gigaevo.experiment.launch._record_pids_and_set_running") as mock_rec,
            patch("gigaevo.experiment.launch._spawn_watchdog") as mock_wd,
        ):
            mock_gen.return_value = Path("/tmp/launch.sh")
            mock_exec.return_value = {"A": 1234, "B": 5678}
            mock_wd.return_value = 9999
            result = run_launch("hover/test")

        assert result.ok
        assert result.status == "running"
        assert result.run_pids == {"A": 1234, "B": 5678}
        assert result.watchdog_pid == 9999
        assert result.last_completed_step == LaunchStep.WATCHDOG_SPAWNED
        mock_rec.assert_called_once()

    def test_exec_failure_rolls_back_claims(self):
        m = _make_manifest()
        with (
            patch("gigaevo.experiment.launch.load_manifest", return_value=m),
            patch("gigaevo.experiment.launch._run_preflight", return_value=[]),
            patch("gigaevo.experiment.launch._claim_dbs", return_value=[]),
            patch("gigaevo.experiment.launch._generate_launch_script") as mock_gen,
            patch(
                "gigaevo.experiment.launch._exec_launch_script",
                side_effect=RuntimeError("launch.sh failed"),
            ),
            patch("gigaevo.experiment.launch._release_claims") as mock_release,
        ):
            mock_gen.return_value = Path("/tmp/launch.sh")
            result = run_launch("hover/test")

        assert not result.ok
        assert "launch.sh failed" in result.error
        mock_release.assert_called_once()
