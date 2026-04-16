"""Tests for gigaevo.cli.launch_cmd — CLI wrapper for launch orchestrator."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner
import pytest

from gigaevo.cli import main
from gigaevo.experiment.launch import LaunchResult, LaunchStep


@pytest.fixture
def runner():
    return CliRunner()


class TestLaunchCmd:
    def test_requires_experiment_flag(self, runner):
        result = runner.invoke(main, ["launch"])
        assert result.exit_code != 0
        assert (
            "experiment" in result.output.lower() or "requires" in result.output.lower()
        )

    def test_dry_run_shows_plan(self, runner):
        launch_result = LaunchResult(
            experiment="hover/test",
            status="implemented",
            last_completed_step=LaunchStep.DBS_CLAIMED,
        )
        with patch("gigaevo.cli.launch_cmd.run_launch", return_value=launch_result):
            result = runner.invoke(main, ["-e", "hover/test", "launch", "--dry-run"])
        assert result.exit_code == 0
        assert "dry run" in result.output.lower() or "DBS_CLAIMED" in result.output

    def test_successful_launch(self, runner):
        launch_result = LaunchResult(
            experiment="hover/test",
            status="running",
            run_pids={"A": 1234, "B": 5678},
            watchdog_pid=9999,
            last_completed_step=LaunchStep.WATCHDOG_SPAWNED,
        )
        with patch("gigaevo.cli.launch_cmd.run_launch", return_value=launch_result):
            result = runner.invoke(main, ["-e", "hover/test", "launch"])
        assert result.exit_code == 0
        assert "running" in result.output.lower() or "1234" in result.output

    def test_failed_launch_nonzero_exit(self, runner):
        launch_result = LaunchResult(
            experiment="hover/test",
            status="implemented",
            last_completed_step=LaunchStep.GATE_CHECK,
            error="Preflight failed: server unreachable",
        )
        with patch("gigaevo.cli.launch_cmd.run_launch", return_value=launch_result):
            result = runner.invoke(main, ["-e", "hover/test", "launch"])
        assert result.exit_code != 0
        assert "Preflight failed" in result.output

    def test_skip_preflight_flag(self, runner):
        launch_result = LaunchResult(
            experiment="hover/test",
            status="implemented",
            last_completed_step=LaunchStep.DBS_CLAIMED,
        )
        with patch(
            "gigaevo.cli.launch_cmd.run_launch", return_value=launch_result
        ) as mock_launch:
            runner.invoke(
                main,
                ["-e", "hover/test", "launch", "--dry-run", "--skip-preflight"],
            )
        mock_launch.assert_called_once_with(
            "hover/test", dry_run=True, skip_preflight=True
        )
