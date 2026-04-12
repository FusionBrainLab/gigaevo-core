"""Tests for lifecycle composite CLI commands (launch, closeout, restart)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from gigaevo.cli import main


class TestLaunchRequiresExperiment:
    def test_no_experiment_shows_error(self):
        """Launch without --experiment shows usage error."""
        runner = CliRunner()
        result = runner.invoke(main, ["launch"], catch_exceptions=False)
        assert result.exit_code != 0
        assert "experiment" in result.output.lower()


class TestLaunchRequiresConfirm:
    def test_no_confirm_is_dry_run(self):
        """Launch without --confirm shows what would happen."""
        manifest = MagicMock()
        manifest.name = "test/exp"
        manifest.status = "implemented"
        manifest.runs = []
        manifest.servers = []

        with patch("tools.experiment.manifest.load_manifest", return_value=manifest):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "test/exp", "launch"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            assert (
                "dry-run" in result.output.lower() or "confirm" in result.output.lower()
            )


class TestCloseoutRequiresExperiment:
    def test_no_experiment_shows_error(self):
        """Closeout without --experiment shows usage error."""
        runner = CliRunner()
        result = runner.invoke(main, ["closeout"], catch_exceptions=False)
        assert result.exit_code != 0
        assert "experiment" in result.output.lower()


class TestRestartRequiresExperiment:
    def test_no_experiment_shows_error(self):
        """Restart without --experiment shows usage error."""
        runner = CliRunner()
        result = runner.invoke(main, ["restart"], catch_exceptions=False)
        assert result.exit_code != 0
        assert "experiment" in result.output.lower()


class TestRestartRequiresConfirm:
    def test_no_confirm_is_dry_run(self):
        """Restart without --confirm shows what would happen."""
        manifest = MagicMock()
        manifest.name = "test/exp"
        manifest.status = "running"
        manifest.runs = [MagicMock(label="A", db=4, prefix="p", pid=12345)]

        with patch("tools.experiment.manifest.load_manifest", return_value=manifest):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "test/exp", "restart"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            assert (
                "dry-run" in result.output.lower() or "confirm" in result.output.lower()
            )
