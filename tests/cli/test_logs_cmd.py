"""Tests for the logs CLI subcommand."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from gigaevo.cli import main


class TestLogsExplicitFile:
    def test_reads_explicit_log_file(self, tmp_path: Path):
        """--file reads the specified log file."""
        log = tmp_path / "test.log"
        log.write_text("line1\nline2\nline3\nline4\nline5\n")

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["logs", "--file", str(log), "-n", "3"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        # tail -n 3 should show last 3 lines
        assert "line3" in result.output
        assert "line5" in result.output

    def test_missing_file_shows_error(self):
        """Missing log file exits with error."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["logs", "--file", "/nonexistent/path.log"],
            catch_exceptions=False,
        )
        assert result.exit_code != 0


class TestLogsDiscovery:
    def test_discovers_log_from_experiment(self, tmp_path: Path):
        """Auto-discovers nohup log in experiment directory."""
        exp_dir = tmp_path / "experiments" / "test_exp"
        exp_dir.mkdir(parents=True)
        log_file = exp_dir / "nohup_run.log"
        log_file.write_text("discovered log line\n")

        with patch("gigaevo.cli.logs._discover_log_file") as mock_discover:
            mock_discover.return_value = log_file

            runner = CliRunner()
            result = runner.invoke(
                main,
                ["-e", "test_exp", "logs", "-n", "10"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            assert "discovered log line" in result.output

    def test_no_experiment_no_file_shows_error(self):
        """No --file and no --experiment shows error."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["logs"],
            catch_exceptions=False,
        )
        assert result.exit_code != 0


class TestLogsFollow:
    def test_follow_flag_calls_tail_f(self, tmp_path: Path):
        """--follow passes -f to tail subprocess."""
        log = tmp_path / "test.log"
        log.write_text("follow test\n")

        with patch("gigaevo.cli.logs.subprocess") as mock_sub:
            mock_sub.run.return_value = None

            runner = CliRunner()
            result = runner.invoke(
                main,
                ["logs", "--file", str(log), "-f"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0
            call_args = mock_sub.run.call_args[0][0]
            assert "tail" in call_args
            assert "-f" in call_args
