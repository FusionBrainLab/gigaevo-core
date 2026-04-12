"""Tests for the flush CLI subcommand."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from gigaevo.cli import main


class TestFlushDryRunDefault:
    def test_no_confirm_is_dry_run(self):
        """Without --confirm, flush_db is called with dry_run=True."""
        with (
            patch("gigaevo.cli.flush.flush_db") as mock_flush,
            patch("gigaevo.cli.flush.find_exec_runner_pids", return_value=[]),
            patch("gigaevo.cli.flush.kill_workers"),
            patch("gigaevo.cli.flush.kill_run_writers"),
        ):
            mock_flush.return_value = True
            runner = CliRunner()
            result = runner.invoke(main, ["flush", "--db", "5"], catch_exceptions=False)
            assert result.exit_code == 0, result.output
            mock_flush.assert_called_once_with(5, "localhost", 6379, True)

    def test_explicit_dry_run_flag(self):
        """--dry-run flag forces dry_run=True even with --confirm."""
        with (
            patch("gigaevo.cli.flush.flush_db") as mock_flush,
            patch("gigaevo.cli.flush.find_exec_runner_pids", return_value=[]),
            patch("gigaevo.cli.flush.kill_workers"),
            patch("gigaevo.cli.flush.kill_run_writers"),
        ):
            mock_flush.return_value = True
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["flush", "--db", "5", "--confirm", "--dry-run"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            mock_flush.assert_called_once_with(5, "localhost", 6379, True)


class TestFlushConfirm:
    def test_confirm_executes_flush(self):
        """--confirm causes flush_db to run with dry_run=False."""
        with (
            patch("gigaevo.cli.flush.flush_db") as mock_flush,
            patch("gigaevo.cli.flush.find_exec_runner_pids", return_value=[]),
            patch("gigaevo.cli.flush.kill_workers"),
            patch("gigaevo.cli.flush.kill_run_writers"),
        ):
            mock_flush.return_value = True
            runner = CliRunner()
            result = runner.invoke(
                main, ["flush", "--db", "5", "--confirm"], catch_exceptions=False
            )
            assert result.exit_code == 0, result.output
            mock_flush.assert_called_once_with(5, "localhost", 6379, False)

    def test_multiple_dbs_flushed(self):
        """Multiple --db values each get flushed."""
        with (
            patch("gigaevo.cli.flush.flush_db") as mock_flush,
            patch("gigaevo.cli.flush.find_exec_runner_pids", return_value=[]),
            patch("gigaevo.cli.flush.kill_workers"),
            patch("gigaevo.cli.flush.kill_run_writers"),
        ):
            mock_flush.return_value = True
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["flush", "--db", "5", "--db", "6", "--confirm"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            assert mock_flush.call_count == 2


class TestFlushDbValidation:
    def test_db_out_of_range_errors(self):
        """DB number > 15 shows error."""
        runner = CliRunner()
        result = runner.invoke(main, ["flush", "--db", "16"], catch_exceptions=False)
        assert result.exit_code != 0
        assert "out of range" in result.output.lower() or "16" in result.output

    def test_negative_db_errors(self):
        """Negative DB number shows error."""
        runner = CliRunner()
        result = runner.invoke(main, ["flush", "--db", "-1"], catch_exceptions=False)
        assert result.exit_code != 0


class TestFlushNoKillWorkers:
    def test_no_kill_workers_skips_killing(self):
        """--no-kill-workers skips worker and writer killing."""
        with (
            patch("gigaevo.cli.flush.flush_db") as mock_flush,
            patch("gigaevo.cli.flush.find_exec_runner_pids") as mock_find,
            patch("gigaevo.cli.flush.kill_workers") as mock_kill,
            patch("gigaevo.cli.flush.kill_run_writers") as mock_kill_writers,
        ):
            mock_flush.return_value = True
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["flush", "--db", "5", "--no-kill-workers"],
                catch_exceptions=False,
            )
            assert result.exit_code == 0, result.output
            mock_find.assert_not_called()
            mock_kill.assert_not_called()
            mock_kill_writers.assert_not_called()


class TestFlushFailure:
    def test_flush_failure_exits_nonzero(self):
        """If flush_db returns False, exit code is 1."""
        with (
            patch("gigaevo.cli.flush.flush_db") as mock_flush,
            patch("gigaevo.cli.flush.find_exec_runner_pids", return_value=[]),
            patch("gigaevo.cli.flush.kill_workers"),
            patch("gigaevo.cli.flush.kill_run_writers"),
        ):
            mock_flush.return_value = False
            runner = CliRunner()
            result = runner.invoke(
                main, ["flush", "--db", "5", "--confirm"], catch_exceptions=False
            )
            assert result.exit_code == 1
