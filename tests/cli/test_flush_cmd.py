"""Tests for the flush CLI subcommand."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

from click.testing import CliRunner

from gigaevo.cli import main


class TestFlushDryRunDefault:
    def test_no_confirm_is_dry_run(self):
        """Without --confirm, flush_db is called with dry_run=True."""
        with (
            patch("gigaevo.cli.flush.flush_db") as mock_flush,
            patch("gigaevo.cli.flush.find_exec_runner_pids", return_value=[]),
            patch("gigaevo.cli.flush.kill_workers"),
            patch("gigaevo.cli.flush.kill_run_writers", return_value=[]),
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
            patch("gigaevo.cli.flush.kill_run_writers", return_value=[]),
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
            patch("gigaevo.cli.flush.kill_run_writers", return_value=[]),
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
            patch("gigaevo.cli.flush.kill_run_writers", return_value=[]),
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

    def test_worker_pids_are_captured_before_writers_are_killed(self):
        order = MagicMock()
        with (
            patch("gigaevo.cli.flush.flush_db", return_value=True),
            patch(
                "gigaevo.cli.flush.find_exec_runner_pids", return_value=[123]
            ) as find,
            patch("gigaevo.cli.flush.kill_run_writers", return_value=[456]) as writers,
            patch("gigaevo.cli.flush.kill_workers") as workers,
            patch("gigaevo.cli.flush.time.sleep"),
        ):
            order.attach_mock(find, "find")
            order.attach_mock(writers, "writers")
            order.attach_mock(workers, "workers")
            result = CliRunner().invoke(
                main, ["flush", "--db", "5", "--confirm"], catch_exceptions=False
            )

        assert result.exit_code == 0, result.output
        assert order.mock_calls[:3] == [
            call.find([5], include_orphans=False),
            call.writers([5], False),
            call.workers([123], False),
        ]


class TestFlushDbValidation:
    def test_db_above_redis_default_range_is_allowed(self):
        """Configured Redis servers may expose DB numbers above 15."""
        with (
            patch("gigaevo.cli.flush.flush_db", return_value=True) as mock_flush,
            patch("gigaevo.cli.flush.find_exec_runner_pids", return_value=[]),
            patch("gigaevo.cli.flush.kill_workers"),
            patch("gigaevo.cli.flush.kill_run_writers", return_value=[]),
        ):
            runner = CliRunner()
            result = runner.invoke(
                main, ["flush", "--db", "16"], catch_exceptions=False
            )
        assert result.exit_code == 0, result.output
        mock_flush.assert_called_once_with(16, "localhost", 6379, True)

    def test_negative_db_errors(self):
        """Negative DB number shows error."""
        runner = CliRunner()
        result = runner.invoke(main, ["flush", "--db", "-1"], catch_exceptions=False)
        assert result.exit_code != 0

    def test_kill_only_and_no_kill_workers_conflict(self):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["flush", "--db", "4", "--kill-only", "--no-kill-workers"],
            catch_exceptions=False,
        )
        assert result.exit_code == 2
        assert "cannot be used together" in result.output


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
            patch("gigaevo.cli.flush.kill_run_writers", return_value=[]),
        ):
            mock_flush.return_value = False
            runner = CliRunner()
            result = runner.invoke(
                main, ["flush", "--db", "5", "--confirm"], catch_exceptions=False
            )
            assert result.exit_code == 1
