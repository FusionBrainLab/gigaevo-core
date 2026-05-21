"""Tests for gigaevo.cli.flush_ops — process kill and Redis flush operations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from gigaevo.cli.flush_ops import (
    _find_run_pids_for_dbs,
    _is_run_py_line,
    flush_db,
)


class TestIsRunPyLine:
    def test_matches_run_py_with_redis_db(self):
        line = "jovyan  12345 0.0 run.py redis.db=5 problem.name=hover"
        assert _is_run_py_line(line) is True

    def test_rejects_grep_line(self):
        line = "jovyan  99999 0.0 grep run.py redis.db=5"
        assert _is_run_py_line(line) is False

    def test_rejects_line_without_redis_db(self):
        line = "jovyan  12345 0.0 run.py problem.name=hover"
        assert _is_run_py_line(line) is False

    def test_rejects_unrelated_process(self):
        line = "jovyan  12345 0.0 python server.py --port 8080"
        assert _is_run_py_line(line) is False

    def test_matches_worktree_run_py(self):
        line = "jovyan  12345 0.0 .claude/worktrees/agent-abc/run.py redis.db=3"
        assert _is_run_py_line(line) is True


class TestFindRunPidsForDbs:
    """The DB-targeting matcher must not collide across DB-number prefixes.

    Regression: ``f"redis.db={db}" in line`` used a substring match, so
    ``db=1`` killed processes running on ``db=10..15``. The user's
    spherical_codes_improver on db=15 was killed by a ``flush --db 1``.
    """

    @staticmethod
    def _ps_output(*cmdlines: str) -> str:
        return "\n".join(
            f"jovyan  {1000 + i}  0.0  0.0  0  0 ?  Sl  00:00  0:00 {cmd}"
            for i, cmd in enumerate(cmdlines)
        )

    def test_db_1_does_not_match_db_10(self):
        ps = self._ps_output(
            "/home/user/python3 run.py redis.db=1 problem.name=hover",
            "/home/user/python3 run.py redis.db=10 problem.name=spherical",
        )
        with patch("gigaevo.cli.flush_ops.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=ps)
            pids = _find_run_pids_for_dbs([1])
        assert pids == {1000}, (
            f"db=1 must NOT match db=10; got {pids}. "
            "If this fails, `flush --db 1` will kill DB-10..15 workers."
        )

    def test_db_1_does_not_match_db_15(self):
        ps = self._ps_output(
            "/home/user/python3 run.py redis.db=15 problem.name=spherical_codes",
        )
        with patch("gigaevo.cli.flush_ops.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=ps)
            pids = _find_run_pids_for_dbs([1])
        assert pids == set(), (
            f"db=1 matched db=15 cmdline; got {pids}. "
            "This is the bug that killed the user's spherical run."
        )

    def test_db_2_does_not_match_db_12(self):
        ps = self._ps_output(
            "/home/user/python3 run.py redis.db=12 problem.name=hover",
        )
        with patch("gigaevo.cli.flush_ops.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=ps)
            pids = _find_run_pids_for_dbs([2])
        assert pids == set()

    def test_exact_db_match_still_works(self):
        ps = self._ps_output(
            "/home/user/python3 run.py redis.db=5 problem.name=hover",
        )
        with patch("gigaevo.cli.flush_ops.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=ps)
            pids = _find_run_pids_for_dbs([5])
        assert pids == {1000}

    def test_two_digit_db_exact_match(self):
        ps = self._ps_output(
            "/home/user/python3 run.py redis.db=15 problem.name=spherical",
        )
        with patch("gigaevo.cli.flush_ops.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=ps)
            pids = _find_run_pids_for_dbs([15])
        assert pids == {1000}

    def test_multiple_target_dbs_no_cross_collision(self):
        ps = self._ps_output(
            "/home/user/python3 run.py redis.db=1 problem.name=a",
            "/home/user/python3 run.py redis.db=10 problem.name=b",
            "/home/user/python3 run.py redis.db=11 problem.name=c",
            "/home/user/python3 run.py redis.db=2 problem.name=d",
        )
        with patch("gigaevo.cli.flush_ops.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=ps)
            pids = _find_run_pids_for_dbs([1, 2])
        assert pids == {1000, 1003}, f"expected db=1 and db=2 only, got {pids}"

    def test_db_at_end_of_line(self):
        ps = "jovyan  1000  0.0  0.0  0  0 ?  Sl  00:00  0:00 python run.py redis.db=7"
        with patch("gigaevo.cli.flush_ops.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=ps)
            pids = _find_run_pids_for_dbs([7])
        assert pids == {1000}


class TestFlushDb:
    def test_dry_run_returns_true(self):
        mock_redis = MagicMock()
        mock_redis.dbsize.return_value = 42

        with patch("gigaevo.cli.flush_ops.redis_lib.Redis", return_value=mock_redis):
            result = flush_db(5, "localhost", 6379, dry_run=True)

        assert result is True
        mock_redis.flushdb.assert_not_called()

    def test_actual_flush_returns_true_on_empty(self):
        mock_redis = MagicMock()
        mock_redis.dbsize.side_effect = [0, 0]

        with patch("gigaevo.cli.flush_ops.redis_lib.Redis", return_value=mock_redis):
            result = flush_db(5, "localhost", 6379, dry_run=False)

        assert result is True
