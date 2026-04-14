"""Tests for gigaevo.cli.flush_ops — process kill and Redis flush operations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from gigaevo.cli.flush_ops import _is_run_py_line, flush_db


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
