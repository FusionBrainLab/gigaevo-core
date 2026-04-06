"""Tests for the adversarial shared utilities."""

import pytest

from problems.adversarial.shared import exec_entrypoint, reset_cache


class TestExecEntrypoint:
    def test_simple_return(self):
        assert exec_entrypoint("def entrypoint():\n    return 42\n") == 42

    def test_returns_callable(self):
        code = "def entrypoint():\n    def fn(x): return x * 2\n    return fn\n"
        result = exec_entrypoint(code)
        assert callable(result)
        assert result(5) == 10

    def test_returns_tuple(self):
        code = "def entrypoint():\n    return (1, 2, 3)\n"
        assert exec_entrypoint(code) == (1, 2, 3)

    def test_missing_entrypoint_raises(self):
        with pytest.raises(ValueError, match="no callable entrypoint"):
            exec_entrypoint("x = 42\n")

    def test_syntax_error_raises(self):
        with pytest.raises(ValueError):
            exec_entrypoint("def entrypoint(\n")

    def test_runtime_error_raises(self):
        with pytest.raises(ValueError, match="failed"):
            exec_entrypoint("def entrypoint():\n    return 1 / 0\n")

    def test_timeout_on_infinite_loop(self):
        with pytest.raises(TimeoutError):
            exec_entrypoint("def entrypoint():\n    while True: pass\n", timeout=0.5)


class TestResetCache:
    def test_reset_clears_cache(self):
        # Just verify it doesn't crash
        reset_cache()
