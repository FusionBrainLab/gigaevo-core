"""Tests for gigaevo.adversarial.stages.FetchOpponentResultsStage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from gigaevo.adversarial.opponent_provider import (
    OpponentArchiveProvider,
    OpponentProgram,
)
from gigaevo.adversarial.stages import FetchOpponentResultsStage
from gigaevo.programs.program import Program
from gigaevo.programs.stages.common import Box

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeProvider(OpponentArchiveProvider):
    """Provider that returns pre-set opponents."""

    def __init__(self, opponents: list[OpponentProgram] | None = None):
        self._opponents = opponents or []

    async def get_opponents(self, n: int = 5) -> list[OpponentProgram]:
        return self._opponents[:n]


def _make_program(code: str = "def entrypoint(): return 42") -> Program:
    return Program(code=code)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFetchOpponentResultsStage:
    def test_init_defaults(self):
        provider = FakeProvider()
        stage = FetchOpponentResultsStage(
            opponent_provider=provider,
            timeout=60.0,
        )
        assert stage._n == 5
        assert stage._fallback_codes == []
        assert stage._per_timeout == 10.0

    def test_init_custom(self):
        provider = FakeProvider()
        stage = FetchOpponentResultsStage(
            opponent_provider=provider,
            n_opponents=3,
            fallback_codes=["code1", "code2"],
            per_opponent_timeout=5.0,
            python_path=[Path("/some/path")],
            max_memory_mb=512,
            timeout=30.0,
        )
        assert stage._n == 3
        assert len(stage._fallback_codes) == 2
        assert stage._per_timeout == 5.0

    @pytest.mark.asyncio
    async def test_empty_archive_no_fallback_returns_empty(self):
        """When archive is empty and no fallback, returns empty list."""
        provider = FakeProvider([])
        stage = FetchOpponentResultsStage(
            opponent_provider=provider,
            timeout=60.0,
        )
        program = _make_program()
        result = await stage.compute(program)
        assert isinstance(result, Box)
        assert result.data == []

    @pytest.mark.asyncio
    async def test_uses_fallback_when_archive_empty(self):
        """When archive is empty, falls back to fallback_codes."""
        provider = FakeProvider([])
        fallback_codes = [
            "def entrypoint(): return 'fallback_a'",
            "def entrypoint(): return 'fallback_b'",
        ]
        stage = FetchOpponentResultsStage(
            opponent_provider=provider,
            fallback_codes=fallback_codes,
            timeout=60.0,
        )

        # Mock run_exec_runner to return simple values
        async def mock_exec(**kwargs):
            code = kwargs["code"]
            if "fallback_a" in code:
                return ("fallback_a", b"", "")
            return ("fallback_b", b"", "")

        program = _make_program()
        with patch("gigaevo.adversarial.stages.run_exec_runner", side_effect=mock_exec):
            result = await stage.compute(program)

        assert isinstance(result, Box)
        assert len(result.data) == 2
        assert "fallback_a" in result.data
        assert "fallback_b" in result.data

    @pytest.mark.asyncio
    async def test_executes_opponent_codes(self):
        """Executes opponent entrypoint() via run_exec_runner."""
        opponents = [
            OpponentProgram(
                program_id="p1", code="def entrypoint(): return 1", fitness=0.5
            ),
            OpponentProgram(
                program_id="p2", code="def entrypoint(): return 2", fitness=0.8
            ),
        ]
        provider = FakeProvider(opponents)
        stage = FetchOpponentResultsStage(
            opponent_provider=provider,
            timeout=60.0,
        )

        call_count = 0

        async def mock_exec(**kwargs):
            nonlocal call_count
            call_count += 1
            return (call_count, b"", "")

        program = _make_program()
        with patch("gigaevo.adversarial.stages.run_exec_runner", side_effect=mock_exec):
            result = await stage.compute(program)

        assert isinstance(result, Box)
        assert len(result.data) == 2
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_filters_out_failed_opponents(self):
        """Opponents that raise exceptions are filtered out."""
        opponents = [
            OpponentProgram(program_id="p1", code="good", fitness=0.5),
            OpponentProgram(program_id="p2", code="bad", fitness=0.8),
        ]
        provider = FakeProvider(opponents)
        stage = FetchOpponentResultsStage(
            opponent_provider=provider,
            timeout=60.0,
        )

        async def mock_exec(**kwargs):
            if kwargs["code"] == "bad":
                raise TimeoutError("timed out")
            return ("good_result", b"", "")

        program = _make_program()
        with patch("gigaevo.adversarial.stages.run_exec_runner", side_effect=mock_exec):
            result = await stage.compute(program)

        assert isinstance(result, Box)
        assert len(result.data) == 1
        assert result.data[0] == "good_result"

    @pytest.mark.asyncio
    async def test_no_cache(self):
        """Stage should use NO_CACHE handler."""
        from gigaevo.programs.stages.cache_handler import NO_CACHE

        provider = FakeProvider()
        stage = FetchOpponentResultsStage(
            opponent_provider=provider,
            timeout=60.0,
        )
        assert stage.cache_handler is NO_CACHE
