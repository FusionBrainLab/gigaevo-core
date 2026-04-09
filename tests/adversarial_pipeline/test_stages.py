"""Tests for gigaevo.adversarial.stages (two-stage DAG pattern)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from gigaevo.adversarial.opponent_provider import (
    OpponentArchiveProvider,
    OpponentProgram,
)
from gigaevo.adversarial.stages import FetchOpponentIdsStage, FetchOpponentResultsStage
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

    async def get_codes_by_ids(self, ids: list[str]) -> list[str]:
        id_map = {o.program_id: o.code for o in self._opponents}
        return [id_map[i] for i in ids if i in id_map]


def _make_program(code: str = "def entrypoint(): return 42") -> Program:
    return Program(code=code)


# ---------------------------------------------------------------------------
# Tests: FetchOpponentIdsStage
# ---------------------------------------------------------------------------


class TestFetchOpponentIdsStage:
    def test_uses_no_cache(self):
        """Stage must use NO_CACHE handler — always re-samples on every DAG run."""
        from gigaevo.programs.stages.cache_handler import NO_CACHE

        provider = FakeProvider()
        stage = FetchOpponentIdsStage(opponent_provider=provider, timeout=60.0)
        assert stage.cache_handler is NO_CACHE

    def test_init_defaults(self):
        provider = FakeProvider()
        stage = FetchOpponentIdsStage(opponent_provider=provider, timeout=60.0)
        assert stage._n == 5

    def test_init_custom_n(self):
        provider = FakeProvider()
        stage = FetchOpponentIdsStage(
            opponent_provider=provider, n_opponents=3, timeout=60.0
        )
        assert stage._n == 3

    @pytest.mark.asyncio
    async def test_returns_box_of_ids(self):
        """compute() returns Box[Any](data=[id1, id2, ...])."""
        opponents = [
            OpponentProgram(program_id="p1", code="c1", fitness=0.5),
            OpponentProgram(program_id="p2", code="c2", fitness=0.8),
        ]
        provider = FakeProvider(opponents)
        stage = FetchOpponentIdsStage(opponent_provider=provider, timeout=60.0)
        program = _make_program()
        result = await stage.compute(program)
        assert isinstance(result, Box)
        assert set(result.data) == {"p1", "p2"}

    @pytest.mark.asyncio
    async def test_empty_archive_returns_empty_box(self):
        provider = FakeProvider([])
        stage = FetchOpponentIdsStage(opponent_provider=provider, timeout=60.0)
        program = _make_program()
        result = await stage.compute(program)
        assert isinstance(result, Box)
        assert result.data == []

    @pytest.mark.asyncio
    async def test_respects_n_opponents(self):
        opponents = [
            OpponentProgram(program_id=f"p{i}", code=f"c{i}", fitness=float(i))
            for i in range(10)
        ]
        provider = FakeProvider(opponents)
        stage = FetchOpponentIdsStage(
            opponent_provider=provider, n_opponents=3, timeout=60.0
        )
        program = _make_program()
        result = await stage.compute(program)
        assert len(result.data) <= 3


# ---------------------------------------------------------------------------
# Tests: FetchOpponentResultsStage
# ---------------------------------------------------------------------------


class TestFetchOpponentResultsStage:
    def test_uses_default_cache(self):
        """Stage uses DEFAULT_CACHE (InputHashCache) — reruns when opponent IDs change."""
        from gigaevo.programs.stages.cache_handler import DEFAULT_CACHE

        provider = FakeProvider()
        stage = FetchOpponentResultsStage(
            opponent_provider=provider,
            timeout=60.0,
        )
        assert stage.cache_handler is DEFAULT_CACHE

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
        stage.attach_inputs({"opponent_ids": Box[object](data=[])})
        program = _make_program()
        result = await stage.compute(program)
        assert isinstance(result, Box)
        assert result.data == []

    @pytest.mark.asyncio
    async def test_uses_fallback_when_archive_empty(self):
        """When archive is empty (no IDs), falls back to fallback_codes."""
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
        stage.attach_inputs({"opponent_ids": Box[object](data=[])})

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
        """Executes opponent entrypoint() via run_exec_runner for fetched IDs."""
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
        stage.attach_inputs({"opponent_ids": Box[object](data=["p1", "p2"])})

        call_count = 0

        async def mock_exec(**_kwargs):
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
        stage.attach_inputs({"opponent_ids": Box[object](data=["p1", "p2"])})

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
    async def test_unknown_ids_silently_skipped(self):
        """IDs not in provider cache are silently skipped (not an error)."""
        opponents = [
            OpponentProgram(
                program_id="p1", code="def entrypoint(): return 1", fitness=0.5
            ),
        ]
        provider = FakeProvider(opponents)
        stage = FetchOpponentResultsStage(
            opponent_provider=provider,
            timeout=60.0,
        )
        # Request IDs including one that doesn't exist
        stage.attach_inputs({"opponent_ids": Box[object](data=["p1", "unknown_id"])})

        async def mock_exec(**_kwargs):
            return (42, b"", "")

        program = _make_program()
        with patch("gigaevo.adversarial.stages.run_exec_runner", side_effect=mock_exec):
            result = await stage.compute(program)

        # Only p1 found; unknown_id silently skipped
        assert len(result.data) == 1
