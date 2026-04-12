"""Tests for get_programs_by_ids and SourceCodeInjectionStage."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.adversarial.opponent_provider import (
    OpponentProgram,
    RedisOpponentArchiveProvider,
)
from gigaevo.adversarial.source_injection import (
    SourceCodeInjectionInput,
    SourceCodeInjectionStage,
)
from gigaevo.programs.program import Program
from gigaevo.programs.stages.common import Box


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_opponent(pid: str, code: str, fitness: float) -> OpponentProgram:
    return OpponentProgram(program_id=pid, code=code, fitness=fitness)


def _make_stage(provider: AsyncMock, source_prompt_k: int = 1) -> SourceCodeInjectionStage:
    return SourceCodeInjectionStage(
        opponent_provider=provider,
        source_prompt_k=source_prompt_k,
        timeout=10.0,
    )


def _set_opponent_ids(stage: SourceCodeInjectionStage, ids: list[str]) -> None:
    """Set the stage's input to simulate FetchOpponentIdsStage output."""
    stage._raw_inputs = {"opponent_ids": Box(data=ids)}
    stage._params_obj = None


def _dummy_program() -> MagicMock:
    prog = MagicMock(spec=Program)
    prog.id = "dummy"
    return prog


# ---------------------------------------------------------------------------
# Tests: get_programs_by_ids
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_programs_by_ids_returns_opponent_programs():
    """get_programs_by_ids returns full OpponentProgram objects for matching IDs."""
    provider = AsyncMock(spec=RedisOpponentArchiveProvider)
    provider._cache = [
        _make_opponent("a", "def solve(): pass", 0.5),
        _make_opponent("b", "def solve(): return 1", 0.8),
        _make_opponent("c", "def solve(): return 2", 0.3),
    ]
    provider.get_programs_by_ids = (
        RedisOpponentArchiveProvider.get_programs_by_ids.__get__(provider)
    )
    result = await provider.get_programs_by_ids(["b", "a"])
    assert len(result) == 2
    assert {p.program_id for p in result} == {"a", "b"}


@pytest.mark.asyncio
async def test_get_programs_by_ids_skips_missing_ids():
    """IDs not in cache are silently skipped."""
    provider = AsyncMock(spec=RedisOpponentArchiveProvider)
    provider._cache = [
        _make_opponent("a", "def solve(): pass", 0.5),
    ]
    provider.get_programs_by_ids = (
        RedisOpponentArchiveProvider.get_programs_by_ids.__get__(provider)
    )
    result = await provider.get_programs_by_ids(["a", "nonexistent"])
    assert len(result) == 1
    assert result[0].program_id == "a"


@pytest.mark.asyncio
async def test_get_programs_by_ids_preserves_fitness():
    """Returned programs carry correct fitness values."""
    provider = AsyncMock(spec=RedisOpponentArchiveProvider)
    provider._cache = [
        _make_opponent("x", "code_x", 0.42),
    ]
    provider.get_programs_by_ids = (
        RedisOpponentArchiveProvider.get_programs_by_ids.__get__(provider)
    )
    result = await provider.get_programs_by_ids(["x"])
    assert result[0].fitness == 0.42
    assert result[0].code == "code_x"


# ---------------------------------------------------------------------------
# Tests: SourceCodeInjectionStage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_injects_source_from_sampled_opponents():
    """Stage uses opponent IDs from FetchOpponentIdsStage input."""
    provider = AsyncMock()
    provider.get_programs_by_ids.return_value = [
        _make_opponent("g1", "def solve():\n    return [[0,0]]*11", 0.04),
    ]
    stage = _make_stage(provider)
    _set_opponent_ids(stage, ["g1"])
    result = await stage.compute(_dummy_program())
    provider.get_programs_by_ids.assert_called_once_with(["g1"])
    assert "def solve():" in result.data
    assert "TARGET CONSTRUCTOR SOURCE CODE" in result.data


@pytest.mark.asyncio
async def test_empty_ids_returns_empty_string():
    """Empty opponent IDs → empty string."""
    provider = AsyncMock()
    stage = _make_stage(provider)
    _set_opponent_ids(stage, [])
    result = await stage.compute(_dummy_program())
    assert result.data == ""
    provider.get_programs_by_ids.assert_not_called()


@pytest.mark.asyncio
async def test_no_programs_found_returns_empty_string():
    """IDs provided but no programs found in cache → empty string."""
    provider = AsyncMock()
    provider.get_programs_by_ids.return_value = []
    stage = _make_stage(provider)
    _set_opponent_ids(stage, ["missing1"])
    result = await stage.compute(_dummy_program())
    assert result.data == ""


@pytest.mark.asyncio
async def test_shows_top_l_by_fitness():
    """When source_prompt_k=1 and 3 opponents, shows only the best."""
    provider = AsyncMock()
    provider.get_programs_by_ids.return_value = [
        _make_opponent("g1", "def solve_a(): pass", 0.02),
        _make_opponent("g2", "def solve_b(): pass", 0.05),
        _make_opponent("g3", "def solve_c(): pass", 0.03),
    ]
    stage = _make_stage(provider, source_prompt_k=1)
    _set_opponent_ids(stage, ["g1", "g2", "g3"])
    result = await stage.compute(_dummy_program())
    assert "def solve_b(): pass" in result.data  # highest fitness
    assert "def solve_a(): pass" not in result.data


@pytest.mark.asyncio
async def test_shows_all_when_l_equals_k():
    """When source_prompt_k equals number of opponents, all are shown."""
    provider = AsyncMock()
    provider.get_programs_by_ids.return_value = [
        _make_opponent("g1", "def solve_a(): pass", 0.02),
        _make_opponent("g2", "def solve_b(): pass", 0.05),
    ]
    stage = _make_stage(provider, source_prompt_k=2)
    _set_opponent_ids(stage, ["g1", "g2"])
    result = await stage.compute(_dummy_program())
    assert "def solve_a(): pass" in result.data
    assert "def solve_b(): pass" in result.data


@pytest.mark.asyncio
async def test_uses_no_cache():
    """Stage must use NO_CACHE — opponents evolve."""
    from gigaevo.programs.stages.cache_handler import NO_CACHE

    provider = AsyncMock()
    stage = _make_stage(provider)
    assert stage.cache_handler is NO_CACHE


@pytest.mark.asyncio
async def test_header_shows_count():
    """Header shows 'N of M opponents'."""
    provider = AsyncMock()
    provider.get_programs_by_ids.return_value = [
        _make_opponent("g1", "code1", 0.03),
        _make_opponent("g2", "code2", 0.05),
        _make_opponent("g3", "code3", 0.01),
    ]
    stage = _make_stage(provider, source_prompt_k=2)
    _set_opponent_ids(stage, ["g1", "g2", "g3"])
    result = await stage.compute(_dummy_program())
    assert "2 of 3 opponents" in result.data
