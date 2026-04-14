"""Tests for GradientInPromptStage."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.adversarial.gradient_prompt import GradientInPromptStage
from gigaevo.adversarial.opponent_provider import OpponentProgram
from gigaevo.programs.program import Program


def _dummy_program(program_id: str = "dummy") -> MagicMock:
    prog = MagicMock(spec=Program)
    prog.id = program_id
    return prog


@pytest.fixture
def provider():
    return AsyncMock()


@pytest.fixture
def dg_tracker():
    return AsyncMock()


@pytest.fixture
def stage(provider):
    return GradientInPromptStage(opponent_provider=provider, timeout=10.0)


@pytest.fixture
def stage_with_tracker(provider, dg_tracker):
    return GradientInPromptStage(
        opponent_provider=provider,
        dg_tracker=dg_tracker,
        timeout=10.0,
    )


@pytest.mark.asyncio
async def test_formats_improvement_strategy_section(stage, provider):
    """Stage returns D's best code as 'Improvement Strategy from Opponent'."""
    provider.get_top_k.return_value = [
        OpponentProgram(
            program_id="d-1",
            code="def improve(pts):\n    return pts * 1.1",
            fitness=0.7,
        )
    ]
    result = await stage.compute(_dummy_program())
    assert "Improvement Strategy from Opponent" in result.data
    assert "def improve(pts):" in result.data


@pytest.mark.asyncio
async def test_empty_archive_returns_empty(stage, provider):
    """Cold start: empty string."""
    provider.get_top_k.return_value = []
    result = await stage.compute(_dummy_program())
    assert result.data == ""


@pytest.mark.asyncio
async def test_no_d_improvement_tag(stage, provider):
    """Gradient-in-prompt does NOT inject code — no d_improvement tag."""
    provider.get_top_k.return_value = [
        OpponentProgram(
            program_id="d-1",
            code="def improve(pts): pass",
            fitness=0.5,
        )
    ]
    result = await stage.compute(_dummy_program())
    assert "d_improvement" not in result.data


@pytest.mark.asyncio
async def test_shows_fitness_value(stage, provider):
    """The fitness value is shown in the prompt."""
    provider.get_top_k.return_value = [
        OpponentProgram(
            program_id="d-1",
            code="def improve(): pass",
            fitness=0.12345,
        )
    ]
    result = await stage.compute(_dummy_program())
    assert "0.12345" in result.data


@pytest.mark.asyncio
async def test_uses_no_cache():
    """Stage must use NO_CACHE — D's archive evolves."""
    from gigaevo.programs.stages.cache_handler import NO_CACHE

    provider = AsyncMock()
    stage = GradientInPromptStage(opponent_provider=provider, timeout=10.0)
    assert stage.cache_handler is NO_CACHE


@pytest.mark.asyncio
async def test_requests_top_1(stage, provider):
    """Stage always requests top-1 from D's archive."""
    provider.get_top_k.return_value = [
        OpponentProgram(program_id="d-1", code="code", fitness=0.5)
    ]
    await stage.compute(_dummy_program())
    provider.get_top_k.assert_called_once_with(1, higher_is_better=True)


# ===================================================================
# Tests for dg_tracker integration (per-G-program D selection)
# ===================================================================


@pytest.mark.asyncio
async def test_tracker_selects_per_program_d(stage_with_tracker, provider, dg_tracker):
    """When dg_tracker has data for the specific G, selects that D (not global best)."""
    dg_tracker.get_best_d_for_g.return_value = ("d-specific", 0.08)
    provider.get_programs_by_ids.return_value = [
        OpponentProgram(
            program_id="d-specific", code="def improve(): pass", fitness=0.6
        )
    ]
    result = await stage_with_tracker.compute(_dummy_program("g-42"))
    dg_tracker.get_best_d_for_g.assert_called_once_with("g-42")
    assert "def improve(): pass" in result.data
    # Should NOT call get_top_k since tracker had data
    provider.get_top_k.assert_not_called()


@pytest.mark.asyncio
async def test_tracker_none_falls_back_to_global(
    stage_with_tracker, provider, dg_tracker
):
    """When dg_tracker returns None (no data for G), falls back to global best D."""
    dg_tracker.get_best_d_for_g.return_value = None
    provider.get_top_k.return_value = [
        OpponentProgram(
            program_id="d-global", code="def global_improve(): pass", fitness=0.9
        )
    ]
    result = await stage_with_tracker.compute(_dummy_program("g-99"))
    provider.get_top_k.assert_called_once_with(1, higher_is_better=True)
    assert "def global_improve(): pass" in result.data


@pytest.mark.asyncio
async def test_no_tracker_uses_global_best(stage, provider):
    """When dg_tracker is None (not configured), uses global best D (backward compatible)."""
    provider.get_top_k.return_value = [
        OpponentProgram(program_id="d-1", code="def fallback(): pass", fitness=0.5)
    ]
    result = await stage.compute(_dummy_program("g-1"))
    provider.get_top_k.assert_called_once_with(1, higher_is_better=True)
    assert "def fallback(): pass" in result.data


@pytest.mark.asyncio
async def test_tracker_queries_by_program_id(stage_with_tracker, provider, dg_tracker):
    """Stage uses program.id to query dg_tracker.get_best_d_for_g(program.id)."""
    dg_tracker.get_best_d_for_g.return_value = ("d-match", 0.1)
    provider.get_programs_by_ids.return_value = [
        OpponentProgram(program_id="d-match", code="code", fitness=0.7)
    ]
    prog = _dummy_program("specific-g-id-123")
    await stage_with_tracker.compute(prog)
    dg_tracker.get_best_d_for_g.assert_called_once_with("specific-g-id-123")


@pytest.mark.asyncio
async def test_tracker_fetches_d_code_by_id(stage_with_tracker, provider, dg_tracker):
    """Stage fetches full D code via get_programs_by_ids after getting d_id from tracker."""
    dg_tracker.get_best_d_for_g.return_value = ("d-abc", 0.05)
    provider.get_programs_by_ids.return_value = [
        OpponentProgram(program_id="d-abc", code="def specific_d(): pass", fitness=0.6)
    ]
    result = await stage_with_tracker.compute(_dummy_program("g-1"))
    provider.get_programs_by_ids.assert_called_once_with(["d-abc"])
    assert "def specific_d(): pass" in result.data


@pytest.mark.asyncio
async def test_tracker_d_evicted_falls_back_to_global(
    stage_with_tracker, provider, dg_tracker
):
    """When tracker returns d_id but get_programs_by_ids returns empty, falls back to global."""
    dg_tracker.get_best_d_for_g.return_value = ("d-evicted", 0.1)
    provider.get_programs_by_ids.return_value = []  # D was evicted
    provider.get_top_k.return_value = [
        OpponentProgram(program_id="d-global", code="def global_d(): pass", fitness=0.8)
    ]
    result = await stage_with_tracker.compute(_dummy_program("g-1"))
    provider.get_top_k.assert_called_once_with(1, higher_is_better=True)
    assert "def global_d(): pass" in result.data


@pytest.mark.asyncio
async def test_prompt_includes_d_fitness_and_code(
    stage_with_tracker, provider, dg_tracker
):
    """The prompt text includes the specific D's fitness and code."""
    dg_tracker.get_best_d_for_g.return_value = ("d-1", 0.07)
    provider.get_programs_by_ids.return_value = [
        OpponentProgram(
            program_id="d-1",
            code="def special_improve(pts): return pts * 2",
            fitness=0.54321,
        )
    ]
    result = await stage_with_tracker.compute(_dummy_program("g-1"))
    assert "0.54321" in result.data
    assert "def special_improve(pts): return pts * 2" in result.data


@pytest.mark.asyncio
async def test_empty_archive_no_tracker_returns_empty(
    stage_with_tracker, provider, dg_tracker
):
    """Empty D archive + no tracker data returns empty string (cold start)."""
    dg_tracker.get_best_d_for_g.return_value = None
    provider.get_top_k.return_value = []
    result = await stage_with_tracker.compute(_dummy_program("g-1"))
    assert result.data == ""
