"""Tests for GradientInPromptStage."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.adversarial.gradient_prompt import GradientInPromptStage
from gigaevo.adversarial.opponent_provider import OpponentProgram
from gigaevo.programs.program import Program


def _dummy_program() -> MagicMock:
    prog = MagicMock(spec=Program)
    prog.id = "dummy"
    return prog


@pytest.fixture
def provider():
    return AsyncMock()


@pytest.fixture
def stage(provider):
    return GradientInPromptStage(opponent_provider=provider, timeout=10.0)


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
