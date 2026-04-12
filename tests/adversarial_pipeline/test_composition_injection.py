"""Tests for CompositionInjectionHook."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from gigaevo.adversarial.composition_injection import CompositionInjectionHook
from gigaevo.adversarial.opponent_provider import OpponentProgram


@pytest.fixture
def d_provider():
    return AsyncMock()


@pytest.fixture
def g_storage():
    return AsyncMock()


@pytest.fixture
def hook(d_provider, g_storage):
    return CompositionInjectionHook(d_provider=d_provider, g_storage=g_storage)


@pytest.mark.asyncio
async def test_injects_d_best_into_g(hook, d_provider, g_storage):
    """Hook reads D's best and creates a tagged program in G's storage."""
    d_provider.get_top_k.return_value = [
        OpponentProgram(
            program_id="d-best-1",
            code="def improve(pts): return pts",
            fitness=0.8,
        )
    ]
    result = await hook.inject()
    assert result is not None
    g_storage.add.assert_called_once()
    injected = g_storage.add.call_args[0][0]
    assert injected.code == "def improve(pts): return pts"
    assert injected.metadata["mutation_type"] == "d_improvement"
    assert injected.metadata["d_source_id"] == "d-best-1"
    assert injected.metadata["d_fitness"] == 0.8


@pytest.mark.asyncio
async def test_empty_d_archive_no_injection(hook, d_provider, g_storage):
    """No injection when D's archive is empty."""
    d_provider.get_top_k.return_value = []
    result = await hook.inject()
    assert result is None
    g_storage.add.assert_not_called()


@pytest.mark.asyncio
async def test_returns_injected_program_id(hook, d_provider, g_storage):
    """inject() returns the UUID of the injected program."""
    d_provider.get_top_k.return_value = [
        OpponentProgram(
            program_id="d-1",
            code="def improve(pts): pass",
            fitness=0.5,
        )
    ]
    result = await hook.inject()
    assert isinstance(result, str)
    assert len(result) == 36  # UUID format


@pytest.mark.asyncio
async def test_injected_program_is_valid_program_object(hook, d_provider, g_storage):
    """The injected object is a proper Program with code and metadata."""
    d_provider.get_top_k.return_value = [
        OpponentProgram(
            program_id="d-2",
            code="def improve(pts): return [[0,0]]*11",
            fitness=0.6,
        )
    ]
    await hook.inject()
    from gigaevo.programs.program import Program

    injected = g_storage.add.call_args[0][0]
    assert isinstance(injected, Program)
    assert injected.code == "def improve(pts): return [[0,0]]*11"
