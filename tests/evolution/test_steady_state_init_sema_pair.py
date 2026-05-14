"""SteadyStateEvolutionEngine.__init__ must allocate both semaphores."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.evolution.engine.config import SteadyStateEngineConfig
from gigaevo.evolution.engine.steady_state import SteadyStateEvolutionEngine


def _make_engine(max_in_flight: int = 7) -> SteadyStateEvolutionEngine:
    cfg = SteadyStateEngineConfig(max_in_flight=max_in_flight)
    writer = MagicMock()
    writer.bind.return_value = writer
    return SteadyStateEvolutionEngine(
        config=cfg,
        storage=AsyncMock(),
        strategy=AsyncMock(),
        mutation_operator=AsyncMock(),
        writer=writer,
        metrics_tracker=MagicMock(),
    )


@pytest.mark.asyncio
async def test_engine_init_creates_both_semaphores() -> None:
    engine = _make_engine(max_in_flight=7)
    assert isinstance(engine._producer_sema, asyncio.Semaphore)
    assert isinstance(engine._buffer_sema, asyncio.Semaphore)
    # Both sized symmetrically to the single knob.
    assert engine._producer_sema._value == 7
    assert engine._buffer_sema._value == 7
