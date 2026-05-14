"""Resume-contract regression: both semaphores re-init at full capacity.

``__init__`` is the entire resume boundary today.  ``_in_flight`` is
rehydrated from Redis by the outer ``run.py`` recovery path, not inside
``SteadyStateEvolutionEngine``.  Because __init__ always starts with an
empty ``_in_flight`` and ``_inflight_tickets``, both semaphores MUST be
at full capacity after construction — otherwise the pipeline would be
permanently throttled on every resume.

If future code populates ``_in_flight`` from stranded RUNNING programs
directly inside ``__init__``, the implementation MUST also acquire
``_buffer_sema`` once per rehydrated id (mirroring what the producer
would have done pre-crash).  Failing to do so breaks the sema invariant
that ``_buffer_sema._value + len(_in_flight) == max_in_flight``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.evolution.engine.config import SteadyStateEngineConfig
from gigaevo.evolution.engine.steady_state import SteadyStateEvolutionEngine


def _make_engine(max_in_flight: int) -> SteadyStateEvolutionEngine:
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
async def test_resume_both_semaphores_start_at_full_capacity() -> None:
    """Fresh-constructed engine must expose both semas at N, with empty tracking sets."""
    n = 4
    engine = _make_engine(max_in_flight=n)

    assert engine._producer_sema._value == n
    assert engine._buffer_sema._value == n
    assert isinstance(engine._producer_sema, asyncio.Semaphore)
    assert isinstance(engine._buffer_sema, asyncio.Semaphore)
    # No in-flight work — tracking state must be empty.
    assert len(engine._in_flight) == 0
    assert len(engine._inflight_tickets) == 0


@pytest.mark.asyncio
async def test_resume_with_stranded_in_flight_rehydration_keeps_semas_full() -> None:
    """__init__ does NOT rehydrate _in_flight; both semas remain at full capacity.

    If a future implementation rehydrates _in_flight from stranded RUNNING
    programs found in storage, it MUST also acquire _buffer_sema once per
    rehydrated id.  The invariant is:
        _buffer_sema._value + len(_in_flight) == max_in_flight
    Today, rehydration does not happen in __init__, so this test asserts the
    baseline: semas full, _in_flight empty.
    """
    n = 3
    engine = _make_engine(max_in_flight=n)

    # Semas start at full capacity regardless of any stranded programs in storage.
    assert engine._buffer_sema._value == n
    assert engine._producer_sema._value == n
    # No rehydration happens in __init__ today.
    assert len(engine._in_flight) == 0
