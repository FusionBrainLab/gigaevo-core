"""Engine advances ``refresh_pass`` via ``_write_snapshot``.

After Phase 3, ``SteadyStateEvolutionEngine._refresh_archive_programs`` is
the only component that bumps ``EngineSnapshot.refresh_pass``; the
filtered lineage stage stops owning the counter. These tests pin that
behaviour end-to-end — both the in-memory engine snapshot AND the Redis
copy AND the in-process mirror advance in lockstep.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.evolution.engine.config import SteadyStateEngineConfig
from gigaevo.evolution.engine.snapshot import (
    _reset_current_snapshot_for_tests,
    get_current_snapshot,
    load_engine_snapshot,
)
from gigaevo.evolution.engine.steady_state import SteadyStateEvolutionEngine
from gigaevo.evolution.engine.stopper import EvolutionStopper
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


@pytest.fixture(autouse=True)
def _reset_mirror():
    _reset_current_snapshot_for_tests()
    yield
    _reset_current_snapshot_for_tests()


def _make_ss_engine(passes: int) -> SteadyStateEvolutionEngine:
    storage = AsyncMock()
    strategy = AsyncMock()
    writer = MagicMock()
    writer.bind.return_value = writer
    metrics_tracker = MagicMock()
    metrics_tracker.format_best_summary.return_value = ""

    # Minimal archive: one program survives the refresh.
    prog = Program(code="def f(): return 0", state=ProgramState.DONE)
    strategy.get_program_ids.return_value = [prog.id]
    storage.mget.return_value = [prog]
    storage.batch_transition_by_ids.return_value = 1
    storage.count_by_status.return_value = 0
    storage.get_all_by_status.return_value = []
    storage.get_ids_by_status.return_value = []
    storage.snapshot = MagicMock()

    # Minimal run-state KV so load_engine_snapshot() sees what
    # _write_snapshot() saved.
    run_state: dict[str, str] = {}

    async def _save(field: str, value):
        run_state[field] = str(value)

    async def _load(field: str):
        return run_state.get(field)

    storage.save_run_state.side_effect = _save
    storage.load_run_state_str.side_effect = _load

    config = SteadyStateEngineConfig(
        max_in_flight=1,
        max_mutations_per_generation=1,
        stopper=EvolutionStopper(),
        refresh_order="generation_bucketed",
        refresh_passes=passes,
        loop_interval=0.01,
    )
    engine = SteadyStateEvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=AsyncMock(),
        config=config,
        writer=writer,
        metrics_tracker=metrics_tracker,
    )
    engine.state = AsyncMock()
    engine._await_idle = AsyncMock()  # type: ignore[method-assign]
    return engine


@pytest.mark.asyncio
async def test_refresh_archive_programs_advances_refresh_pass_once():
    engine = _make_ss_engine(passes=1)
    assert engine._snapshot.refresh_pass == 0

    await engine._refresh_archive_programs()

    assert engine._snapshot.refresh_pass == 1
    snap = await load_engine_snapshot(engine.storage)
    assert snap.refresh_pass == 1
    assert get_current_snapshot().refresh_pass == 1


@pytest.mark.asyncio
async def test_refresh_archive_programs_advances_refresh_pass_twice():
    engine = _make_ss_engine(passes=2)
    assert engine._snapshot.refresh_pass == 0

    await engine._refresh_archive_programs()

    assert engine._snapshot.refresh_pass == 2
    snap = await load_engine_snapshot(engine.storage)
    assert snap.refresh_pass == 2
    assert get_current_snapshot().refresh_pass == 2
