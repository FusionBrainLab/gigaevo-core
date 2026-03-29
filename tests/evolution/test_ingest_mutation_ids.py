"""Mutation-killing tests for _ingest_completed_programs mutation_ids fast path.

The mutation_ids parameter controls which DONE programs are deserialized vs
batch-discarded. These tests catch operator mutations (in/not in inversions),
None vs [] edge cases, and overlapping ID sets.

Gap identified by mutation testing analysis: +15-20% kill rate improvement.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.evolution.engine.config import EngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS, Program
from gigaevo.programs.program_state import ProgramState


# ---------------------------------------------------------------------------
# Helpers (mirror test_evolution_engine.py patterns)
# ---------------------------------------------------------------------------


def _make_engine() -> EvolutionEngine:
    storage = AsyncMock()
    strategy = AsyncMock()
    writer = MagicMock()
    writer.bind.return_value = writer
    metrics_tracker = MagicMock()

    storage.count_by_status.return_value = 0
    storage.get_all_by_status.return_value = []
    storage.get_ids_by_status.return_value = []

    engine = EvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=AsyncMock(),
        config=EngineConfig(),
        writer=writer,
        metrics_tracker=metrics_tracker,
    )
    engine.state = AsyncMock()
    return engine


def _prog(state: ProgramState = ProgramState.DONE) -> Program:
    return Program(code="def solve(): return 42", state=state)


# ===========================================================================
# mutation_ids=None path (no fast-discard)
# ===========================================================================


class TestIngestMutationIdsNone:
    """When mutation_ids is None, ALL non-archive DONE programs should be
    deserialized and evaluated — no fast-discard.
    """

    async def test_none_deserializes_all_non_archive(self) -> None:
        """mutation_ids=None: every non-archive DONE program is deserialized."""
        engine = _make_engine()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True

        p1 = _prog()
        p2 = _prog()
        engine.storage.get_ids_by_status.return_value = [p1.id, p2.id]
        engine.storage.mget.return_value = [p1, p2]
        engine.strategy.get_program_ids.return_value = []

        await engine._ingest_completed_programs(mutation_ids=None)

        # Both programs should be deserialized (mget called with both IDs)
        call_args = engine.storage.mget.call_args
        assert set(call_args[0][0]) == {p1.id, p2.id}

    async def test_none_does_not_call_batch_move(self) -> None:
        """mutation_ids=None: batch_move_status_sets is NOT called for stale discard."""
        engine = _make_engine()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True

        p1 = _prog()
        engine.storage.get_ids_by_status.return_value = [p1.id]
        engine.storage.mget.return_value = [p1]
        engine.strategy.get_program_ids.return_value = []

        await engine._ingest_completed_programs(mutation_ids=None)

        engine.storage.batch_move_status_sets.assert_not_called()


# ===========================================================================
# mutation_ids=[] path (empty list — everything is stale)
# ===========================================================================


class TestIngestMutationIdsEmpty:
    """When mutation_ids is an empty list, ALL non-archive DONE programs
    should be considered stale and batch-discarded.
    """

    async def test_empty_list_discards_all_non_archive(self) -> None:
        """mutation_ids=[]: all non-archive DONE programs are stale → batch discard."""
        engine = _make_engine()

        stale1 = _prog()
        stale2 = _prog()
        engine.storage.get_ids_by_status.return_value = [stale1.id, stale2.id]
        engine.strategy.get_program_ids.return_value = []

        await engine._ingest_completed_programs(mutation_ids=[])

        # All should be batch-discarded as stale
        engine.storage.batch_move_status_sets.assert_called_once()
        stale_ids = engine.storage.batch_move_status_sets.call_args[0][0]
        assert set(stale_ids) == {stale1.id, stale2.id}

    async def test_empty_list_does_not_deserialize(self) -> None:
        """mutation_ids=[]: no programs should be deserialized (mget not called)."""
        engine = _make_engine()

        stale = _prog()
        engine.storage.get_ids_by_status.return_value = [stale.id]
        engine.strategy.get_program_ids.return_value = []

        await engine._ingest_completed_programs(mutation_ids=[])

        engine.storage.mget.assert_not_called()


# ===========================================================================
# mutation_ids with specific IDs (fast-discard stale, deserialize new)
# ===========================================================================


class TestIngestMutationIdsPartition:
    """When mutation_ids contains specific IDs, only those IDs should be
    deserialized. Everything else is stale → batch-discard.
    """

    async def test_stale_discarded_new_deserialized(self) -> None:
        """mutation_ids separates stale (not in set) from new (in set)."""
        engine = _make_engine()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True

        new_prog = _prog()
        stale_prog = _prog()
        engine.storage.get_ids_by_status.return_value = [new_prog.id, stale_prog.id]
        engine.storage.mget.return_value = [new_prog]
        engine.strategy.get_program_ids.return_value = []

        await engine._ingest_completed_programs(mutation_ids=[new_prog.id])

        # Stale program should be batch-discarded
        engine.storage.batch_move_status_sets.assert_called_once()
        stale_ids = engine.storage.batch_move_status_sets.call_args[0][0]
        assert stale_ids == [stale_prog.id]

        # New program should be deserialized
        mget_ids = engine.storage.mget.call_args[0][0]
        assert mget_ids == [new_prog.id]

    async def test_mutation_id_in_archive_not_deserialized(self) -> None:
        """If a mutation_id is already in the archive, it's filtered out before
        the fast-discard check (archive filter happens first).
        """
        engine = _make_engine()

        archive_prog = _prog()
        engine.storage.get_ids_by_status.return_value = [archive_prog.id]
        engine.strategy.get_program_ids.return_value = [archive_prog.id]

        await engine._ingest_completed_programs(mutation_ids=[archive_prog.id])

        # Archive programs are filtered before mutation_ids check
        engine.storage.mget.assert_not_called()
        engine.storage.batch_move_status_sets.assert_not_called()

    async def test_all_non_archive_are_in_mutation_ids(self) -> None:
        """When every non-archive DONE program is in mutation_ids, no stale discard."""
        engine = _make_engine()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True

        p1 = _prog()
        p2 = _prog()
        engine.storage.get_ids_by_status.return_value = [p1.id, p2.id]
        engine.storage.mget.return_value = [p1, p2]
        engine.strategy.get_program_ids.return_value = []

        await engine._ingest_completed_programs(mutation_ids=[p1.id, p2.id])

        engine.storage.batch_move_status_sets.assert_not_called()
        mget_ids = engine.storage.mget.call_args[0][0]
        assert set(mget_ids) == {p1.id, p2.id}

    async def test_batch_discard_exception_doesnt_crash(self) -> None:
        """If batch_move_status_sets raises, ingestion should continue."""
        engine = _make_engine()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True

        new_prog = _prog()
        stale_prog = _prog()
        engine.storage.get_ids_by_status.return_value = [new_prog.id, stale_prog.id]
        engine.storage.mget.return_value = [new_prog]
        engine.strategy.get_program_ids.return_value = []
        engine.storage.batch_move_status_sets.side_effect = RuntimeError("Redis error")

        # Should NOT raise — exception is caught and logged
        await engine._ingest_completed_programs(mutation_ids=[new_prog.id])

        # New program should still be processed despite stale discard failure
        engine.strategy.add.assert_called_once_with(new_prog)


# ===========================================================================
# Chaos-hacker fixes: post-mget state filter, exclude param, mixed states
# ===========================================================================


class TestIngestPostMgetStateFilter:
    """The production code filters mget results to only DONE programs (line 474).
    Without this, stale programs that changed state between SMEMBERS and mget
    would slip through to strategy.add().
    """

    async def test_non_done_programs_filtered_after_mget(self) -> None:
        """Programs that changed state between SMEMBERS and mget are filtered out.

        MUTATION TARGET: removing the `if p.state == ProgramState.DONE` filter
        would let non-DONE programs reach strategy.add().
        """
        engine = _make_engine()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True

        done_prog = _prog(ProgramState.DONE)
        # Simulate a program that was DONE during SMEMBERS but changed to RUNNING
        # by the time mget deserializes it (TOCTOU race)
        stale_running = _prog(ProgramState.RUNNING)

        engine.storage.get_ids_by_status.return_value = [done_prog.id, stale_running.id]
        engine.storage.mget.return_value = [done_prog, stale_running]
        engine.strategy.get_program_ids.return_value = []

        await engine._ingest_completed_programs(mutation_ids=None)

        # Only the DONE program should reach strategy.add
        engine.strategy.add.assert_called_once_with(done_prog)

    async def test_all_non_done_after_mget_skips_processing(self) -> None:
        """If all programs changed state, processing should be skipped entirely."""
        engine = _make_engine()

        running = _prog(ProgramState.RUNNING)
        queued = _prog(ProgramState.QUEUED)

        engine.storage.get_ids_by_status.return_value = [running.id, queued.id]
        engine.storage.mget.return_value = [running, queued]
        engine.strategy.get_program_ids.return_value = []

        await engine._ingest_completed_programs(mutation_ids=None)

        engine.strategy.add.assert_not_called()


class TestIngestExcludeParameter:
    """Verify that mget is called with exclude=EXCLUDE_STAGE_RESULTS."""

    async def test_mget_passes_exclude_stage_results(self) -> None:
        """MUTATION TARGET: removing the exclude kwarg silently bloats payloads."""
        engine = _make_engine()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True

        prog = _prog()
        engine.storage.get_ids_by_status.return_value = [prog.id]
        engine.storage.mget.return_value = [prog]
        engine.strategy.get_program_ids.return_value = []

        await engine._ingest_completed_programs(mutation_ids=None)

        call_kwargs = engine.storage.mget.call_args
        assert call_kwargs.kwargs.get("exclude") == EXCLUDE_STAGE_RESULTS, (
            "mget must be called with exclude=EXCLUDE_STAGE_RESULTS "
            "to avoid deserializing stage results during ingestion"
        )
