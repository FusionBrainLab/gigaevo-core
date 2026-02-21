"""Unit tests for EvolutionEngine._ingest_completed_programs and _refresh_archive_programs."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from gigaevo.evolution.engine.config import EngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine() -> EvolutionEngine:
    """Build a minimal EvolutionEngine with all external dependencies mocked."""
    storage = AsyncMock()
    strategy = AsyncMock()
    writer = MagicMock()
    writer.bind.return_value = writer
    metrics_tracker = MagicMock()

    engine = EvolutionEngine(
        storage=storage,
        strategy=strategy,
        mutation_operator=MagicMock(),
        config=EngineConfig(),
        writer=writer,
        metrics_tracker=metrics_tracker,
    )
    # Replace the real ProgramStateManager with a mock so we can assert on
    # set_program_state calls without touching Redis.
    engine.state = AsyncMock()
    return engine


def _prog(state: ProgramState = ProgramState.DONE) -> Program:
    return Program(code="def solve(): return 42", state=state)


# ---------------------------------------------------------------------------
# _refresh_archive_programs
# ---------------------------------------------------------------------------


class TestRefreshArchivePrograms:
    async def test_only_done_programs_are_transitioned(self) -> None:
        """Programs already QUEUED in the archive are skipped; only DONE ones are re-queued."""
        engine = _make_engine()
        done_prog = _prog(ProgramState.DONE)
        queued_prog = _prog(ProgramState.QUEUED)  # e.g. crash mid-refresh

        engine.strategy.get_program_ids.return_value = [done_prog.id, queued_prog.id]
        engine.storage.mget.return_value = [done_prog, queued_prog]

        count = await engine._refresh_archive_programs()

        assert count == 1
        engine.state.set_program_state.assert_called_once_with(
            done_prog, ProgramState.QUEUED
        )

    async def test_all_done_programs_are_transitioned(self) -> None:
        """When the entire archive is DONE, all programs are re-queued."""
        engine = _make_engine()
        progs = [_prog(ProgramState.DONE) for _ in range(3)]
        engine.strategy.get_program_ids.return_value = [p.id for p in progs]
        engine.storage.mget.return_value = progs

        count = await engine._refresh_archive_programs()

        assert count == 3
        assert engine.state.set_program_state.call_count == 3

    async def test_empty_archive_returns_zero(self) -> None:
        """No archive programs → no transitions, returns 0."""
        engine = _make_engine()
        engine.strategy.get_program_ids.return_value = []

        count = await engine._refresh_archive_programs()

        assert count == 0
        engine.state.set_program_state.assert_not_called()

    async def test_no_done_programs_returns_zero(self) -> None:
        """Archive has programs but none are DONE → returns 0, no transitions."""
        engine = _make_engine()
        running_prog = _prog(ProgramState.RUNNING)
        engine.strategy.get_program_ids.return_value = [running_prog.id]
        engine.storage.mget.return_value = [running_prog]

        count = await engine._refresh_archive_programs()

        assert count == 0
        engine.state.set_program_state.assert_not_called()


# ---------------------------------------------------------------------------
# _ingest_completed_programs
# ---------------------------------------------------------------------------


class TestIngestCompletedPrograms:
    async def test_archive_known_programs_skipped(self) -> None:
        """Programs already in the archive are skipped — strategy.add not called."""
        engine = _make_engine()
        archive_prog = _prog(ProgramState.DONE)
        engine.storage.get_all_by_status.return_value = [archive_prog]
        engine.strategy.get_program_ids.return_value = [archive_prog.id]

        await engine._ingest_completed_programs()

        engine.strategy.add.assert_not_called()
        engine.state.set_program_state.assert_not_called()

    async def test_new_accepted_program_stays_done(self) -> None:
        """A newly accepted program is added to the strategy and stays DONE (no state write)."""
        engine = _make_engine()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True

        new_prog = _prog(ProgramState.DONE)
        engine.storage.get_all_by_status.return_value = [new_prog]
        engine.strategy.get_program_ids.return_value = []

        await engine._ingest_completed_programs()

        engine.strategy.add.assert_called_once_with(new_prog)
        engine.state.set_program_state.assert_not_called()

    async def test_rejected_by_acceptor_is_discarded(self) -> None:
        """Programs rejected by the acceptor are discarded."""
        engine = _make_engine()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = False

        rej_prog = _prog(ProgramState.DONE)
        engine.storage.get_all_by_status.return_value = [rej_prog]
        engine.strategy.get_program_ids.return_value = []

        await engine._ingest_completed_programs()

        engine.strategy.add.assert_not_called()
        engine.state.set_program_state.assert_called_once_with(
            rej_prog, ProgramState.DISCARDED
        )

    async def test_rejected_by_strategy_is_discarded(self) -> None:
        """Programs rejected by strategy.add() are discarded."""
        engine = _make_engine()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = False

        rej_prog = _prog(ProgramState.DONE)
        engine.storage.get_all_by_status.return_value = [rej_prog]
        engine.strategy.get_program_ids.return_value = []

        await engine._ingest_completed_programs()

        engine.state.set_program_state.assert_called_once_with(
            rej_prog, ProgramState.DISCARDED
        )

    async def test_empty_done_set_returns_early(self) -> None:
        """No DONE programs → strategy.get_program_ids never called."""
        engine = _make_engine()
        engine.storage.get_all_by_status.return_value = []

        await engine._ingest_completed_programs()

        engine.strategy.get_program_ids.assert_not_called()

    async def test_mixed_archive_and_new_programs(self) -> None:
        """Archive-known programs are skipped; new programs are evaluated independently."""
        engine = _make_engine()
        engine.config.program_acceptor = MagicMock()
        engine.config.program_acceptor.is_accepted.return_value = True
        engine.strategy.add.return_value = True

        archive_prog = _prog(ProgramState.DONE)
        new_prog = _prog(ProgramState.DONE)

        engine.storage.get_all_by_status.return_value = [archive_prog, new_prog]
        engine.strategy.get_program_ids.return_value = [archive_prog.id]

        await engine._ingest_completed_programs()

        # Only the new program went through strategy.add
        engine.strategy.add.assert_called_once_with(new_prog)
        engine.state.set_program_state.assert_not_called()
