"""Tests for BusedEvolutionEngine — integration with migration bus."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from gigaevo.evolution.bus.engine import BusedEvolutionEngine
from gigaevo.evolution.bus.node import MigrationNode
from gigaevo.evolution.engine.config import EngineConfig
from gigaevo.llm.bandit import MutationOutcome
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


def _mock_node() -> MigrationNode:
    node = MagicMock(spec=MigrationNode)
    node.start = AsyncMock()
    node.stop = AsyncMock()
    node.publish = AsyncMock()
    node.drain_received = MagicMock(return_value=[])
    return node


def _make_engine(migration_node=None, max_imports=10) -> BusedEvolutionEngine:
    storage = AsyncMock()
    strategy = AsyncMock()
    writer = MagicMock()
    writer.bind.return_value = writer
    metrics_tracker = MagicMock()

    if migration_node is None:
        migration_node = _mock_node()

    engine = BusedEvolutionEngine(
        migration_node=migration_node,
        max_imports_per_generation=max_imports,
        storage=storage,
        strategy=strategy,
        mutation_operator=AsyncMock(),
        config=EngineConfig(),
        writer=writer,
        metrics_tracker=metrics_tracker,
    )
    engine.state = AsyncMock()
    return engine


def _valid_prog() -> Program:
    p = Program(code="def solve(): return 42", state=ProgramState.DONE)
    p.add_metrics({"is_valid": 1.0, "fitness": 0.8})
    return p


def _invalid_prog() -> Program:
    p = Program(code="def solve(): return 0", state=ProgramState.DONE)
    p.add_metrics({"is_valid": 0.0, "fitness": 0.0})
    return p


def _migrant_prog() -> Program:
    p = Program(code="def solve(): return 99", state=ProgramState.DONE)
    p.add_metrics({"is_valid": 1.0, "fitness": 0.7})
    p.set_metadata("is_migrant", True)
    p.set_metadata("migration_source_run", "other@db1")
    return p


# ---------------------------------------------------------------------------
# _notify_hook — publish on REJECTED_STRATEGY for valid programs
# ---------------------------------------------------------------------------


class TestNotifyHook:
    async def test_publishes_on_rejected_strategy_valid(self) -> None:
        engine = _make_engine()
        prog = _valid_prog()
        prog.set_metadata("iteration", 5)

        await engine._notify_hook(prog, MutationOutcome.REJECTED_STRATEGY)

        engine._migration_node.publish.assert_called_once_with(prog, 5)

    async def test_no_publish_on_accepted(self) -> None:
        engine = _make_engine()
        await engine._notify_hook(_valid_prog(), MutationOutcome.ACCEPTED)
        engine._migration_node.publish.assert_not_called()

    async def test_no_publish_on_rejected_acceptor(self) -> None:
        engine = _make_engine()
        await engine._notify_hook(_valid_prog(), MutationOutcome.REJECTED_ACCEPTOR)
        engine._migration_node.publish.assert_not_called()

    async def test_no_publish_on_rejected_strategy_invalid(self) -> None:
        engine = _make_engine()
        await engine._notify_hook(_invalid_prog(), MutationOutcome.REJECTED_STRATEGY)
        engine._migration_node.publish.assert_not_called()

    async def test_publish_failure_non_fatal(self) -> None:
        engine = _make_engine()
        engine._migration_node.publish = AsyncMock(side_effect=RuntimeError("fail"))

        # Should not raise
        await engine._notify_hook(_valid_prog(), MutationOutcome.REJECTED_STRATEGY)

    async def test_calls_super_notify_hook(self) -> None:
        engine = _make_engine()
        prog = _valid_prog()

        await engine._notify_hook(prog, MutationOutcome.ACCEPTED)

        engine.mutation_operator.on_program_ingested.assert_called_once()


# ---------------------------------------------------------------------------
# _import_bus_arrivals
# ---------------------------------------------------------------------------


class TestImportBusArrivals:
    async def test_imports_drained_programs(self) -> None:
        engine = _make_engine()
        migrant = _migrant_prog()
        engine._migration_node.drain_received.return_value = [migrant]
        engine.strategy.add = AsyncMock(return_value=True)

        await engine._import_bus_arrivals()

        engine.storage.add.assert_called_once_with(migrant)
        engine.strategy.add.assert_called_once_with(migrant)

    async def test_no_arrivals_is_noop(self) -> None:
        engine = _make_engine()
        engine._migration_node.drain_received.return_value = []

        await engine._import_bus_arrivals()

        engine.storage.add.assert_not_called()
        engine.strategy.add.assert_not_called()

    async def test_strategy_reject_still_stored(self) -> None:
        """Program is stored even if strategy rejects."""
        engine = _make_engine()
        migrant = _migrant_prog()
        engine._migration_node.drain_received.return_value = [migrant]
        engine.strategy.add = AsyncMock(return_value=False)

        await engine._import_bus_arrivals()

        engine.storage.add.assert_called_once()
        engine.strategy.add.assert_called_once()

    async def test_respects_max_imports(self) -> None:
        engine = _make_engine(max_imports=2)
        engine._migration_node.drain_received.return_value = [
            _migrant_prog(),
            _migrant_prog(),
            _migrant_prog(),
        ]

        await engine._import_bus_arrivals()
        engine._migration_node.drain_received.assert_called_with(2)

    async def test_acceptor_check_applied(self) -> None:
        """Programs that fail the acceptor check are not imported."""
        engine = _make_engine()
        migrant = _migrant_prog()
        engine._migration_node.drain_received.return_value = [migrant]
        engine.config.program_acceptor.is_accepted = MagicMock(return_value=False)

        await engine._import_bus_arrivals()

        engine.config.program_acceptor.is_accepted.assert_called_once_with(migrant)
        engine.storage.add.assert_not_called()

    async def test_per_program_exception_isolation(self) -> None:
        """One failing import does not block the rest."""
        engine = _make_engine()
        good = _migrant_prog()
        bad = _migrant_prog()
        engine._migration_node.drain_received.return_value = [bad, good]
        engine.strategy.add = AsyncMock(return_value=True)

        call_count = 0

        async def _add_with_fail(prog):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("storage exploded")

        engine.storage.add = AsyncMock(side_effect=_add_with_fail)

        await engine._import_bus_arrivals()

        # Second program should still have been attempted
        assert engine.storage.add.call_count == 2


# ---------------------------------------------------------------------------
# step — node lifecycle
# ---------------------------------------------------------------------------


class TestStep:
    async def test_step_starts_node_once(self) -> None:
        engine = _make_engine()
        with patch.object(
            BusedEvolutionEngine.__bases__[0], "step", new_callable=AsyncMock
        ):
            await engine.step()
            await engine.step()

        engine._migration_node.start.assert_called_once()
        assert engine._node_started is True

    async def test_step_drains_before_parent_step(self) -> None:
        """Verify drain happens before parent step."""
        engine = _make_engine()
        call_order = []

        engine._migration_node.drain_received.side_effect = lambda n: (
            call_order.append("drain") or []
        )

        async def mock_parent_step():
            call_order.append("parent_step")

        with patch.object(
            BusedEvolutionEngine.__bases__[0], "step", side_effect=mock_parent_step
        ):
            await engine.step()

        assert call_order == ["drain", "parent_step"]


# ---------------------------------------------------------------------------
# stop — cleanup
# ---------------------------------------------------------------------------


class TestStop:
    async def test_stop_stops_node(self) -> None:
        engine = _make_engine()
        engine._node_started = True

        engine.storage.close = AsyncMock()
        engine._metrics_tracker.stop = AsyncMock()

        await engine.stop()

        engine._migration_node.stop.assert_called_once()

    async def test_stop_without_start_skips_node(self) -> None:
        engine = _make_engine()
        assert engine._node_started is False

        engine.storage.close = AsyncMock()
        engine._metrics_tracker.stop = AsyncMock()

        await engine.stop()

        engine._migration_node.stop.assert_not_called()


# ---------------------------------------------------------------------------
# Integration: 3-run exclusive claiming
# ---------------------------------------------------------------------------


class TestThreeRunIntegration:
    async def test_exclusive_import(self) -> None:
        """Simulate 3 engines: a migrant published by engine A is imported by exactly one other."""
        engines = [_make_engine() for _ in range(3)]

        migrant = _migrant_prog()

        # Only engine 1 has the migrant in its drain
        engines[0]._migration_node.drain_received.return_value = []
        engines[1]._migration_node.drain_received.return_value = [migrant]
        engines[2]._migration_node.drain_received.return_value = []

        for e in engines:
            e.strategy.add = AsyncMock(return_value=True)
            await e._import_bus_arrivals()

        total_stored = sum(e.storage.add.call_count for e in engines)
        assert total_stored == 1
        assert engines[1].storage.add.call_count == 1
