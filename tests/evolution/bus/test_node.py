"""Tests for MigrationNode — buffer management and orphan conversion."""

from __future__ import annotations

import unittest.mock
from unittest.mock import AsyncMock

from gigaevo.evolution.bus.node import MigrationNode
from gigaevo.evolution.bus.topology import BusTopology
from gigaevo.evolution.bus.transport import MigrantEnvelope
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


def _envelope(
    source_run_id: str = "other@db1",
    program_id: str = "aaaa-bbbb",
    generation: int = 3,
) -> MigrantEnvelope:
    prog = Program(code="def solve(): return 1")
    return MigrantEnvelope(
        source_run_id=source_run_id,
        program_id=prog.id if program_id == "aaaa-bbbb" else program_id,
        program_data=prog.to_dict(),
        published_at=1000.0,
        generation=generation,
    )


def _make_node(
    run_id: str = "local@db0",
    max_buffer_size: int = 50,
) -> tuple[MigrationNode, AsyncMock]:
    transport = AsyncMock()
    transport.consume = AsyncMock(return_value=[])
    topology = BusTopology()

    node = MigrationNode(
        run_id=run_id,
        transport=transport,
        topology=topology,
        max_buffer_size=max_buffer_size,
        consume_interval=0.01,
        max_consume_per_poll=20,
    )
    return node, transport


# ---------------------------------------------------------------------------
# Orphan conversion
# ---------------------------------------------------------------------------


class TestOrphanConversion:
    def test_envelope_to_orphan_basic(self) -> None:
        node, _ = _make_node()
        env = _envelope(source_run_id="src@db1", generation=5)
        orphan = node._envelope_to_orphan(env)

        assert isinstance(orphan, Program)
        assert orphan.state == ProgramState.DONE
        assert orphan.lineage.parents == []
        assert orphan.lineage.children == []
        assert orphan.metadata["is_migrant"] is True
        assert orphan.metadata["migration_source_run"] == "src@db1"
        assert orphan.metadata["migration_generation"] == 5

    def test_orphan_gets_fresh_uuid(self) -> None:
        """Orphan gets a new UUID to avoid cross-run ID collisions."""
        node, _ = _make_node()
        env = _envelope()
        original_id = env.program_id
        orphan = node._envelope_to_orphan(env)
        assert orphan.id != original_id
        assert orphan.metadata["migration_source_id"] == original_id

    def test_orphan_preserves_code(self) -> None:
        node, _ = _make_node()
        env = _envelope()
        orphan = node._envelope_to_orphan(env)
        assert orphan.code == "def solve(): return 1"

    def test_orphan_preserves_metrics(self) -> None:
        node, _ = _make_node()
        prog = Program(code="def solve(): return 1")
        prog.add_metrics({"fitness": 0.9, "is_valid": 1.0})
        env = MigrantEnvelope(
            source_run_id="src@db1",
            program_id=prog.id,
            program_data=prog.to_dict(),
            published_at=1000.0,
            generation=3,
        )
        orphan = node._envelope_to_orphan(env)
        assert orphan.metrics["fitness"] == 0.9
        assert orphan.metrics["is_valid"] == 1.0


# ---------------------------------------------------------------------------
# Buffer draining
# ---------------------------------------------------------------------------


class TestDrainReceived:
    def test_drain_empty(self) -> None:
        node, _ = _make_node()
        assert node.drain_received(10) == []

    def test_drain_partial(self) -> None:
        node, _ = _make_node()
        for i in range(5):
            prog = Program(code=f"def solve(): return {i}")
            env = MigrantEnvelope(
                source_run_id="other@db1",
                program_id=prog.id,
                program_data=prog.to_dict(),
                published_at=1000.0,
                generation=1,
            )
            node._buffer.append(env)

        result = node.drain_received(3)
        assert len(result) == 3
        assert len(node._buffer) == 2

    def test_drain_all(self) -> None:
        node, _ = _make_node()
        prog = Program(code="def solve(): return 1")
        env = MigrantEnvelope(
            source_run_id="other@db1",
            program_id=prog.id,
            program_data=prog.to_dict(),
            published_at=1000.0,
            generation=1,
        )
        node._buffer.append(env)
        result = node.drain_received(100)
        assert len(result) == 1
        assert len(node._buffer) == 0

    def test_drained_programs_are_orphans(self) -> None:
        node, _ = _make_node()
        prog = Program(code="def solve(): return 1")
        env = MigrantEnvelope(
            source_run_id="other@db1",
            program_id=prog.id,
            program_data=prog.to_dict(),
            published_at=1000.0,
            generation=1,
        )
        node._buffer.append(env)
        result = node.drain_received(1)
        assert result[0].state == ProgramState.DONE
        assert result[0].metadata["is_migrant"] is True


# ---------------------------------------------------------------------------
# Buffer overflow
# ---------------------------------------------------------------------------


class TestBufferOverflow:
    async def test_buffer_drops_when_full(self) -> None:
        node, transport = _make_node(max_buffer_size=2)

        envelopes = []
        for i in range(5):
            prog = Program(code=f"def solve(): return {i}")
            envelopes.append(
                MigrantEnvelope(
                    source_run_id="other@db1",
                    program_id=prog.id,
                    program_data=prog.to_dict(),
                    published_at=1000.0,
                    generation=1,
                )
            )

        transport.consume = AsyncMock(return_value=envelopes)

        # Manually run one poll cycle
        node._running = True

        async def _break_sleep(t):
            node._running = False

        with unittest.mock.patch("asyncio.sleep", side_effect=_break_sleep):
            await node._poll_loop()

        assert len(node._buffer) == 2
        assert node._dropped_count == 3


# ---------------------------------------------------------------------------
# Publish
# ---------------------------------------------------------------------------


class TestPublish:
    async def test_publish_calls_transport(self) -> None:
        node, transport = _make_node()
        prog = Program(code="def solve(): return 1")
        await node.publish(prog, generation=7)
        transport.publish.assert_called_once()
        env = transport.publish.call_args[0][0]
        assert env.source_run_id == "local@db0"
        assert env.generation == 7

    async def test_publish_increments_counter(self) -> None:
        node, transport = _make_node()
        prog = Program(code="def solve(): return 1")
        await node.publish(prog, generation=1)
        await node.publish(prog, generation=2)
        assert node._published_count == 2
