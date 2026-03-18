"""Tests for RedisStreamTransport + SETNX exclusivity."""

from __future__ import annotations

import fakeredis.aioredis
import pytest

from gigaevo.evolution.bus.topology import BusTopology, RingTopology
from gigaevo.evolution.bus.transport import MigrantEnvelope, RedisStreamTransport


def _envelope(
    source_run_id: str = "run@db0",
    program_id: str = "aaaa-bbbb",
    generation: int = 5,
) -> MigrantEnvelope:
    return MigrantEnvelope(
        source_run_id=source_run_id,
        program_id=program_id,
        program_data={"code": "def solve(): return 1", "id": program_id},
        published_at=1000.0,
        generation=generation,
    )


@pytest.fixture
def fake_server():
    return fakeredis.aioredis.FakeServer()


@pytest.fixture
def bus_topo():
    return BusTopology()


async def _make_transport(
    fake_server,
    run_id: str = "run@db0",
    stream_key: str = "test:bus",
) -> RedisStreamTransport:
    t = RedisStreamTransport(
        run_id=run_id,
        stream_key=stream_key,
        host="localhost",
        port=6379,
        db=15,
        max_stream_len=100,
        claim_ttl=120,
        block_ms=100,  # Short block for tests
    )
    t._redis = fakeredis.aioredis.FakeRedis(server=fake_server)
    return t


# ---------------------------------------------------------------------------
# MigrantEnvelope serialization round-trip
# ---------------------------------------------------------------------------


class TestMigrantEnvelope:
    def test_roundtrip(self) -> None:
        env = _envelope()
        fields = env.to_stream_fields()
        restored = MigrantEnvelope.from_stream_fields(fields)
        assert restored.source_run_id == env.source_run_id
        assert restored.program_id == env.program_id
        assert restored.generation == env.generation
        assert restored.program_data == env.program_data

    def test_roundtrip_bytes_keys(self) -> None:
        env = _envelope()
        fields = env.to_stream_fields()
        byte_fields = {k.encode(): v.encode() for k, v in fields.items()}
        restored = MigrantEnvelope.from_stream_fields(byte_fields)
        assert restored.source_run_id == env.source_run_id
        assert restored.program_id == env.program_id


# ---------------------------------------------------------------------------
# Publish + Consume
# ---------------------------------------------------------------------------


class TestRedisStreamTransport:
    async def test_publish_and_consume(self, fake_server, bus_topo) -> None:
        pub = await _make_transport(fake_server, run_id="run@db0")
        con = await _make_transport(fake_server, run_id="run@db1")

        env = _envelope(source_run_id="run@db0")
        await pub.publish(env)

        result = await con.consume(
            max_count=10, topology=bus_topo, local_run_id="run@db1"
        )
        assert len(result) == 1
        assert result[0].program_id == env.program_id

    async def test_self_messages_skipped(self, fake_server, bus_topo) -> None:
        t = await _make_transport(fake_server, run_id="run@db0")
        await t.publish(_envelope(source_run_id="run@db0"))
        result = await t.consume(
            max_count=10, topology=bus_topo, local_run_id="run@db0"
        )
        assert len(result) == 0

    async def test_setnx_exclusivity(self, fake_server, bus_topo) -> None:
        """Two consumers: only one claims each message."""
        pub = await _make_transport(fake_server, run_id="pub@db0")
        con_a = await _make_transport(fake_server, run_id="con@db1")
        con_b = await _make_transport(fake_server, run_id="con@db2")

        await pub.publish(_envelope(source_run_id="pub@db0", program_id="prog-1"))

        result_a = await con_a.consume(
            max_count=10, topology=bus_topo, local_run_id="con@db1"
        )
        result_b = await con_b.consume(
            max_count=10, topology=bus_topo, local_run_id="con@db2"
        )

        # Exactly one consumer should have claimed it
        assert len(result_a) + len(result_b) == 1

    async def test_setnx_three_consumers(self, fake_server, bus_topo) -> None:
        """Three consumers competing — exactly one wins per program."""
        pub = await _make_transport(fake_server, run_id="pub@db0")
        consumers = [
            await _make_transport(fake_server, run_id=f"con@db{i}") for i in range(3)
        ]

        for i in range(5):
            await pub.publish(
                _envelope(source_run_id="pub@db0", program_id=f"prog-{i}")
            )

        total_claimed = 0
        for con in consumers:
            result = await con.consume(
                max_count=10, topology=bus_topo, local_run_id=con.run_id
            )
            total_claimed += len(result)

        # Each of 5 programs claimed exactly once
        assert total_claimed == 5

    async def test_max_count_respected(self, fake_server, bus_topo) -> None:
        pub = await _make_transport(fake_server, run_id="pub@db0")
        con = await _make_transport(fake_server, run_id="con@db1")

        for i in range(10):
            await pub.publish(
                _envelope(source_run_id="pub@db0", program_id=f"prog-{i}")
            )

        result = await con.consume(
            max_count=3, topology=bus_topo, local_run_id="con@db1"
        )
        assert len(result) == 3

    async def test_cursor_persistence(self, fake_server, bus_topo) -> None:
        """Cursor save/restore — consumer resumes from last position."""
        pub = await _make_transport(fake_server, run_id="pub@db0")
        con = await _make_transport(fake_server, run_id="con@db1")

        # Publish 3, consume all
        for i in range(3):
            await pub.publish(
                _envelope(source_run_id="pub@db0", program_id=f"batch1-{i}")
            )
        result = await con.consume(
            max_count=10, topology=bus_topo, local_run_id="con@db1"
        )
        assert len(result) == 3

        # Save cursor
        await con.save_cursor()

        # Simulate restart: new transport with same run_id
        con2 = await _make_transport(fake_server, run_id="con@db1")
        await con2.restore_cursor()

        # Publish 2 more
        for i in range(2):
            await pub.publish(
                _envelope(source_run_id="pub@db0", program_id=f"batch2-{i}")
            )

        result2 = await con2.consume(
            max_count=10, topology=bus_topo, local_run_id="con@db1"
        )
        # Should only get the 2 new messages
        assert len(result2) == 2

    async def test_stop_closes_connection(self, fake_server) -> None:
        t = await _make_transport(fake_server, run_id="run@db0")
        await t.stop()
        assert t._redis is None

    async def test_ring_topology_respected_before_claim(self, fake_server) -> None:
        """RingTopology filter runs before SETNX — correct consumer gets the claim."""
        ring = RingTopology(run_ids=["A", "B", "C"])
        pub = await _make_transport(fake_server, run_id="A")
        con_b = await _make_transport(fake_server, run_id="B")
        con_c = await _make_transport(fake_server, run_id="C")

        await pub.publish(_envelope(source_run_id="A", program_id="prog-1"))

        # C should NOT accept from A (predecessor of C is B)
        result_c = await con_c.consume(max_count=10, topology=ring, local_run_id="C")
        assert len(result_c) == 0

        # B SHOULD accept from A (predecessor of B is A)
        result_b = await con_b.consume(max_count=10, topology=ring, local_run_id="B")
        assert len(result_b) == 1

    async def test_non_blocking_on_empty_stream(self, fake_server, bus_topo) -> None:
        """consume() returns empty after block_ms timeout, doesn't hang."""
        con = await _make_transport(fake_server, run_id="con@db1")
        result = await con.consume(
            max_count=10, topology=bus_topo, local_run_id="con@db1"
        )
        assert result == []
