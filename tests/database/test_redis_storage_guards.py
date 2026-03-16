"""Tests for RedisProgramStorage write guards, with_redis, mget, and key patterns.

Covers the most-called untested production code paths:
- _check_write_allowed (9 production callers)
- with_redis (9 production callers)
- mget / _mget_by_keys (10 production callers)
- program_pattern (5 production callers)
- _safe_deserialize (corruption handling)
- _chunks (batching helper)
"""

from __future__ import annotations

import fakeredis.aioredis
import pytest

from gigaevo.database.redis import RedisProgramStorageConfig
from gigaevo.database.redis_program_storage import (
    RedisProgramStorage,
)
from gigaevo.exceptions import StorageError
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def rw_storage():
    """Read-write RedisProgramStorage backed by fakeredis."""
    server = fakeredis.FakeServer()
    config = RedisProgramStorageConfig(
        redis_url="redis://fake:6379/0",
        key_prefix="test",
    )
    storage = RedisProgramStorage(config)
    fake_redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    storage._conn._redis = fake_redis
    storage._conn._closing = False
    yield storage
    await storage.close()


@pytest.fixture
async def ro_storage():
    """Read-only RedisProgramStorage backed by fakeredis."""
    server = fakeredis.FakeServer()
    config = RedisProgramStorageConfig(
        redis_url="redis://fake:6379/0",
        key_prefix="test",
        read_only=True,
    )
    storage = RedisProgramStorage(config)
    fake_redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    storage._conn._redis = fake_redis
    storage._conn._closing = False
    yield storage
    await storage.close()


def _make_prog(code: str = "def solve(): return 42") -> Program:
    return Program(code=code, state=ProgramState.QUEUED, atomic_counter=999_999_999)


# ===================================================================
# _check_write_allowed
# ===================================================================


class TestCheckWriteAllowed:
    """_check_write_allowed guards every write path (9 production callers)."""

    def test_rw_mode_does_not_raise(self, rw_storage: RedisProgramStorage):
        """Write-allowed storage should not raise."""
        rw_storage._check_write_allowed("add")

    def test_ro_mode_raises_storage_error(self, ro_storage: RedisProgramStorage):
        """Read-only storage must raise StorageError with operation name."""
        with pytest.raises(StorageError, match="add"):
            ro_storage._check_write_allowed("add")

    def test_ro_error_message_contains_operation(self, ro_storage: RedisProgramStorage):
        """Error message should mention the blocked operation."""
        with pytest.raises(StorageError, match="update"):
            ro_storage._check_write_allowed("update")

    async def test_ro_blocks_add(self, ro_storage: RedisProgramStorage):
        """add() in read-only mode raises StorageError."""
        with pytest.raises(StorageError, match="add"):
            await ro_storage.add(_make_prog())

    async def test_ro_blocks_update(self, ro_storage: RedisProgramStorage):
        """update() in read-only mode raises StorageError."""
        with pytest.raises(StorageError, match="update"):
            await ro_storage.update(_make_prog())

    async def test_ro_blocks_remove(self, ro_storage: RedisProgramStorage):
        """remove() in read-only mode raises StorageError."""
        with pytest.raises(StorageError, match="remove"):
            await ro_storage.remove("some-id")

    async def test_ro_blocks_write_exclusive(self, ro_storage: RedisProgramStorage):
        """write_exclusive() in read-only mode raises StorageError."""
        with pytest.raises(StorageError, match="write_exclusive"):
            await ro_storage.write_exclusive(_make_prog())

    async def test_ro_blocks_transition_status(self, ro_storage: RedisProgramStorage):
        """transition_status() in read-only mode raises StorageError."""
        with pytest.raises(StorageError, match="transition_status"):
            await ro_storage.transition_status("id", "old", "new")

    async def test_ro_blocks_atomic_state_transition(
        self, ro_storage: RedisProgramStorage
    ):
        """atomic_state_transition() in read-only mode raises StorageError."""
        with pytest.raises(StorageError, match="atomic_state_transition"):
            await ro_storage.atomic_state_transition(_make_prog(), "old", "new")

    async def test_ro_blocks_flushdb(self, ro_storage: RedisProgramStorage):
        """flushdb() in read-only mode raises StorageError."""
        with pytest.raises(StorageError, match="flushdb"):
            await ro_storage.flushdb()

    async def test_ro_allows_reads(self, ro_storage: RedisProgramStorage):
        """Read operations should work in read-only mode."""
        assert await ro_storage.get("nonexistent") is None
        assert await ro_storage.exists("nonexistent") is False
        assert await ro_storage.mget([]) == []
        assert await ro_storage.size() == 0


# ===================================================================
# with_redis
# ===================================================================


class TestWithRedis:
    """with_redis delegates to _conn.execute (9 production callers)."""

    async def test_executes_callback(self, rw_storage: RedisProgramStorage):
        """Callback receives a Redis client and returns result."""
        result = await rw_storage.with_redis("test_op", lambda r: r.ping())
        assert result is True

    async def test_passes_through_return_value(self, rw_storage: RedisProgramStorage):
        """Return value from callback is passed through."""

        async def _set_and_get(r):
            await r.set("mykey", "hello")
            return await r.get("mykey")

        result = await rw_storage.with_redis("test_op", _set_and_get)
        assert result == "hello"

    async def test_wraps_exceptions_in_storage_error(
        self, rw_storage: RedisProgramStorage
    ):
        """Exceptions from callback are wrapped in StorageError."""

        async def _raise(r):
            raise ValueError("boom")

        with pytest.raises(StorageError, match="boom"):
            await rw_storage.with_redis("test_op", _raise)


# ===================================================================
# mget / _mget_by_keys
# ===================================================================


class TestMget:
    """mget and _mget_by_keys (10 production callers)."""

    async def test_empty_list_returns_empty(self, rw_storage: RedisProgramStorage):
        """mget([]) returns [] without hitting Redis."""
        result = await rw_storage.mget([])
        assert result == []

    async def test_single_program(self, rw_storage: RedisProgramStorage):
        """mget with one existing ID returns it."""
        prog = _make_prog()
        await rw_storage.add(prog)
        result = await rw_storage.mget([prog.id])
        assert len(result) == 1
        assert result[0].id == prog.id

    async def test_multiple_programs(self, rw_storage: RedisProgramStorage):
        """mget with multiple IDs returns all."""
        progs = [_make_prog(f"def f{i}(): return {i}") for i in range(5)]
        for p in progs:
            await rw_storage.add(p)

        result = await rw_storage.mget([p.id for p in progs])
        assert len(result) == 5
        assert {r.id for r in result} == {p.id for p in progs}

    async def test_missing_ids_skipped(self, rw_storage: RedisProgramStorage):
        """mget skips IDs that don't exist."""
        prog = _make_prog()
        await rw_storage.add(prog)
        result = await rw_storage.mget([prog.id, "nonexistent-1", "nonexistent-2"])
        assert len(result) == 1
        assert result[0].id == prog.id

    async def test_corrupt_data_skipped(self, rw_storage: RedisProgramStorage):
        """Corrupt JSON in Redis is skipped with a warning, not a crash."""
        prog = _make_prog()
        await rw_storage.add(prog)

        # Corrupt one key directly
        r = await rw_storage._conn.get()
        corrupt_key = rw_storage._keys.program("corrupt-id")
        await r.set(corrupt_key, "not-valid-json{{{")

        result = await rw_storage.mget([prog.id, "corrupt-id"])
        assert len(result) == 1
        assert result[0].id == prog.id


# ===================================================================
# _chunks helper
# ===================================================================


class TestChunks:
    """_chunks batching helper used by _mget_by_keys."""

    def test_empty_input(self):
        result = list(RedisProgramStorage._chunks([], 5))
        assert result == []

    def test_single_chunk(self):
        result = list(RedisProgramStorage._chunks([1, 2, 3], 5))
        assert result == [[1, 2, 3]]

    def test_exact_chunk_boundary(self):
        result = list(RedisProgramStorage._chunks([1, 2, 3, 4], 2))
        assert result == [[1, 2], [3, 4]]

    def test_partial_last_chunk(self):
        result = list(RedisProgramStorage._chunks([1, 2, 3, 4, 5], 3))
        assert result == [[1, 2, 3], [4, 5]]

    def test_chunk_size_one(self):
        result = list(RedisProgramStorage._chunks([1, 2, 3], 1))
        assert result == [[1], [2], [3]]


# ===================================================================
# _safe_deserialize
# ===================================================================


class TestSafeDeserialize:
    """_safe_deserialize handles corrupt data gracefully."""

    def test_valid_json_returns_program(self):
        prog = _make_prog()
        from gigaevo.utils.json import dumps as _dumps

        raw = _dumps(prog.to_dict())
        result = RedisProgramStorage._safe_deserialize(raw, "test")
        assert result is not None
        assert result.id == prog.id

    def test_invalid_json_returns_none(self):
        result = RedisProgramStorage._safe_deserialize("not-json{{{", "test")
        assert result is None

    def test_valid_json_bad_schema_returns_none(self):
        import json

        result = RedisProgramStorage._safe_deserialize(
            json.dumps({"not": "a program"}), "test"
        )
        assert result is None


# ===================================================================
# program_pattern (key generation)
# ===================================================================


class TestProgramPattern:
    """program_pattern generates correct SCAN pattern (5 production callers)."""

    def test_pattern_contains_prefix(self, rw_storage: RedisProgramStorage):
        pattern = rw_storage._keys.program_pattern()
        assert "test" in pattern

    def test_pattern_ends_with_wildcard(self, rw_storage: RedisProgramStorage):
        pattern = rw_storage._keys.program_pattern()
        assert pattern.endswith("*")

    async def test_pattern_matches_stored_programs(
        self, rw_storage: RedisProgramStorage
    ):
        """SCAN with program_pattern finds all stored programs."""
        progs = [_make_prog(f"def f{i}(): return {i}") for i in range(3)]
        for p in progs:
            await rw_storage.add(p)

        r = await rw_storage._conn.get()
        found = []
        async for key in r.scan_iter(
            match=rw_storage._keys.program_pattern(), count=100
        ):
            found.append(key)

        assert len(found) == 3


# ===================================================================
# write_exclusive
# ===================================================================


class TestWriteExclusive:
    """write_exclusive fast-path write (4 production callers)."""

    async def test_basic_write(self, rw_storage: RedisProgramStorage):
        """write_exclusive stores program retrievable via get."""
        prog = _make_prog()
        await rw_storage.add(prog)
        prog.add_metrics({"score": 99.0})
        await rw_storage.write_exclusive(prog)

        fetched = await rw_storage.get(prog.id)
        assert fetched is not None
        assert fetched.metrics["score"] == 99.0

    async def test_overwrites_without_merge(self, rw_storage: RedisProgramStorage):
        """write_exclusive does not merge — last write wins."""
        prog = _make_prog()
        await rw_storage.add(prog)

        prog.state = ProgramState.DONE
        await rw_storage.write_exclusive(prog)

        fetched = await rw_storage.get(prog.id)
        assert fetched.state == ProgramState.DONE
