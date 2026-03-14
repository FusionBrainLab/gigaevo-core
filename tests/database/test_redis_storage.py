"""RedisProgramStorage CRUD and status operations tests with fakeredis."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis
import numpy as np
import pytest

from gigaevo.database.redis import RedisProgramStorageConfig
from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.exceptions import StorageError
from gigaevo.programs.core_types import (
    ProgramStageResult,
    StageError,
    StageState,
)
from gigaevo.programs.program import Lineage, Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.utils import pickle_b64_deserialize, pickle_b64_serialize
from tests.conftest import MockOutput

# ===================================================================
# Category A: Basic CRUD
# ===================================================================


class TestBasicCRUD:
    async def test_add_and_get(self, fakeredis_storage, make_program):
        """Add program, get by ID, verify equality."""
        prog = make_program()
        await fakeredis_storage.add(prog)

        fetched = await fakeredis_storage.get(prog.id)
        assert fetched is not None
        assert fetched.id == prog.id
        assert fetched.code == prog.code

    async def test_get_nonexistent_returns_none(self, fakeredis_storage):
        """Unknown ID returns None."""
        fetched = await fakeredis_storage.get("nonexistent-id")
        assert fetched is None

    async def test_exists_true_and_false(self, fakeredis_storage, make_program):
        """exists() returns correct bool."""
        prog = make_program()
        assert await fakeredis_storage.exists(prog.id) is False

        await fakeredis_storage.add(prog)
        assert await fakeredis_storage.exists(prog.id) is True

    async def test_remove_program(self, fakeredis_storage, make_program):
        """Remove deletes from Redis; get returns None."""
        prog = make_program()
        await fakeredis_storage.add(prog)

        await fakeredis_storage.remove(prog.id)
        assert await fakeredis_storage.get(prog.id) is None

    async def test_update_preserves_identity(self, fakeredis_storage, make_program):
        """update() keeps id, created_at; merges metrics."""
        prog = make_program()
        await fakeredis_storage.add(prog)
        original_created = prog.created_at

        prog.add_metrics({"score": 10.0})
        await fakeredis_storage.update(prog)

        fetched = await fakeredis_storage.get(prog.id)
        assert fetched.id == prog.id
        assert fetched.created_at == original_created
        assert fetched.metrics["score"] == 10.0


# ===================================================================
# Category B: Batch Operations
# ===================================================================


class TestBatchOperations:
    async def test_mget_returns_all(self, fakeredis_storage, make_program):
        """mget with multiple IDs returns all programs."""
        progs = [make_program() for _ in range(3)]
        for p in progs:
            await fakeredis_storage.add(p)

        ids = [p.id for p in progs]
        fetched = await fakeredis_storage.mget(ids)
        assert len(fetched) == 3
        fetched_ids = {p.id for p in fetched}
        assert fetched_ids == set(ids)

    async def test_mget_empty_list(self, fakeredis_storage):
        """mget([]) returns []."""
        result = await fakeredis_storage.mget([])
        assert result == []

    async def test_mget_with_missing_ids(self, fakeredis_storage, make_program):
        """mget with some invalid IDs returns only found programs."""
        prog = make_program()
        await fakeredis_storage.add(prog)

        fetched = await fakeredis_storage.mget(
            [prog.id, "nonexistent-1", "nonexistent-2"]
        )
        assert len(fetched) == 1
        assert fetched[0].id == prog.id


# ===================================================================
# Category C: Status Operations
# ===================================================================


class TestStatusOperations:
    async def test_add_sets_status_set(self, fakeredis_storage, make_program):
        """After add, program ID appears in status set."""
        prog = make_program(state=ProgramState.RUNNING)
        await fakeredis_storage.add(prog)

        count = await fakeredis_storage.count_by_status(ProgramState.RUNNING.value)
        assert count >= 1

    async def test_count_by_status(self, fakeredis_storage, make_program):
        """count_by_status returns correct count."""
        for _ in range(3):
            await fakeredis_storage.add(make_program(state=ProgramState.QUEUED))

        count = await fakeredis_storage.count_by_status(ProgramState.QUEUED.value)
        assert count == 3

    async def test_get_all_by_status(self, fakeredis_storage, make_program):
        """Returns only programs matching status."""
        queued_prog = make_program(state=ProgramState.QUEUED)
        running_prog = make_program(state=ProgramState.RUNNING)
        await fakeredis_storage.add(queued_prog)
        await fakeredis_storage.add(running_prog)

        queued_list = await fakeredis_storage.get_all_by_status(
            ProgramState.QUEUED.value
        )
        assert len(queued_list) == 1
        assert queued_list[0].id == queued_prog.id

    async def test_transition_status(self, fakeredis_storage, make_program):
        """Moves ID between status sets."""
        prog = make_program(state=ProgramState.QUEUED)
        await fakeredis_storage.add(prog)

        await fakeredis_storage.transition_status(
            prog.id,
            ProgramState.QUEUED.value,
            ProgramState.RUNNING.value,
        )

        old_count = await fakeredis_storage.count_by_status(ProgramState.QUEUED.value)
        new_count = await fakeredis_storage.count_by_status(ProgramState.RUNNING.value)
        assert old_count == 0
        assert new_count == 1

    async def test_atomic_state_transition(self, fakeredis_storage, make_program):
        """Full program + status sets updated atomically."""
        prog = make_program(state=ProgramState.QUEUED)
        await fakeredis_storage.add(prog)

        prog.state = ProgramState.RUNNING
        await fakeredis_storage.atomic_state_transition(
            prog,
            ProgramState.QUEUED.value,
            ProgramState.RUNNING.value,
        )

        fetched = await fakeredis_storage.get(prog.id)
        assert fetched.state == ProgramState.RUNNING

        old_count = await fakeredis_storage.count_by_status(ProgramState.QUEUED.value)
        new_count = await fakeredis_storage.count_by_status(ProgramState.RUNNING.value)
        assert old_count == 0
        assert new_count == 1


# ===================================================================
# Category D: Serialization Round-Trip
# ===================================================================


class TestSerializationRoundTrip:
    async def test_program_with_metrics_roundtrip(
        self, fakeredis_storage, make_program
    ):
        """Metrics survive add -> get."""
        prog = make_program(metrics={"acc": 0.95, "loss": 0.05})
        await fakeredis_storage.add(prog)

        fetched = await fakeredis_storage.get(prog.id)
        assert fetched.metrics["acc"] == 0.95
        assert fetched.metrics["loss"] == 0.05

    async def test_program_with_stage_results_roundtrip(
        self, fakeredis_storage, make_program
    ):
        """ProgramStageResult + output survives."""
        output = MockOutput(value=123)
        result = ProgramStageResult.success(output=output)
        prog = make_program(stage_results={"validation": result})
        await fakeredis_storage.add(prog)

        fetched = await fakeredis_storage.get(prog.id)
        fetched_res = fetched.stage_results["validation"]
        assert fetched_res.status == StageState.COMPLETED
        assert fetched_res.output.value == 123

    async def test_program_with_metadata_roundtrip(
        self, fakeredis_storage, make_program
    ):
        """Arbitrary metadata (dicts, nested) survives."""
        metadata = {
            "experiment": "test-1",
            "config": {"lr": 0.01, "layers": [64, 32]},
        }
        prog = make_program(metadata=metadata)
        await fakeredis_storage.add(prog)

        fetched = await fakeredis_storage.get(prog.id)
        assert fetched.metadata["experiment"] == "test-1"
        assert fetched.metadata["config"]["lr"] == 0.01
        assert fetched.metadata["config"]["layers"] == [64, 32]


# ===================================================================
# Category E: Merge Strategy
# ===================================================================


class TestMergeStrategy:
    async def test_update_merges_metrics(self, fakeredis_storage, make_program):
        """Update merges metrics (latest wins via atomic_counter)."""
        prog = make_program(metrics={"a": 1.0})
        await fakeredis_storage.add(prog)

        prog.add_metrics({"b": 2.0})
        await fakeredis_storage.update(prog)

        fetched = await fakeredis_storage.get(prog.id)
        assert "a" in fetched.metrics
        assert "b" in fetched.metrics

    async def test_update_merges_stage_results(self, fakeredis_storage, make_program):
        """Stage results from both sides preserved."""
        res_a = ProgramStageResult.success(output=MockOutput(value=1))
        prog = make_program(stage_results={"stage_a": res_a})
        await fakeredis_storage.add(prog)

        res_b = ProgramStageResult.success(output=MockOutput(value=2))
        prog.stage_results["stage_b"] = res_b
        await fakeredis_storage.update(prog)

        fetched = await fakeredis_storage.get(prog.id)
        assert "stage_a" in fetched.stage_results
        assert "stage_b" in fetched.stage_results


# ===================================================================
# Category F: write_exclusive
# ===================================================================


class TestWriteExclusive:
    async def test_write_exclusive_persists_correctly(
        self, fakeredis_storage, make_program
    ):
        """write_exclusive saves program data including stage_results."""
        output = MockOutput(value=55)
        result = ProgramStageResult.success(output=output)
        prog = make_program(metrics={"score": 7.5})
        await fakeredis_storage.add(prog)

        prog.stage_results["stage_x"] = result
        prog.add_metrics({"new_metric": 3.14})
        await fakeredis_storage.write_exclusive(prog)

        fetched = await fakeredis_storage.get(prog.id)
        assert fetched is not None
        assert fetched.stage_results["stage_x"].status == StageState.COMPLETED
        assert fetched.stage_results["stage_x"].output.value == 55
        assert fetched.metrics["new_metric"] == 3.14

    async def test_write_exclusive_updates_atomic_counter(
        self, fakeredis_storage, make_program
    ):
        """write_exclusive increments the atomic counter on each write."""
        prog = make_program()
        await fakeredis_storage.add(prog)

        first = await fakeredis_storage.get(prog.id)
        counter_after_add = first.atomic_counter

        await fakeredis_storage.write_exclusive(prog)
        second = await fakeredis_storage.get(prog.id)
        assert second.atomic_counter > counter_after_add

    async def test_write_exclusive_overwrites_redis(
        self, fakeredis_storage, make_program
    ):
        """write_exclusive replaces existing data in Redis (no merge)."""
        prog = make_program(metrics={"a": 1.0})
        await fakeredis_storage.add(prog)

        # Simulate: remote write adds metric "b" (concurrent; won't be seen locally)
        prog2 = make_program(metrics={"a": 1.0})
        prog2.id = prog.id  # same program
        prog2.add_metrics({"b": 2.0})
        # write_exclusive does NOT merge — it writes the local in-memory state
        prog.add_metrics({"c": 3.0})
        await fakeredis_storage.write_exclusive(prog)

        fetched = await fakeredis_storage.get(prog.id)
        assert fetched.metrics["a"] == 1.0
        assert fetched.metrics["c"] == 3.0


# ===================================================================
# Category G: Edge Cases
# ===================================================================


class TestEdgeCases:
    async def test_size_counts_programs(self, fakeredis_storage, make_program):
        """size() returns correct count after adds/removes."""
        assert await fakeredis_storage.size() == 0

        p1 = make_program()
        p2 = make_program()
        await fakeredis_storage.add(p1)
        await fakeredis_storage.add(p2)
        assert await fakeredis_storage.size() == 2

        await fakeredis_storage.remove(p1.id)
        assert await fakeredis_storage.size() == 1

    async def test_has_data(self, fakeredis_storage, make_program):
        """has_data() returns True/False correctly."""
        assert await fakeredis_storage.has_data() is False

        prog = make_program()
        await fakeredis_storage.add(prog)
        assert await fakeredis_storage.has_data() is True


# ===================================================================
# Category H: Serialization Edge Cases (pickle_b64 roundtrips)
# ===================================================================


class TestPickleB64EdgeCases:
    """Test pickle_b64_serialize/deserialize with edge cases."""

    def test_none_roundtrip(self):
        s = pickle_b64_serialize(None)
        assert pickle_b64_deserialize(s) is None

    def test_empty_dict_roundtrip(self):
        s = pickle_b64_serialize({})
        assert pickle_b64_deserialize(s) == {}

    def test_empty_list_roundtrip(self):
        s = pickle_b64_serialize([])
        assert pickle_b64_deserialize(s) == []

    def test_nested_complex_structure(self):
        value = {
            "a": [1, 2.5, None, True, False],
            "b": {"nested": {"deep": [{"key": "val"}]}},
            "c": (1, 2, 3),
            "d": set(),
        }
        s = pickle_b64_serialize(value)
        result = pickle_b64_deserialize(s)
        assert result["a"] == [1, 2.5, None, True, False]
        assert result["b"]["nested"]["deep"] == [{"key": "val"}]
        assert result["c"] == (1, 2, 3)
        assert result["d"] == set()

    def test_lambda_roundtrip(self):
        """cloudpickle can serialize lambdas."""
        fn = lambda x: x * 2  # noqa: E731
        s = pickle_b64_serialize(fn)
        restored = pickle_b64_deserialize(s)
        assert restored(5) == 10

    def test_numpy_array_roundtrip(self):
        arr = np.array([1.0, 2.0, 3.0])
        s = pickle_b64_serialize(arr)
        restored = pickle_b64_deserialize(s)
        assert np.array_equal(restored, arr)

    def test_datetime_roundtrip(self):
        dt = datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc)
        s = pickle_b64_serialize(dt)
        assert pickle_b64_deserialize(s) == dt

    def test_corrupt_base64_raises(self):
        import pytest

        with pytest.raises(Exception):
            pickle_b64_deserialize("not-valid-base64!!!")

    def test_corrupt_pickle_raises(self):
        import base64

        import pytest

        bad = base64.b64encode(b"not a pickle").decode("utf-8")
        with pytest.raises(Exception):
            pickle_b64_deserialize(bad)


class TestSerializationRoundTripEdgeCases:
    """Advanced roundtrip tests through full Redis storage pipeline."""

    async def test_empty_metadata_roundtrip(self, fakeredis_storage, make_program):
        """Empty metadata dict survives storage roundtrip."""
        prog = make_program(metadata={})
        await fakeredis_storage.add(prog)

        fetched = await fakeredis_storage.get(prog.id)
        assert fetched.metadata == {}

    async def test_metadata_with_none_values(self, fakeredis_storage, make_program):
        prog = make_program(metadata={"key": None, "nested": {"inner": None}})
        await fakeredis_storage.add(prog)

        fetched = await fakeredis_storage.get(prog.id)
        assert fetched.metadata["key"] is None
        assert fetched.metadata["nested"]["inner"] is None

    async def test_stage_result_with_error_roundtrip(
        self, fakeredis_storage, make_program
    ):
        """StageError in ProgramStageResult survives roundtrip."""
        error = StageError(
            type="RuntimeError",
            message="something broke",
            stage="TestStage",
            traceback="Traceback...\n  line 42",
        )
        result = ProgramStageResult(status=StageState.FAILED, error=error)
        prog = make_program(stage_results={"broken_stage": result})
        await fakeredis_storage.add(prog)

        fetched = await fakeredis_storage.get(prog.id)
        fetched_res = fetched.stage_results["broken_stage"]
        assert fetched_res.status == StageState.FAILED
        assert fetched_res.error.type == "RuntimeError"
        assert fetched_res.error.message == "something broke"
        assert "line 42" in fetched_res.error.traceback

    async def test_multiple_stage_results_roundtrip(
        self, fakeredis_storage, make_program
    ):
        """Multiple stage results with different statuses survive."""
        results = {
            "stage_a": ProgramStageResult.success(output=MockOutput(value=1)),
            "stage_b": ProgramStageResult(status=StageState.FAILED),
            "stage_c": ProgramStageResult(status=StageState.PENDING),
        }
        prog = make_program(stage_results=results)
        await fakeredis_storage.add(prog)

        fetched = await fakeredis_storage.get(prog.id)
        assert fetched.stage_results["stage_a"].status == StageState.COMPLETED
        assert fetched.stage_results["stage_a"].output.value == 1
        assert fetched.stage_results["stage_b"].status == StageState.FAILED
        assert fetched.stage_results["stage_c"].status == StageState.PENDING

    async def test_safe_deserialize_returns_none_on_corrupt(self, fakeredis_storage):
        """_safe_deserialize returns None for corrupt data instead of crashing."""
        from gigaevo.database.redis_program_storage import RedisProgramStorage

        result = RedisProgramStorage._safe_deserialize("not json at all{{{", "test")
        assert result is None

    async def test_large_metrics_dict_roundtrip(self, fakeredis_storage, make_program):
        """A large metrics dict (100 keys) survives roundtrip."""
        metrics = {f"metric_{i}": float(i) for i in range(100)}
        prog = make_program(metrics=metrics)
        await fakeredis_storage.add(prog)

        fetched = await fakeredis_storage.get(prog.id)
        for i in range(100):
            assert fetched.metrics[f"metric_{i}"] == float(i)


# ===================================================================
# Helper: create a read-only storage backed by fakeredis
# ===================================================================


def _make_read_only_storage() -> RedisProgramStorage:
    """Create a RedisProgramStorage in read-only mode with a fakeredis backend."""
    server = fakeredis.FakeServer()
    config = RedisProgramStorageConfig(
        redis_url="redis://fake:6379/0",
        key_prefix="test_ro",
        read_only=True,
    )
    storage = RedisProgramStorage(config)
    fake_redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    storage._conn._redis = fake_redis
    storage._conn._closing = False
    return storage


# ===================================================================
# Category I: Read-Only Mode
# ===================================================================


class TestReadOnlyMode:
    """Verify write operations raise StorageError in read-only mode."""

    async def test_add_raises_in_read_only(self, make_program):
        """add() raises StorageError in read-only mode."""
        storage = _make_read_only_storage()
        try:
            prog = make_program()
            with pytest.raises(StorageError, match="read-only"):
                await storage.add(prog)
        finally:
            await storage.close()

    async def test_update_raises_in_read_only(self, make_program):
        """update() raises StorageError in read-only mode."""
        storage = _make_read_only_storage()
        try:
            prog = make_program()
            with pytest.raises(StorageError, match="read-only"):
                await storage.update(prog)
        finally:
            await storage.close()

    async def test_remove_raises_in_read_only(self):
        """remove() raises StorageError in read-only mode."""
        storage = _make_read_only_storage()
        try:
            with pytest.raises(StorageError, match="read-only"):
                await storage.remove("some-id")
        finally:
            await storage.close()

    async def test_flushdb_raises_in_read_only(self):
        """flushdb() raises StorageError in read-only mode."""
        storage = _make_read_only_storage()
        try:
            with pytest.raises(StorageError, match="read-only"):
                await storage.flushdb()
        finally:
            await storage.close()

    async def test_get_works_in_read_only(self, make_program):
        """get() succeeds in read-only mode (returns None for missing ID)."""
        storage = _make_read_only_storage()
        try:
            result = await storage.get("nonexistent-id")
            assert result is None
        finally:
            await storage.close()

    async def test_write_exclusive_raises_in_read_only(self, make_program):
        """write_exclusive() raises StorageError in read-only mode."""
        storage = _make_read_only_storage()
        try:
            prog = make_program()
            with pytest.raises(StorageError, match="read-only"):
                await storage.write_exclusive(prog)
        finally:
            await storage.close()

    async def test_transition_status_raises_in_read_only(self):
        """transition_status() raises StorageError in read-only mode."""
        storage = _make_read_only_storage()
        try:
            with pytest.raises(StorageError, match="read-only"):
                await storage.transition_status("some-id", "QUEUED", "RUNNING")
        finally:
            await storage.close()


# ===================================================================
# Category J: Context Manager
# ===================================================================


class TestContextManager:
    """Verify __aenter__/__aexit__ behavior."""

    async def test_context_manager_acquires_lock_and_starts_metrics(self, make_program):
        """Normal (non read-only) context manager acquires lock and starts metrics."""
        server = fakeredis.FakeServer()
        config = RedisProgramStorageConfig(
            redis_url="redis://fake:6379/0",
            key_prefix="test_ctx",
            read_only=False,
        )
        storage = RedisProgramStorage(config)
        fake_redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
        storage._conn._redis = fake_redis
        storage._conn._closing = False

        # Patch lock.acquire to avoid real locking; start() is synchronous
        storage._lock.acquire = AsyncMock(return_value=True)
        storage._metrics.start = MagicMock()

        async with storage as s:
            assert s is storage
            storage._lock.acquire.assert_awaited_once()
            storage._metrics.start.assert_called_once()

    async def test_context_manager_read_only_skips_lock(self):
        """Read-only context manager skips lock acquisition."""
        storage = _make_read_only_storage()
        storage._lock.acquire = AsyncMock(return_value=True)

        async with storage as s:
            assert s is storage
            # Lock should NOT have been acquired in read-only mode
            storage._lock.acquire.assert_not_awaited()


# ===================================================================
# Category K: WatchError Retries
# ===================================================================


class TestWatchErrorRetries:
    """Verify update and atomic_state_transition retry on WatchError."""

    async def test_update_retries_on_watch_error(self, fakeredis_storage, make_program):
        """update() retries when WatchError is raised, then succeeds."""
        from redis.exceptions import WatchError

        prog = make_program()
        await fakeredis_storage.add(prog)
        prog.add_metrics({"retried": 1.0})

        call_count = 0
        original_execute = fakeredis_storage._conn.execute

        async def patched_execute(name, fn):
            nonlocal call_count
            if name == "update":
                # Wrap fn so first call raises WatchError inside the pipeline
                real_fn = fn

                async def watch_error_fn(r):
                    nonlocal call_count
                    call_count += 1
                    if call_count == 1:
                        raise WatchError("simulated conflict")
                    return await real_fn(r)

                return await original_execute(name, watch_error_fn)
            return await original_execute(name, fn)

        fakeredis_storage._conn.execute = patched_execute

        await fakeredis_storage.update(prog)
        fetched = await fakeredis_storage.get(prog.id)
        assert fetched.metrics["retried"] == 1.0
        assert call_count >= 2

    async def test_atomic_state_transition_retries_on_watch_error(
        self, fakeredis_storage, make_program
    ):
        """atomic_state_transition() retries when WatchError is raised."""
        from redis.exceptions import WatchError

        prog = make_program(state=ProgramState.QUEUED)
        await fakeredis_storage.add(prog)

        prog.state = ProgramState.RUNNING
        call_count = 0
        original_execute = fakeredis_storage._conn.execute

        async def patched_execute(name, fn):
            nonlocal call_count
            if name == "atomic_state_transition":
                real_fn = fn

                async def watch_error_fn(r):
                    nonlocal call_count
                    call_count += 1
                    if call_count == 1:
                        raise WatchError("simulated conflict")
                    return await real_fn(r)

                return await original_execute(name, watch_error_fn)
            return await original_execute(name, fn)

        fakeredis_storage._conn.execute = patched_execute

        await fakeredis_storage.atomic_state_transition(
            prog, ProgramState.QUEUED.value, ProgramState.RUNNING.value
        )
        fetched = await fakeredis_storage.get(prog.id)
        assert fetched.state == ProgramState.RUNNING
        assert call_count >= 2


# ===================================================================
# Category L: Stream Operations
# ===================================================================


class TestStreamOperations:
    """Verify publish_status_event, wait_for_activity, and fallback."""

    async def test_publish_status_event(self, fakeredis_storage, make_program):
        """publish_status_event writes to the status stream with correct fields."""
        prog = make_program()
        await fakeredis_storage.add(prog)

        await fakeredis_storage.publish_status_event(
            status="DONE",
            program_id=prog.id,
            extra={"event": "completed"},
        )

        # Read back from the stream to verify
        r = await fakeredis_storage._conn.get()
        stream_key = fakeredis_storage._keys.status_stream()
        entries = await r.xrange(stream_key)
        # At least 2 entries: one from add(), one from publish_status_event()
        assert len(entries) >= 2
        last_entry = entries[-1]
        assert last_entry[1]["id"] == prog.id
        assert last_entry[1]["status"] == "DONE"
        assert last_entry[1]["event"] == "completed"

        # Verify earlier entries have the expected structure
        first_entry = entries[0]
        assert "id" in first_entry[1]
        assert "status" in first_entry[1]

    async def test_wait_for_activity_returns_on_timeout(self, fakeredis_storage):
        """wait_for_activity returns after the timeout with no events."""
        # Should not hang; returns after a short timeout
        await fakeredis_storage.wait_for_activity(timeout=0.05)

    async def test_wait_for_activity_fallback_on_error(self, fakeredis_storage):
        """wait_for_activity falls back to asyncio.sleep on stream error."""
        r = await fakeredis_storage._conn.get()

        original_xread = r.xread

        async def broken_xread(*args, **kwargs):
            raise ConnectionError("simulated stream error")

        r.xread = broken_xread

        start = asyncio.get_event_loop().time()
        await fakeredis_storage.wait_for_activity(timeout=0.05)
        elapsed = asyncio.get_event_loop().time() - start
        # Should have fallen back to asyncio.sleep(0.05)
        assert elapsed >= 0.04

        # Restore original
        r.xread = original_xread


# ===================================================================
# Category M: Audit Finding 1 — Full Program round-trip (all fields)
# ===================================================================


class TestFullProgramRoundTrip:
    """Audit finding 1: round-trip test must check ALL Program fields, not just id and code."""

    async def test_all_fields_survive_redis_roundtrip(
        self, fakeredis_storage, make_program
    ):
        """Create a Program with every field populated, store it, retrieve it,
        and verify every single field matches."""
        # Build a fully-populated program
        stage_result_ok = ProgramStageResult.success(output=MockOutput(value=77))
        error = StageError(
            type="ValueError",
            message="some error",
            stage="ErrStage",
            traceback="Traceback...\n  line 99",
        )
        stage_result_fail = ProgramStageResult(status=StageState.FAILED, error=error)

        lineage = Lineage(
            parents=["00000000-0000-0000-0000-000000000001"],
            children=["00000000-0000-0000-0000-000000000002"],
            mutation="test_mutation_op",
            generation=5,
        )
        metadata = {
            "experiment": "full-roundtrip",
            "config": {"lr": 0.001, "epochs": 100},
            "tags": ["alpha", "beta"],
        }
        metrics = {"accuracy": 0.97, "loss": 0.03, "f1": 0.95}

        prog = Program(
            code="def solve(x): return x * 2",
            state=ProgramState.DONE,
            name="roundtrip-test-prog",
            lineage=lineage,
            metrics=metrics,
            metadata=metadata,
            stage_results={
                "validation": stage_result_ok,
                "optimization": stage_result_fail,
            },
            atomic_counter=999_999_999,
        )

        await fakeredis_storage.add(prog)
        fetched = await fakeredis_storage.get(prog.id)

        assert fetched is not None

        # 1. id
        assert fetched.id == prog.id
        # 2. code
        assert fetched.code == prog.code
        # 3. name
        assert fetched.name == prog.name
        # 4. state
        assert fetched.state == prog.state
        # 5. metrics (all keys and values)
        assert fetched.metrics == prog.metrics
        # 6. metadata (nested structure)
        assert fetched.metadata["experiment"] == "full-roundtrip"
        assert fetched.metadata["config"]["lr"] == 0.001
        assert fetched.metadata["config"]["epochs"] == 100
        assert fetched.metadata["tags"] == ["alpha", "beta"]
        # 7. lineage (parents, children, mutation, generation)
        assert fetched.lineage.parents == lineage.parents
        assert fetched.lineage.children == lineage.children
        assert fetched.lineage.mutation == lineage.mutation
        assert fetched.lineage.generation == lineage.generation
        # 8. stage_results (success + failure)
        assert "validation" in fetched.stage_results
        assert fetched.stage_results["validation"].status == StageState.COMPLETED
        assert fetched.stage_results["validation"].output.value == 77
        assert "optimization" in fetched.stage_results
        assert fetched.stage_results["optimization"].status == StageState.FAILED
        assert fetched.stage_results["optimization"].error.type == "ValueError"
        assert fetched.stage_results["optimization"].error.message == "some error"
        assert "line 99" in fetched.stage_results["optimization"].error.traceback
        # 9. created_at preserved
        assert fetched.created_at == prog.created_at
        # 10. generation property delegates to lineage
        assert fetched.generation == 5


# ===================================================================
# Category N: Audit Finding 2 — Concurrent writers on SAME stage key
# ===================================================================


class TestConcurrentSameKeyUpdate:
    """Audit finding 2: concurrent updates to the SAME stage_result key."""

    async def test_concurrent_writers_same_stage_key_consistent(
        self, fakeredis_storage, make_program
    ):
        """Multiple concurrent writers updating the SAME stage_result key.
        After all writes complete, the final state must be one of the written values
        (not corrupted), and the program must be retrievable."""
        prog = make_program()
        await fakeredis_storage.add(prog)

        async def write_stage_result(value: int):
            """Update the same stage key with a specific value."""
            result = ProgramStageResult.success(output=MockOutput(value=value))
            prog.stage_results["shared_stage"] = result
            await fakeredis_storage.update(prog)

        # Run 5 concurrent writers all targeting "shared_stage"
        values = list(range(5))
        await asyncio.gather(*(write_stage_result(v) for v in values))

        fetched = await fakeredis_storage.get(prog.id)
        assert fetched is not None
        # The final value must be one of the values we wrote (not corrupted)
        assert "shared_stage" in fetched.stage_results
        assert fetched.stage_results["shared_stage"].status == StageState.COMPLETED
        final_value = fetched.stage_results["shared_stage"].output.value
        assert final_value in values, (
            f"Final value {final_value} is not one of the expected values {values}"
        )


# ===================================================================
# Category O: Audit Finding 3 — State transitions persisted to Redis
# ===================================================================


class TestStateTransitionPersistence:
    """Audit finding 3: state transitions must be readable after re-fetch from Redis."""

    async def test_transition_status_persisted_and_refetched(
        self, fakeredis_storage, make_program
    ):
        """transition_status moves program between status sets;
        after re-fetch, the status set counts are correct."""
        prog = make_program(state=ProgramState.QUEUED)
        await fakeredis_storage.add(prog)

        # Transition QUEUED -> RUNNING at the status-set level
        await fakeredis_storage.transition_status(
            prog.id, ProgramState.QUEUED.value, ProgramState.RUNNING.value
        )

        # Re-fetch from Redis and verify status set membership
        queued_ids = await fakeredis_storage.get_ids_by_status(
            ProgramState.QUEUED.value
        )
        running_ids = await fakeredis_storage.get_ids_by_status(
            ProgramState.RUNNING.value
        )
        assert prog.id not in queued_ids
        assert prog.id in running_ids

    async def test_atomic_state_transition_persisted_and_refetched(
        self, fakeredis_storage, make_program
    ):
        """atomic_state_transition updates both the program data and status sets;
        after re-fetch, the program has the new state."""
        prog = make_program(state=ProgramState.QUEUED)
        await fakeredis_storage.add(prog)

        prog.state = ProgramState.RUNNING
        await fakeredis_storage.atomic_state_transition(
            prog, ProgramState.QUEUED.value, ProgramState.RUNNING.value
        )

        # Re-fetch the actual program from Redis
        fetched = await fakeredis_storage.get(prog.id)
        assert fetched is not None
        assert fetched.state == ProgramState.RUNNING

        # Also verify status set membership via get_all_by_status
        running_progs = await fakeredis_storage.get_all_by_status(
            ProgramState.RUNNING.value
        )
        assert any(p.id == prog.id for p in running_progs)

        queued_progs = await fakeredis_storage.get_all_by_status(
            ProgramState.QUEUED.value
        )
        assert not any(p.id == prog.id for p in queued_progs)

    async def test_full_lifecycle_state_persisted_each_step(
        self, fakeredis_storage, make_program
    ):
        """QUEUED -> RUNNING -> DONE: each step is persisted and re-fetchable."""
        prog = make_program(state=ProgramState.QUEUED)
        await fakeredis_storage.add(prog)

        # Step 1: QUEUED -> RUNNING
        prog.state = ProgramState.RUNNING
        await fakeredis_storage.atomic_state_transition(
            prog, ProgramState.QUEUED.value, ProgramState.RUNNING.value
        )
        fetched = await fakeredis_storage.get(prog.id)
        assert fetched.state == ProgramState.RUNNING

        # Step 2: RUNNING -> DONE
        prog.state = ProgramState.DONE
        await fakeredis_storage.atomic_state_transition(
            prog, ProgramState.RUNNING.value, ProgramState.DONE.value
        )
        fetched = await fakeredis_storage.get(prog.id)
        assert fetched.state == ProgramState.DONE

        # Verify status set cleanup across the lifecycle
        queued_count = await fakeredis_storage.count_by_status(
            ProgramState.QUEUED.value
        )
        running_count = await fakeredis_storage.count_by_status(
            ProgramState.RUNNING.value
        )
        done_count = await fakeredis_storage.count_by_status(ProgramState.DONE.value)
        assert queued_count == 0
        assert running_count == 0
        assert done_count == 1


# ===================================================================
# Category P: Audit Finding 4 — remove() cleans up status sets
# ===================================================================


class TestRemoveStatusSetCleanup:
    """Audit finding 4: remove() must clean up status set membership."""

    async def test_remove_cleans_up_queued_status_set(
        self, fakeredis_storage, make_program
    ):
        """After remove(), get_all_by_status should NOT return the removed program."""
        prog = make_program(state=ProgramState.QUEUED)
        await fakeredis_storage.add(prog)

        # Verify it's in the QUEUED set
        queued_before = await fakeredis_storage.get_all_by_status(
            ProgramState.QUEUED.value
        )
        assert any(p.id == prog.id for p in queued_before)

        # Remove the program
        await fakeredis_storage.remove(prog.id)

        # Verify status set is cleaned up
        queued_after = await fakeredis_storage.get_all_by_status(
            ProgramState.QUEUED.value
        )
        assert not any(p.id == prog.id for p in queued_after)

        # count_by_status should also reflect the removal
        count = await fakeredis_storage.count_by_status(ProgramState.QUEUED.value)
        assert count == 0

    async def test_remove_cleans_up_running_status_set(
        self, fakeredis_storage, make_program
    ):
        """Remove a RUNNING program; status set membership cleaned up."""
        prog = make_program(state=ProgramState.RUNNING)
        await fakeredis_storage.add(prog)

        count_before = await fakeredis_storage.count_by_status(
            ProgramState.RUNNING.value
        )
        assert count_before == 1

        await fakeredis_storage.remove(prog.id)

        count_after = await fakeredis_storage.count_by_status(
            ProgramState.RUNNING.value
        )
        assert count_after == 0

        # Also verify get_ids_by_status
        running_ids = await fakeredis_storage.get_ids_by_status(
            ProgramState.RUNNING.value
        )
        assert prog.id not in running_ids

    async def test_remove_cleans_up_done_status_set(
        self, fakeredis_storage, make_program
    ):
        """Remove a DONE program; status set is cleaned up."""
        prog = make_program(state=ProgramState.DONE)
        await fakeredis_storage.add(prog)

        await fakeredis_storage.remove(prog.id)

        done_ids = await fakeredis_storage.get_ids_by_status(ProgramState.DONE.value)
        assert prog.id not in done_ids
        assert await fakeredis_storage.count_by_status(ProgramState.DONE.value) == 0

    async def test_remove_nonexistent_does_not_corrupt_status_sets(
        self, fakeredis_storage, make_program
    ):
        """Removing a non-existent program doesn't affect other programs' status sets."""
        prog = make_program(state=ProgramState.QUEUED)
        await fakeredis_storage.add(prog)

        await fakeredis_storage.remove("nonexistent-program-id")

        # Original program should still be in QUEUED set
        count = await fakeredis_storage.count_by_status(ProgramState.QUEUED.value)
        assert count == 1


# ===================================================================
# Category Q: Audit Finding 5 — Key isolation between prefixes
# ===================================================================


class TestKeyIsolationBetweenPrefixes:
    """Audit finding 5: two storage instances with different prefixes must be isolated."""

    async def test_different_prefixes_are_isolated(self, make_program):
        """Programs stored in one prefix are invisible to another prefix."""
        server = fakeredis.FakeServer()

        config_a = RedisProgramStorageConfig(
            redis_url="redis://fake:6379/0",
            key_prefix="prefix_alpha",
        )
        config_b = RedisProgramStorageConfig(
            redis_url="redis://fake:6379/0",
            key_prefix="prefix_beta",
        )

        storage_a = RedisProgramStorage(config_a)
        storage_b = RedisProgramStorage(config_b)

        # Both share the same fakeredis server
        fake_redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
        storage_a._conn._redis = fake_redis
        storage_a._conn._closing = False
        storage_b._conn._redis = fake_redis
        storage_b._conn._closing = False

        try:
            prog_a = make_program(code="def alpha(): return 1")
            prog_b = make_program(code="def beta(): return 2")

            await storage_a.add(prog_a)
            await storage_b.add(prog_b)

            # storage_a can see prog_a but not prog_b
            fetched_a = await storage_a.get(prog_a.id)
            assert fetched_a is not None
            assert fetched_a.code == "def alpha(): return 1"
            assert await storage_a.get(prog_b.id) is None

            # storage_b can see prog_b but not prog_a
            fetched_b = await storage_b.get(prog_b.id)
            assert fetched_b is not None
            assert fetched_b.code == "def beta(): return 2"
            assert await storage_b.get(prog_a.id) is None

            # size() is prefix-scoped
            assert await storage_a.size() == 1
            assert await storage_b.size() == 1

            # get_all() is prefix-scoped
            all_a = await storage_a.get_all()
            all_b = await storage_b.get_all()
            assert len(all_a) == 1
            assert len(all_b) == 1
            assert all_a[0].id == prog_a.id
            assert all_b[0].id == prog_b.id

            # Status sets are prefix-scoped
            a_by_status = await storage_a.get_all_by_status(ProgramState.RUNNING.value)
            b_by_status = await storage_b.get_all_by_status(ProgramState.RUNNING.value)
            assert len(a_by_status) == 1
            assert len(b_by_status) == 1
            assert a_by_status[0].id == prog_a.id
            assert b_by_status[0].id == prog_b.id

        finally:
            await storage_a.close()
            await storage_b.close()


# ===================================================================
# Category: recover_stranded_programs — ghost ID regression
# ===================================================================


class TestRecoverStrandedGhostId:
    """Regression: a dangling ID in status:RUNNING (no program hash) must be
    silently removed without raising and must not be counted as recovered."""

    async def test_ghost_id_cleaned_up_no_exception(self, fakeredis_storage) -> None:
        ghost_id = "ghost-id-dangling-1234"

        # Inject raw ID into the RUNNING status set bypassing normal add()
        redis_conn = fakeredis_storage._conn._redis
        running_key = fakeredis_storage._keys.status_set(ProgramState.RUNNING.value)
        await redis_conn.sadd(running_key, ghost_id)

        assert await fakeredis_storage.count_by_status(ProgramState.RUNNING.value) == 1

        # Should not raise; ghost IDs are cleaned up silently
        recovered = await fakeredis_storage.recover_stranded_programs()

        # Ghost counts as 0 recovered (no real program moved to QUEUED)
        assert recovered == 0

        # Ghost ID must be gone from the RUNNING set
        ids = await fakeredis_storage.get_ids_by_status(ProgramState.RUNNING.value)
        assert ghost_id not in ids
        assert await fakeredis_storage.count_by_status(ProgramState.RUNNING.value) == 0

    async def test_ghost_id_mixed_with_real_program(
        self, fakeredis_storage, make_program
    ) -> None:
        """Ghost ID alongside a real RUNNING program: ghost removed, real recovered."""
        ghost_id = "ghost-id-dangling-5678"
        redis_conn = fakeredis_storage._conn._redis
        running_key = fakeredis_storage._keys.status_set(ProgramState.RUNNING.value)
        await redis_conn.sadd(running_key, ghost_id)

        # Also add a real program in RUNNING state
        prog = make_program(state=ProgramState.RUNNING)
        await fakeredis_storage.add(prog)

        assert await fakeredis_storage.count_by_status(ProgramState.RUNNING.value) == 2

        recovered = await fakeredis_storage.recover_stranded_programs()

        # Only the real program counts as recovered
        assert recovered == 1

        # Ghost gone, real program moved to QUEUED
        ids = await fakeredis_storage.get_ids_by_status(ProgramState.RUNNING.value)
        assert ghost_id not in ids
        queued = await fakeredis_storage.get_ids_by_status(ProgramState.QUEUED.value)
        assert prog.id in queued
