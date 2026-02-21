"""RedisProgramStorage CRUD and status operations tests with fakeredis."""

from __future__ import annotations

from gigaevo.programs.core_types import (
    ProgramStageResult,
    StageState,
)
from gigaevo.programs.program_state import ProgramState
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
