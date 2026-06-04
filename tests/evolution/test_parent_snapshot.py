"""Snapshot of the parent stage outputs that produced each child.

At birth a child records, per parent, the cache id of every parent stage output
(content-addressable, written to the storage stage-output store). The child holds
only the lean id-map; the heavy outputs live once in the store and dedup across
siblings. This survives the parent's NO_CACHE context/suggestion stages being
overwritten when the parent is re-selected or re-evaluated.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from gigaevo.evolution.engine.mutation import generate_one_mutation
from gigaevo.evolution.mutation.base import MutationSpec
from gigaevo.evolution.mutation.constants import (
    MUTATION_PARENT_STAGE_OUTPUTS_METADATA_KEY,
)
from gigaevo.evolution.mutation.parent_snapshot import (
    resolve_parent_stage_outputs,
    snapshot_parent_stage_outputs,
    stage_output_cache_id,
)
from gigaevo.programs.core_types import ProgramStageResult
from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS, Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.common import StringContainer


def _parent(code: str = "def f(): return 1", **stage_outputs: object) -> Program:
    p = Program(code=code, state=ProgramState.DONE, atomic_counter=999_999_999)
    p.stage_results = {
        name: ProgramStageResult.success(output=out)
        for name, out in stage_outputs.items()
    }
    return p


class TestCacheId:
    def test_same_blob_same_id(self):
        assert stage_output_cache_id("blob-A") == stage_output_cache_id("blob-A")

    def test_different_blob_different_id(self):
        assert stage_output_cache_id("blob-A") != stage_output_cache_id("blob-B")


class TestSnapshot:
    async def test_builds_id_map_keyed_by_parent_then_stage(self, fakeredis_storage):
        parent = _parent(
            MutationContextStage=StringContainer(data="## Program Insights\n- knn"),
            EvolutionaryStatisticsCollector=StringContainer(data="gen 3"),
        )

        id_map = await snapshot_parent_stage_outputs([parent], fakeredis_storage)

        assert set(id_map) == {parent.id}
        assert set(id_map[parent.id]) == {
            "MutationContextStage",
            "EvolutionaryStatisticsCollector",
        }

    async def test_outputs_written_to_store_and_resolvable(self, fakeredis_storage):
        ctx = StringContainer(data="## Program Insights\n- act on knn suggestion")
        parent = _parent(MutationContextStage=ctx)

        id_map = await snapshot_parent_stage_outputs([parent], fakeredis_storage)
        cid = id_map[parent.id]["MutationContextStage"]
        assert await fakeredis_storage.get_stage_output(cid) is not None

        child = Program(code="def g(): return 2")
        child.set_metadata(MUTATION_PARENT_STAGE_OUTPUTS_METADATA_KEY, id_map)
        resolved = await resolve_parent_stage_outputs(child, fakeredis_storage)
        assert resolved[parent.id]["MutationContextStage"] == ctx

    async def test_identical_outputs_dedup_to_same_id(self, fakeredis_storage):
        shared = StringContainer(data="identical context")
        pa = _parent(code="def a(): ...", MutationContextStage=shared)
        pb = _parent(code="def b(): ...", MutationContextStage=shared)

        id_map = await snapshot_parent_stage_outputs([pa, pb], fakeredis_storage)

        assert (
            id_map[pa.id]["MutationContextStage"]
            == id_map[pb.id]["MutationContextStage"]
        )

    async def test_skips_none_outputs(self, fakeredis_storage):
        parent = _parent(MutationContextStage=StringContainer(data="ctx"))
        parent.stage_results["SkippedStage"] = ProgramStageResult.skipped(
            message="no input", stage="SkippedStage"
        )

        id_map = await snapshot_parent_stage_outputs([parent], fakeredis_storage)

        assert set(id_map[parent.id]) == {"MutationContextStage"}

    async def test_empty_when_no_stage_results(self, fakeredis_storage):
        parent = _parent()  # no stage outputs
        id_map = await snapshot_parent_stage_outputs([parent], fakeredis_storage)
        assert id_map == {}

    async def test_reloads_full_record_when_parent_is_lean(self, fakeredis_storage):
        """Steady-state hands mutation a lean parent (stage_results excluded for
        speed); the full outputs live only in storage. The snapshot must re-load
        the full record so those outputs are still captured."""
        full = _parent(
            MutationContextStage=StringContainer(data="ctx for child"),
            MutationSuggestionStage=StringContainer(data="suggest knn"),
        )
        await fakeredis_storage.add(full)

        # Exactly how the refresher loads parents for mutation.
        lean = (await fakeredis_storage.mget([full.id], exclude=EXCLUDE_STAGE_RESULTS))[
            0
        ]
        assert not lean.stage_results  # precondition: lean object has no stage outputs

        id_map = await snapshot_parent_stage_outputs([lean], fakeredis_storage)

        assert set(id_map[full.id]) == {
            "MutationContextStage",
            "MutationSuggestionStage",
        }

    async def test_resolve_immune_to_parent_overwrite(self, fakeredis_storage):
        original = StringContainer(data="## Program Insights\n- original")
        parent = _parent(MutationContextStage=original)

        id_map = await snapshot_parent_stage_outputs([parent], fakeredis_storage)
        child = Program(code="def g(): return 2")
        child.set_metadata(MUTATION_PARENT_STAGE_OUTPUTS_METADATA_KEY, id_map)

        # Parent re-selected: NO_CACHE context stage reruns, clobbering its output.
        parent.stage_results["MutationContextStage"] = ProgramStageResult.success(
            output=StringContainer(data="OVERWRITTEN later suggestion")
        )

        resolved = await resolve_parent_stage_outputs(child, fakeredis_storage)
        assert resolved[parent.id]["MutationContextStage"] == original

    async def test_resolve_empty_when_no_metadata(self, fakeredis_storage):
        child = Program(code="def g(): return 2")
        assert await resolve_parent_stage_outputs(child, fakeredis_storage) == {}

    async def test_resolve_skips_evicted_store_entry(self, fakeredis_storage):
        """Store entries can be flushed/evicted independently of a child's id-map;
        a reference to an absent cache id is dropped, not raised."""
        child = Program(code="def g(): return 2")
        child.set_metadata(
            MUTATION_PARENT_STAGE_OUTPUTS_METADATA_KEY,
            {"parent-x": {"MutationContextStage": "evicted-cache-id"}},
        )

        resolved = await resolve_parent_stage_outputs(child, fakeredis_storage)

        assert resolved == {"parent-x": {}}


class TestEngineWiring:
    async def test_generate_one_mutation_stamps_and_resolves(
        self, fakeredis_storage, state_manager
    ):
        ctx = StringContainer(data="## Program Insights\n- knn feature")
        parent = _parent(MutationContextStage=ctx)

        mutator = AsyncMock()
        mutator.mutate_single.return_value = MutationSpec(
            code="def child(): return 9",
            parents=[parent],
            name="m",
            metadata={},
        )

        child_id = await generate_one_mutation(
            [parent],
            mutator=mutator,
            storage=fakeredis_storage,
            state_manager=state_manager,
            iteration=0,
        )

        assert child_id is not None
        child = await fakeredis_storage.get(child_id)
        id_map = child.get_metadata(MUTATION_PARENT_STAGE_OUTPUTS_METADATA_KEY)
        assert id_map[parent.id]["MutationContextStage"]

        resolved = await resolve_parent_stage_outputs(child, fakeredis_storage)
        assert resolved[parent.id]["MutationContextStage"] == ctx

    async def test_snapshot_failure_does_not_block_mutation(
        self, fakeredis_storage, state_manager
    ):
        """A debug snapshot must never kill a mutation: if the store write
        raises, the child is still persisted."""
        ctx = StringContainer(data="ctx")
        parent = _parent(MutationContextStage=ctx)

        mutator = AsyncMock()
        mutator.mutate_single.return_value = MutationSpec(
            code="def child(): return 9", parents=[parent], name="m", metadata={}
        )
        fakeredis_storage.put_stage_output = AsyncMock(
            side_effect=RuntimeError("store down")
        )

        child_id = await generate_one_mutation(
            [parent],
            mutator=mutator,
            storage=fakeredis_storage,
            state_manager=state_manager,
            iteration=0,
        )

        assert child_id is not None
        child = await fakeredis_storage.get(child_id)
        assert child is not None
