"""DiskProgramStorage crash-recovery: status sets reconcile with program state."""

from __future__ import annotations

import json

from gigaevo.database.disk_program_storage import (
    STAGE_OUTPUTS_DIR,
    STATUS_SETS_FILE,
    DiskProgramStorage,
    DiskProgramStorageConfig,
)
from gigaevo.evolution.mutation.constants import (
    MUTATION_PARENT_STAGE_OUTPUTS_METADATA_KEY,
)
from gigaevo.evolution.mutation.parent_snapshot import (
    resolve_parent_stage_outputs,
    snapshot_parent_stage_outputs,
)
from gigaevo.programs.core_types import ProgramStageResult
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.common import StringContainer


def _storage(tmp_path) -> DiskProgramStorage:
    return DiskProgramStorage(
        DiskProgramStorageConfig(root_dir=str(tmp_path), key_prefix="toy")
    )


async def test_dangling_status_entry_dropped_on_load(tmp_path):
    storage = _storage(tmp_path)
    async with storage:
        prog = Program(code="def f(): return 1", iteration=0)
        await storage.add(prog)

    status_path = tmp_path / "toy" / STATUS_SETS_FILE
    sets = json.loads(status_path.read_text())
    sets.setdefault(ProgramState.QUEUED.value, []).append("0" * 32)
    status_path.write_text(json.dumps(sets))

    async with _storage(tmp_path) as reopened:
        assert await reopened.count_by_status(ProgramState.QUEUED.value) == 1
        assert await reopened.get_ids_by_status(ProgramState.QUEUED.value) == [prog.id]


async def test_stale_status_entry_follows_program_state(tmp_path):
    storage = _storage(tmp_path)
    async with storage:
        prog = Program(code="def f(): return 1", iteration=0)
        prog.state = ProgramState.DONE
        await storage.add(prog)

    status_path = tmp_path / "toy" / STATUS_SETS_FILE
    sets = json.loads(status_path.read_text())
    sets[ProgramState.QUEUED.value] = [prog.id]
    sets[ProgramState.DONE.value] = []
    status_path.write_text(json.dumps(sets))

    async with _storage(tmp_path) as reopened:
        assert await reopened.count_by_status(ProgramState.QUEUED.value) == 0
        assert await reopened.count_by_status(ProgramState.DONE.value) == 1


async def test_program_missing_from_all_sets_is_recovered(tmp_path):
    storage = _storage(tmp_path)
    async with storage:
        prog = Program(code="def f(): return 1", iteration=0)
        await storage.add(prog)

    status_path = tmp_path / "toy" / STATUS_SETS_FILE
    status_path.write_text(json.dumps({}))

    async with _storage(tmp_path) as reopened:
        assert await reopened.get_ids_by_status(ProgramState.QUEUED.value) == [prog.id]


async def test_stage_output_store_persists_across_reopen(tmp_path):
    async with _storage(tmp_path) as storage:
        await storage.put_stage_output("cid-1", "blob-A")
        assert await storage.get_stage_output("cid-1") == "blob-A"

    async with _storage(tmp_path) as reopened:
        assert await reopened.get_stage_output("cid-1") == "blob-A"
        assert await reopened.get_stage_output("never-written") is None


async def test_stage_output_store_is_first_write_wins(tmp_path):
    async with _storage(tmp_path) as storage:
        await storage.put_stage_output("cid-2", "blob-A")
        await storage.put_stage_output("cid-2", "blob-B")

        assert await storage.get_stage_output("cid-2") == "blob-A"


async def test_stage_output_store_is_namespaced_under_prefix(tmp_path):
    async with _storage(tmp_path) as storage:
        await storage.put_stage_output("cid-3", "blob-C")

    assert (tmp_path / "toy" / STAGE_OUTPUTS_DIR / "cid-3").is_file()


async def test_parent_snapshot_roundtrips_on_disk_storage(tmp_path):
    parent = Program(code="def f(): return 1", state=ProgramState.DONE)
    ctx = StringContainer(data="## Program Insights\n- original card-derived context")
    parent.stage_results = {
        "MutationContextStage": ProgramStageResult.success(output=ctx)
    }

    async with _storage(tmp_path) as storage:
        id_map = await snapshot_parent_stage_outputs([parent], storage)
        child = Program(code="def g(): return 2")
        child.set_metadata(MUTATION_PARENT_STAGE_OUTPUTS_METADATA_KEY, id_map)
        resolved = await resolve_parent_stage_outputs(child, storage)

    assert resolved[parent.id]["MutationContextStage"] == ctx
