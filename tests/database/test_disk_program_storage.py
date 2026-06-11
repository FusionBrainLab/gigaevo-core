"""DiskProgramStorage crash-recovery: status sets reconcile with program state."""

from __future__ import annotations

import json

from gigaevo.database.disk_program_storage import (
    STATUS_SETS_FILE,
    DiskProgramStorage,
    DiskProgramStorageConfig,
)
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


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
