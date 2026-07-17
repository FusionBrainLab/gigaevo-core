"""Behavioral contract every ProgramStorage backend must satisfy."""

from __future__ import annotations

import pytest

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.exceptions import StorageError
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from tests.database.storage_backends import BACKENDS, StorageBackend


@pytest.fixture(params=BACKENDS, ids=lambda b: b.id)
def backend(request) -> StorageBackend:
    return request.param


def _program(code: str = "def f(): return 1") -> Program:
    return Program(code=code, iteration=0)


async def test_add_get_roundtrip(backend):
    async with backend.make() as storage:
        prog = _program()
        await storage.add(prog)
        fetched = await storage.get(prog.id)
        assert fetched is not None and fetched.id == prog.id
        assert fetched.code == prog.code


async def test_exists_and_remove(backend):
    async with backend.make() as storage:
        prog = _program()
        await storage.add(prog)
        assert await storage.exists(prog.id) is True
        await storage.remove(prog.id)
        assert await storage.exists(prog.id) is False


async def test_size_and_has_data(backend):
    async with backend.make() as storage:
        assert await storage.has_data() is False
        await storage.add(_program("def a(): return 1"))
        await storage.add(_program("def b(): return 2"))
        assert await storage.size() == 2
        assert await storage.has_data() is True


async def test_clear_wipes_everything(backend):
    async with backend.make() as storage:
        await storage.add(_program())
        await storage.clear()
        assert await storage.has_data() is False


async def test_get_all_returns_added_programs(backend):
    async with backend.make() as storage:
        ids = set()
        for i in range(3):
            p = _program(f"def f(): return {i}")
            await storage.add(p)
            ids.add(p.id)
        assert {p.id for p in await storage.get_all()} == ids


async def test_read_only_rejects_writes(backend):
    async with backend.make(read_only=True) as storage:
        with pytest.raises(StorageError):
            await storage.add(_program())


async def test_key_prefix_is_exposed(backend):
    async with backend.make() as storage:
        assert isinstance(storage.key_prefix, str) and storage.key_prefix


async def test_async_context_manager_is_usable(backend):
    async with backend.make() as storage:
        assert isinstance(storage, ProgramStorage)


async def test_transition_status_persisted_and_refetched(backend):
    """transition_status moves program between status sets;
    after re-fetch, the status set counts are correct."""
    async with backend.make() as storage:
        prog = _program()
        prog.state = ProgramState.QUEUED
        await storage.add(prog)

        await storage.transition_status(
            prog.id, ProgramState.QUEUED.value, ProgramState.RUNNING.value
        )

        queued_ids = await storage.get_ids_by_status(ProgramState.QUEUED.value)
        running_ids = await storage.get_ids_by_status(ProgramState.RUNNING.value)
        assert prog.id not in queued_ids
        assert prog.id in running_ids


async def test_atomic_state_transition_persisted_and_refetched(backend):
    """atomic_state_transition updates both the program data and status sets;
    after re-fetch, the program has the new state."""
    async with backend.make() as storage:
        prog = _program()
        prog.state = ProgramState.QUEUED
        await storage.add(prog)

        prog.state = ProgramState.RUNNING
        await storage.atomic_state_transition(
            prog, ProgramState.QUEUED.value, ProgramState.RUNNING.value
        )

        fetched = await storage.get(prog.id)
        assert fetched is not None
        assert fetched.state == ProgramState.RUNNING

        running_progs = await storage.get_all_by_status(ProgramState.RUNNING.value)
        assert any(p.id == prog.id for p in running_progs)

        queued_progs = await storage.get_all_by_status(ProgramState.QUEUED.value)
        assert not any(p.id == prog.id for p in queued_progs)


async def test_data_update_preserves_live_state_and_status_index(backend):
    """A stale data writer cannot undo a concurrent lifecycle transition."""
    async with backend.make() as storage:
        parent = _program()
        parent.state = ProgramState.DONE
        await storage.add(parent)
        stale_parent = await storage.get(parent.id)
        assert stale_parent is not None

        parent_ids = [parent.id]
        assert await storage.batch_transition_by_ids(
            parent_ids,
            ProgramState.DONE.value,
            ProgramState.QUEUED.value,
        ) == len(parent_ids)
        assert await storage.batch_transition_by_ids(
            parent_ids,
            ProgramState.QUEUED.value,
            ProgramState.RUNNING.value,
        ) == len(parent_ids)

        child = _program()
        stale_parent.lineage.add_child(child.id)
        await storage.update(stale_parent)

        fetched = await storage.get(parent.id)
        assert fetched is not None
        assert fetched.state == ProgramState.RUNNING
        assert child.id in fetched.lineage.children
        assert parent.id in await storage.get_ids_by_status(ProgramState.RUNNING.value)
        assert parent.id not in await storage.get_ids_by_status(ProgramState.DONE.value)


async def test_count_by_status(backend):
    """count_by_status returns correct count."""
    async with backend.make() as storage:
        for _ in range(3):
            prog = _program()
            prog.state = ProgramState.QUEUED
            await storage.add(prog)

        count = await storage.count_by_status(ProgramState.QUEUED.value)
        assert count == 3


async def test_close_releases_instance_lock(backend):
    """After close(), the instance lock is relinquished (renew must fail)."""
    async with backend.make() as storage:
        assert await storage.acquire_instance_lock()
        await storage.close()
        assert not await storage.renew_instance_lock()
