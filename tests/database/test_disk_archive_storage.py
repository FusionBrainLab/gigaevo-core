"""Tests for DiskArchiveStorage in gigaevo/evolution/storage/archive_storage.py"""

from __future__ import annotations

import pytest

from gigaevo.database.disk_program_storage import (
    DiskProgramStorage,
    DiskProgramStorageConfig,
)
from gigaevo.evolution.storage.archive_storage import DiskArchiveStorage
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


def _prog(metrics=None):
    p = Program(code="def solve(): return 1", state=ProgramState.DONE)
    if metrics:
        p.add_metrics(metrics)
    return p


def _always_better(new, current):
    return True


def _score_better(new, current):
    return new.metrics["score"] > current.metrics["score"]


@pytest.fixture
def storage(tmp_path) -> DiskProgramStorage:
    config = DiskProgramStorageConfig(root_dir=str(tmp_path), key_prefix="test")
    return DiskProgramStorage(config)


@pytest.fixture
def archive(storage) -> DiskArchiveStorage:
    return DiskArchiveStorage(storage, key_prefix="test")


async def test_add_elite_new_cell(storage, archive):
    p = _prog(metrics={"score": 10.0})
    await storage.add(p)

    assert await archive.add_elite((0, 1), p, _always_better) is True

    elite = await archive.get_elite((0, 1))
    assert elite is not None and elite.id == p.id


async def test_add_elite_loses_to_occupant(storage, archive):
    good = _prog(metrics={"score": 10.0})
    bad = _prog(metrics={"score": 1.0})
    await storage.add(good)
    await storage.add(bad)

    await archive.add_elite((0,), good, _always_better)
    assert await archive.add_elite((0,), bad, _score_better) is False

    elite = await archive.get_elite((0,))
    assert elite.id == good.id


async def test_add_elite_beats_occupant_evicts_and_updates_reverse(storage, archive):
    old = _prog(metrics={"score": 1.0})
    new = _prog(metrics={"score": 10.0})
    await storage.add(old)
    await storage.add(new)

    await archive.add_elite((0,), old, _always_better)
    assert await archive.add_elite((0,), new, _score_better) is True

    elite = await archive.get_elite((0,))
    assert elite.id == new.id
    assert await archive.get_all_elites() == [new.id]
    # Evicted occupant left the reverse index — removing it by id is a no-op
    assert await archive.remove_elite_by_id(old.id) is False


async def test_add_elite_idempotent_readd(storage, archive):
    p = _prog(metrics={"score": 1.0})
    await storage.add(p)
    await archive.add_elite((0,), p, _always_better)

    # Re-add of an existing elite returns True even if it would lose
    assert await archive.add_elite((1,), p, lambda n, c: False) is True
    assert await archive.size() == 1
    assert (await archive.get_elite((0,))).id == p.id


async def test_add_elite_missing_program_returns_false(archive):
    p = _prog()
    assert await archive.add_elite((0,), p, _always_better) is False
    assert await archive.size() == 0


async def test_remove_elite_by_id(storage, archive):
    p = _prog()
    await storage.add(p)
    await archive.add_elite((0,), p, _always_better)

    assert await archive.remove_elite_by_id(p.id) is True
    assert await archive.get_elite((0,)) is None
    assert await archive.remove_elite_by_id(p.id) is False


async def test_remove_elite_by_cell(storage, archive):
    p = _prog()
    await storage.add(p)
    await archive.add_elite((0,), p, _always_better)

    assert await archive.remove_elite((0,)) is True
    assert await archive.remove_elite((0,)) is False
    assert await archive.get_all_elites() == []


async def test_bulk_remove(storage, archive):
    progs = [_prog() for _ in range(3)]
    for i, p in enumerate(progs):
        await storage.add(p)
        await archive.add_elite((i,), p, _always_better)

    removed = await archive.bulk_remove_elites_by_id(
        [progs[0].id, progs[2].id, "not-an-elite"]
    )
    assert removed == 2
    assert await archive.get_all_elites() == [progs[1].id]


async def test_clear_all(storage, archive):
    for i in range(3):
        p = _prog()
        await storage.add(p)
        await archive.add_elite((i,), p, _always_better)

    assert await archive.clear_all_elites() == 3
    assert await archive.size() == 0
    assert await archive.clear_all_elites() == 0


async def test_bulk_add_and_size(storage, archive):
    p1 = _prog(metrics={"score": 1.0})
    p2 = _prog(metrics={"score": 2.0})
    await storage.add(p1)
    await storage.add(p2)

    added = await archive.bulk_add_elites([((0,), p1), ((1,), p2)], _always_better)
    assert added == 2
    assert await archive.size() == 2


async def test_replace_all_moves_cells_and_resolves_collisions(storage, archive):
    low = _prog(metrics={"score": 1.0})
    high = _prog(metrics={"score": 10.0})
    await storage.add(low)
    await storage.add(high)
    await archive.add_elite((0,), low, _always_better)

    count = await archive.replace_all_elites([((2,), low), ((2,), high)], _score_better)

    assert count == 1
    assert await archive.get_elite((0,)) is None
    assert (await archive.get_elite((2,))).id == high.id
    assert await archive.remove_elite_by_id(low.id) is False


async def test_persistence_across_instance_recreation(storage, archive):
    p1 = _prog(metrics={"score": 1.0})
    p2 = _prog(metrics={"score": 2.0})
    await storage.add(p1)
    await storage.add(p2)
    await archive.add_elite((0,), p1, _always_better)
    await archive.add_elite((1,), p2, _always_better)

    fresh_storage = DiskProgramStorage(
        DiskProgramStorageConfig(root_dir=storage.config.root_dir, key_prefix="test")
    )
    assert await fresh_storage.has_data() is True
    reloaded = DiskArchiveStorage(fresh_storage, key_prefix="test")
    assert await reloaded.size() == 2
    assert set(await reloaded.get_all_elites()) == {p1.id, p2.id}
    assert (await reloaded.get_elite((0,))).id == p1.id
    # Reverse index survives the reload too
    assert await reloaded.remove_elite_by_id(p2.id) is True
    assert await reloaded.size() == 1
