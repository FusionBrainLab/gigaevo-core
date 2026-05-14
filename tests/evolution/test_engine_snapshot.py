from __future__ import annotations

import asyncio

from pydantic import ValidationError
import pytest

from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.engine.snapshot import (
    ENGINE_SNAPSHOT_KEY,
    EngineSnapshot,
    get_current_snapshot,
    load_engine_snapshot,
)


def test_default_snapshot_fields_are_zero_or_none():
    snap = EngineSnapshot()
    assert snap.total_mutants == 0
    assert snap.programs_processed == 0
    assert snap.completion_reason is None
    assert snap.version == 0


def test_extra_fields_are_forbidden():
    with pytest.raises(ValidationError):
        EngineSnapshot(unknown_field=1)


def test_get_current_snapshot_returns_defaults_on_fresh_module():
    snap = get_current_snapshot()
    assert snap == EngineSnapshot()


class _StubStorage:
    def __init__(self, payload: str | None = None):
        self._payload = payload
        self.saved: list[tuple[str, str]] = []

    async def load_run_state_str(self, field: str) -> str | None:
        if field != ENGINE_SNAPSHOT_KEY:
            return None
        return self._payload

    async def save_run_state(self, field: str, value: int | str) -> None:
        self.saved.append((field, str(value)))


@pytest.mark.asyncio
async def test_load_engine_snapshot_returns_defaults_when_missing():
    storage = _StubStorage(payload=None)
    snap = await load_engine_snapshot(storage)
    assert snap == EngineSnapshot()


@pytest.mark.asyncio
async def test_load_engine_snapshot_round_trips_json():
    payload = EngineSnapshot(total_mutants=7, version=3).model_dump_json()
    storage = _StubStorage(payload=payload)
    snap = await load_engine_snapshot(storage)
    assert snap.total_mutants == 7
    assert snap.version == 3


@pytest.mark.asyncio
async def test_load_engine_snapshot_tolerates_corrupt_json():
    storage = _StubStorage(payload="{not json")
    snap = await load_engine_snapshot(storage)
    assert snap == EngineSnapshot()


@pytest.fixture
def engine_with_storage(fakeredis_storage):
    engine = object.__new__(EvolutionEngine)
    engine.storage = fakeredis_storage
    engine._snapshot = EngineSnapshot()
    engine._snapshot_lock = asyncio.Lock()
    return engine


@pytest.mark.asyncio
async def test_write_snapshot_merges_updates_and_bumps_version(engine_with_storage):
    await engine_with_storage._write_snapshot(total_mutants=3)
    assert engine_with_storage._snapshot.total_mutants == 3
    assert engine_with_storage._snapshot.version == 1
    assert get_current_snapshot().total_mutants == 3

    await engine_with_storage._write_snapshot(programs_processed=5)
    assert engine_with_storage._snapshot.total_mutants == 3  # preserved
    assert engine_with_storage._snapshot.programs_processed == 5
    assert engine_with_storage._snapshot.version == 2


@pytest.mark.asyncio
async def test_write_snapshot_persists_to_redis(engine_with_storage, fakeredis_storage):
    await engine_with_storage._write_snapshot(total_mutants=7)
    raw = await fakeredis_storage.load_run_state_str(ENGINE_SNAPSHOT_KEY)
    snap = EngineSnapshot.model_validate_json(raw)
    assert snap.total_mutants == 7
    assert snap.version == 1


@pytest.mark.asyncio
async def test_write_snapshot_with_no_updates_still_bumps_version(engine_with_storage):
    """Heartbeat behavior: empty _write_snapshot() bumps version only."""
    await engine_with_storage._write_snapshot()
    assert engine_with_storage._snapshot.version == 1
    assert engine_with_storage._snapshot.total_mutants == 0
    assert engine_with_storage._snapshot.programs_processed == 0
    assert engine_with_storage._snapshot.completion_reason is None


@pytest.mark.asyncio
async def test_load_snapshot_on_resume_hydrates_from_redis(
    engine_with_storage, fakeredis_storage
):
    payload = EngineSnapshot(total_mutants=5, version=4).model_dump_json()
    await fakeredis_storage.save_run_state(ENGINE_SNAPSHOT_KEY, payload)

    await engine_with_storage._load_snapshot_on_resume()

    assert engine_with_storage._snapshot.total_mutants == 5
    assert engine_with_storage._snapshot.version == 4
    assert get_current_snapshot().total_mutants == 5


@pytest.mark.asyncio
async def test_concurrent_write_snapshot_keeps_redis_and_memory_in_sync(
    engine_with_storage, fakeredis_storage
):
    """Concurrent _write_snapshot calls from many mutant tasks must leave
    the Redis-persisted snapshot version equal to the in-memory mirror.
    Without the snapshot lock, two awaits could resolve out of order and
    Redis would end at an older version than memory — losing updates on
    resume.
    """
    n = 50
    await asyncio.gather(
        *(engine_with_storage._write_snapshot(total_mutants=i) for i in range(1, n + 1))
    )
    # Each call bumps version by 1; total writes == n.
    assert engine_with_storage._snapshot.version == n
    raw = await fakeredis_storage.load_run_state_str(ENGINE_SNAPSHOT_KEY)
    redis_snap = EngineSnapshot.model_validate_json(raw)
    assert redis_snap.version == engine_with_storage._snapshot.version, (
        f"Redis version {redis_snap.version} diverged from in-memory "
        f"{engine_with_storage._snapshot.version} — snapshot race not contained"
    )
    assert redis_snap.total_mutants == engine_with_storage._snapshot.total_mutants
