"""Tests for CellStratifiedRedisOpponentArchiveProvider (v3 2D BD niche-diverse opponent HoF).

Schema contract (matches ``gigaevo/evolution/storage/archive_storage.py``):
  - ``island_{island_id}:archive``  HASH  cell_field -> program_id
  - ``{prefix}:program:{pid}``       STRING (JSON)  with ``.code`` and
    ``.metrics[fitness_key]``

Tests exercise both:
  1. A direct fake-Redis harness that writes the production schema by hand,
  2. An integration-style harness that seeds the archive through the actual
     ``RedisArchiveStorage.add_elite`` writer (fakeredis backend) to catch
     schema drift between producer and consumer — the failure mode that let
     the pre-v3 bogus-schema implementation pass its tests.
"""

from __future__ import annotations

import json

import fakeredis.aioredis
import pytest

from gigaevo.adversarial.opponent_provider import (
    CellStratifiedRedisOpponentArchiveProvider,
)

ISLAND_ID = "fitness_island"
ARCHIVE_KEY = f"island_{ISLAND_ID}:archive"


def _program_json(pid: str, code: str, metrics: dict[str, float]) -> str:
    return json.dumps({"id": pid, "code": code, "metrics": metrics})


# --------------------- Direct schema harness ---------------------


@pytest.fixture
async def provider():
    p = CellStratifiedRedisOpponentArchiveProvider(
        host="localhost",
        port=6379,
        db=0,
        prefix="test",
        fitness_key="quality",
        k=3,
        island_id=ISLAND_ID,
        cache_ttl=0.0,
    )
    p._redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield p
    await p.close()


@pytest.fixture
async def provider_with_archive(provider):
    """Seed archive with production schema: one program per cell field."""
    seed = [
        ("p1", 0.95, "0,0"),
        ("p2", 0.90, "0,1"),
        ("p3", 0.88, "1,0"),
        ("p4", 0.85, "1,1"),
        ("p5", 0.75, "2,0"),
    ]
    for pid, quality, cell in seed:
        await provider._redis.hset(ARCHIVE_KEY, cell, pid)
        await provider._redis.set(
            f"{provider._prefix}:program:{pid}",
            _program_json(pid, f"code_{pid}", {"quality": quality, "fitness": quality}),
        )
    return provider


@pytest.mark.asyncio
async def test_top_k_returns_distinct_cells(provider_with_archive):
    programs = await provider_with_archive.get_top_k(k=3)
    assert len(programs) == 3
    # Top-3 by quality desc: p1 (0.95), p2 (0.90), p3 (0.88)
    assert [p.program_id for p in programs] == ["p1", "p2", "p3"]
    assert [p.fitness for p in programs] == [0.95, 0.90, 0.88]


@pytest.mark.asyncio
async def test_top_k_empty_archive(provider):
    assert await provider.get_top_k(k=3) == []


@pytest.mark.asyncio
async def test_top_k_returns_fewer_than_k_when_archive_small(provider):
    # One cell, one program
    await provider._redis.hset(ARCHIVE_KEY, "0,0", "solo")
    await provider._redis.set(
        f"{provider._prefix}:program:solo",
        _program_json("solo", "code_solo", {"quality": 0.5}),
    )
    programs = await provider.get_top_k(k=3)
    assert len(programs) == 1
    assert programs[0].program_id == "solo"


@pytest.mark.asyncio
async def test_top_k_skips_programs_without_fitness_key(provider):
    # p1 has quality, p2 does not; provider must skip p2 silently
    await provider._redis.hset(ARCHIVE_KEY, "0,0", "p1")
    await provider._redis.hset(ARCHIVE_KEY, "0,1", "p2")
    await provider._redis.set(
        f"{provider._prefix}:program:p1",
        _program_json("p1", "code_p1", {"quality": 0.5}),
    )
    await provider._redis.set(
        f"{provider._prefix}:program:p2",
        # Missing `quality` — only has `fitness`
        _program_json("p2", "code_p2", {"fitness": 0.9}),
    )
    programs = await provider.get_top_k(k=3)
    assert [p.program_id for p in programs] == ["p1"]


@pytest.mark.asyncio
async def test_top_k_deterministic_tiebreak_by_program_id(provider):
    # Two programs with identical fitness — tiebreak by program_id ASC
    await provider._redis.hset(ARCHIVE_KEY, "0,0", "zeta")
    await provider._redis.hset(ARCHIVE_KEY, "0,1", "alpha")
    for pid in ("zeta", "alpha"):
        await provider._redis.set(
            f"{provider._prefix}:program:{pid}",
            _program_json(pid, f"code_{pid}", {"quality": 0.5}),
        )
    programs = await provider.get_top_k(k=1)
    # Sort key is (-fitness, program_id) so "alpha" < "zeta" → alpha first
    assert programs[0].program_id == "alpha"


@pytest.mark.asyncio
async def test_top_k_honours_fitness_key_not_fitness(provider):
    # Program A wins on `quality`, program B wins on `fitness`.
    # fitness_key="quality" must pick A.
    await provider._redis.hset(ARCHIVE_KEY, "0,0", "A")
    await provider._redis.hset(ARCHIVE_KEY, "0,1", "B")
    await provider._redis.set(
        f"{provider._prefix}:program:A",
        _program_json("A", "code_A", {"quality": 0.9, "fitness": 0.1}),
    )
    await provider._redis.set(
        f"{provider._prefix}:program:B",
        _program_json("B", "code_B", {"quality": 0.2, "fitness": 0.99}),
    )
    programs = await provider.get_top_k(k=2)
    assert programs[0].program_id == "A"
    assert programs[0].fitness == 0.9


@pytest.mark.asyncio
async def test_top_k_lower_is_better(provider):
    # fitness_key="quality", higher_is_better=False (e.g. loss metric)
    provider._higher_is_better = False
    for pid, q, cell in (("p1", 0.1, "0,0"), ("p2", 0.5, "0,1"), ("p3", 0.9, "1,0")):
        await provider._redis.hset(ARCHIVE_KEY, cell, pid)
        await provider._redis.set(
            f"{provider._prefix}:program:{pid}",
            _program_json(pid, f"code_{pid}", {"quality": q}),
        )
    programs = await provider.get_top_k(k=2, higher_is_better=False)
    assert [p.program_id for p in programs] == ["p1", "p2"]


@pytest.mark.asyncio
async def test_get_opponents_by_ids_parent_behavior(provider_with_archive):
    # Inherited method must still work against the new schema
    programs = await provider_with_archive.get_programs_by_ids(["p1", "p4"])
    ids = sorted(p.program_id for p in programs)
    assert ids == ["p1", "p4"]


# --------------------- HOF_ROTATE emission ---------------------


@pytest.fixture
def loguru_sink():
    """Capture loguru messages into a list (loguru doesn't propagate to caplog)."""
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m), level="INFO")
    yield messages
    logger.remove(sink_id)


@pytest.mark.asyncio
async def test_hof_rotate_emits_on_initial_and_composition_change(
    provider, loguru_sink
):
    """HOF_ROTATE fires when elite-id set changes between consecutive fetches.

    Contract:
      1. First non-empty fetch: old=0, new=k  → emit (initial rotation).
      2. Same archive, second fetch: set unchanged → no new emit.
      3. One cell's program swapped: set changed → emit.
    """
    seed = [("p1", 0.95, "0,0"), ("p2", 0.90, "0,1"), ("p3", 0.88, "1,0")]
    for pid, q, cell in seed:
        await provider._redis.hset(ARCHIVE_KEY, cell, pid)
        await provider._redis.set(
            f"{provider._prefix}:program:{pid}",
            _program_json(pid, f"code_{pid}", {"quality": q}),
        )

    def _rotates_in(start_idx: int) -> list[dict]:
        out = []
        for msg in loguru_sink[start_idx:]:
            if "[HOF_ROTATE]" in msg:
                payload = json.loads(msg.split("[HOF_ROTATE]", 1)[1].strip())
                out.append(payload)
        return out

    # First fetch — initial rotation from empty HoF.
    mark = len(loguru_sink)
    picked_1 = await provider.get_top_k(k=3)
    assert [p.program_id for p in picked_1] == ["p1", "p2", "p3"]
    rotates_1 = _rotates_in(mark)
    assert len(rotates_1) == 1, (
        f"expected 1 HOF_ROTATE on initial fetch, got {len(rotates_1)}"
    )
    assert rotates_1[0]["event"] == "HOF_ROTATE"
    assert rotates_1[0]["old_hof_size"] == 0
    assert rotates_1[0]["new_hof_size"] == 3

    # Second fetch — unchanged archive, no rotation.
    mark = len(loguru_sink)
    picked_2 = await provider.get_top_k(k=3)
    assert [p.program_id for p in picked_2] == ["p1", "p2", "p3"]
    assert _rotates_in(mark) == [], "no rotation expected on stable archive"

    # Swap cell 0,1 program p2 → q2 : composition change, same size.
    await provider._redis.hset(ARCHIVE_KEY, "0,1", "q2")
    await provider._redis.set(
        f"{provider._prefix}:program:q2",
        _program_json("q2", "code_q2", {"quality": 0.92}),
    )

    mark = len(loguru_sink)
    picked_3 = await provider.get_top_k(k=3)
    assert {p.program_id for p in picked_3} == {"p1", "q2", "p3"}
    rotates_3 = _rotates_in(mark)
    assert len(rotates_3) == 1, (
        f"expected HOF_ROTATE on composition change (p2→q2), got {len(rotates_3)}"
    )
    assert rotates_3[0]["old_hof_size"] == 3
    assert rotates_3[0]["new_hof_size"] == 3


@pytest.mark.asyncio
async def test_hof_rotate_suppressed_when_archive_empty(provider, loguru_sink):
    """Empty archive → empty fetch → no HOF_ROTATE (there is nothing to rotate)."""
    picked = await provider.get_top_k(k=3)
    assert picked == []
    assert not any("[HOF_ROTATE]" in m for m in loguru_sink)


# --------------------- Integration harness ---------------------


@pytest.mark.asyncio
async def test_reads_schema_written_by_real_archive_storage(monkeypatch):
    """Critical integration test: seed archive via RedisArchiveStorage.add_elite.

    This is the test that would have caught C1 (key-schema drift). It does
    NOT hand-construct Redis keys — it drives the real producer and then
    verifies the real consumer reads what was written.
    """
    from types import SimpleNamespace

    from gigaevo.evolution.storage.archive_storage import RedisArchiveStorage
    from gigaevo.programs.program import Program

    # --- wire a fakeredis-backed program storage + archive storage ---
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)

    # Minimal RedisProgramStorage-shaped stub: we only need `with_redis`,
    # `exists`, `mget`, `get`, and `config.key_prefix`.
    stored: dict[str, Program] = {}
    PREFIX = "intg"

    async def _with_redis(_tag, op):
        return await op(fake)

    async def _exists(pid: str) -> bool:
        return pid in stored

    async def _get(pid: str) -> Program | None:
        return stored.get(pid)

    async def _mget(pids: list[str]) -> list[Program]:
        return [stored[p] for p in pids if p in stored]

    program_storage_stub = SimpleNamespace(
        with_redis=_with_redis,
        exists=_exists,
        get=_get,
        mget=_mget,
        config=SimpleNamespace(key_prefix=f"island_{ISLAND_ID}"),
    )

    archive = RedisArchiveStorage(
        program_storage=program_storage_stub,  # type: ignore[arg-type]
        key_prefix=f"island_{ISLAND_ID}",
    )

    def _make(pid: str, quality: float) -> Program:
        prog = Program(
            id=pid,
            code=f"code_{pid}",
            metrics={"quality": quality, "fitness": quality},
        )
        stored[pid] = prog
        return prog

    # Seed programs into both the stub store AND a per-run Redis JSON record
    # (what production does — the opponent provider reads the per-run JSON).
    import uuid

    seed_raw = [
        ("A", 0.95, (0, 0)),
        ("B", 0.90, (0, 1)),
        ("C", 0.88, (1, 0)),
    ]
    # Deterministic UUIDs so top-K ordering is stable across runs.
    seed = [
        (str(uuid.UUID(int=i + 1)), tag, q, cell)
        for i, (tag, q, cell) in enumerate(seed_raw)
    ]
    for pid, _tag, q, cell in seed:
        prog = _make(pid, q)
        await fake.set(
            f"{PREFIX}:program:{pid}",
            _program_json(pid, prog.code, prog.metrics),
        )
        ok = await archive.add_elite(cell, prog, is_better=lambda a, b: True)
        assert ok, f"add_elite rejected cell={cell} pid={pid}"

    # --- now instantiate the provider and verify it reads the real schema ---
    provider = CellStratifiedRedisOpponentArchiveProvider(
        host="localhost",
        port=6379,
        db=0,
        prefix=PREFIX,
        fitness_key="quality",
        k=3,
        island_id=ISLAND_ID,
        cache_ttl=0.0,
    )
    provider._redis = fake
    try:
        programs = await provider.get_top_k(k=3)
    finally:
        await provider.close()

    expected_ids = [pid for pid, _tag, _q, _cell in seed]
    expected_fit = [q for _pid, _tag, q, _cell in seed]
    assert [p.program_id for p in programs] == expected_ids
    assert [p.fitness for p in programs] == expected_fit
