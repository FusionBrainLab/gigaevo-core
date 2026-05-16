"""Integration tests for the archive cell-swap API.

Covers ``archive_swap.lua`` plus :meth:`DataPlane.try_replace_elite`.
The Lua script atomically compares a candidate's score against the
occupant's stored score and either inserts (empty cell), swaps
(candidate wins), or rejects (occupant wins). Tie-break is one bit.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import fakeredis.aioredis
import pytest

import gigaevo.dataplane as dp


@pytest.fixture
async def coord() -> AsyncIterator[dp.DataPlane]:
    server = fakeredis.FakeServer()
    coord = dp.DataPlane("redis://embedded/0", key_prefix="test")
    fake = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    coord._connection._pool = fake  # type: ignore[attr-defined]
    from gigaevo.dataplane.scripts import LuaRegistry

    lua = LuaRegistry(fake)
    coord._register_builtin_scripts(lua)  # type: ignore[attr-defined]
    await lua.load_all()
    coord._lua = lua  # type: ignore[attr-defined]
    coord._started = True  # type: ignore[attr-defined]
    try:
        yield coord
    finally:
        coord._started = False  # type: ignore[attr-defined]
        coord._lua = None  # type: ignore[attr-defined]
        coord._connection._pool = None  # type: ignore[attr-defined]
        await fake.aclose()  # type: ignore[attr-defined]


def _cell_token(cell: str) -> dp.Token[dp.CellKey]:
    return dp.mint_root(dp.CellKey(cell))


# ── insert into empty cell ───────────────────────────────────────────


class TestInsert:
    async def test_first_candidate_inserts(self, coord: dp.DataPlane) -> None:
        result = await coord.try_replace_elite(
            dp.CellKey("3,1,7"),
            dp.ProgramId("p-1"),
            token=_cell_token("3,1,7"),
            candidate_score=1.5,
            tiebreak_bit=0,
        )
        assert isinstance(result, dp.Ok)
        assert isinstance(result.value, dp.EliteInserted)

    async def test_inserted_value_visible_in_archive(self, coord: dp.DataPlane) -> None:
        await coord.try_replace_elite(
            dp.CellKey("cell-a"),
            dp.ProgramId("p-occ"),
            token=_cell_token("cell-a"),
            candidate_score=0.5,
            tiebreak_bit=0,
        )
        pool = coord._connection.pool  # type: ignore[attr-defined]
        occupant = await pool.hget("test:archive", "cell-a")  # type: ignore[misc]
        reverse = await pool.hget("test:archive:reverse", "p-occ")  # type: ignore[misc]
        score = await pool.hget("test:archive:scores", "cell-a")  # type: ignore[misc]
        assert occupant == "p-occ"
        assert reverse == "cell-a"
        assert float(score) == 0.5


# ── swap path ────────────────────────────────────────────────────────


class TestSwap:
    async def test_strictly_better_swaps(self, coord: dp.DataPlane) -> None:
        await coord.try_replace_elite(
            dp.CellKey("cell-x"),
            dp.ProgramId("p-low"),
            token=_cell_token("cell-x"),
            candidate_score=1.0,
            tiebreak_bit=0,
        )
        result = await coord.try_replace_elite(
            dp.CellKey("cell-x"),
            dp.ProgramId("p-high"),
            token=_cell_token("cell-x"),
            candidate_score=2.0,
            tiebreak_bit=0,
        )
        assert isinstance(result, dp.Ok)
        match result.value:
            case dp.EliteSwapped(displaced_id=loser):
                assert loser == "p-low"
            case _:
                pytest.fail(f"expected EliteSwapped, got {result.value!r}")

    async def test_swap_updates_reverse_index(self, coord: dp.DataPlane) -> None:
        await coord.try_replace_elite(
            dp.CellKey("cell-y"),
            dp.ProgramId("p-old"),
            token=_cell_token("cell-y"),
            candidate_score=1.0,
            tiebreak_bit=0,
        )
        await coord.try_replace_elite(
            dp.CellKey("cell-y"),
            dp.ProgramId("p-new"),
            token=_cell_token("cell-y"),
            candidate_score=2.0,
            tiebreak_bit=0,
        )
        pool = coord._connection.pool  # type: ignore[attr-defined]
        old_reverse = await pool.hget("test:archive:reverse", "p-old")  # type: ignore[misc]
        new_reverse = await pool.hget("test:archive:reverse", "p-new")  # type: ignore[misc]
        assert old_reverse is None
        assert new_reverse == "cell-y"


# ── reject path ──────────────────────────────────────────────────────


class TestReject:
    async def test_strictly_worse_rejected(self, coord: dp.DataPlane) -> None:
        await coord.try_replace_elite(
            dp.CellKey("cell-z"),
            dp.ProgramId("p-king"),
            token=_cell_token("cell-z"),
            candidate_score=10.0,
            tiebreak_bit=0,
        )
        result = await coord.try_replace_elite(
            dp.CellKey("cell-z"),
            dp.ProgramId("p-loser"),
            token=_cell_token("cell-z"),
            candidate_score=5.0,
            tiebreak_bit=0,
        )
        assert isinstance(result, dp.Ok)
        match result.value:
            case dp.EliteRejected(occupant_id=king):
                assert king == "p-king"
            case _:
                pytest.fail(f"expected EliteRejected, got {result.value!r}")

    async def test_reject_does_not_perturb_archive(self, coord: dp.DataPlane) -> None:
        await coord.try_replace_elite(
            dp.CellKey("c"),
            dp.ProgramId("incumbent"),
            token=_cell_token("c"),
            candidate_score=5.0,
            tiebreak_bit=0,
        )
        await coord.try_replace_elite(
            dp.CellKey("c"),
            dp.ProgramId("challenger"),
            token=_cell_token("c"),
            candidate_score=1.0,
            tiebreak_bit=0,
        )
        pool = coord._connection.pool  # type: ignore[attr-defined]
        assert (await pool.hget("test:archive", "c")) == "incumbent"  # type: ignore[misc]
        # The losing challenger must NOT appear in the reverse index.
        assert (
            await pool.hget("test:archive:reverse", "challenger")  # type: ignore[misc]
            is None
        )


# ── ties ─────────────────────────────────────────────────────────────


class TestTiebreak:
    async def test_tie_with_bit_one_swaps(self, coord: dp.DataPlane) -> None:
        await coord.try_replace_elite(
            dp.CellKey("t"),
            dp.ProgramId("first"),
            token=_cell_token("t"),
            candidate_score=1.5,
            tiebreak_bit=0,
        )
        result = await coord.try_replace_elite(
            dp.CellKey("t"),
            dp.ProgramId("second"),
            token=_cell_token("t"),
            candidate_score=1.5,
            tiebreak_bit=1,
        )
        assert isinstance(result, dp.Ok)
        assert isinstance(result.value, dp.EliteSwapped)

    async def test_tie_with_bit_zero_rejects(self, coord: dp.DataPlane) -> None:
        await coord.try_replace_elite(
            dp.CellKey("t"),
            dp.ProgramId("first"),
            token=_cell_token("t"),
            candidate_score=1.5,
            tiebreak_bit=0,
        )
        result = await coord.try_replace_elite(
            dp.CellKey("t"),
            dp.ProgramId("second"),
            token=_cell_token("t"),
            candidate_score=1.5,
            tiebreak_bit=0,
        )
        assert isinstance(result, dp.Ok)
        assert isinstance(result.value, dp.EliteRejected)


# ── cell isolation ───────────────────────────────────────────────────


class TestCellIsolation:
    async def test_independent_cells_isolated(self, coord: dp.DataPlane) -> None:
        await coord.try_replace_elite(
            dp.CellKey("a"),
            dp.ProgramId("p-a"),
            token=_cell_token("a"),
            candidate_score=1.0,
            tiebreak_bit=0,
        )
        await coord.try_replace_elite(
            dp.CellKey("b"),
            dp.ProgramId("p-b"),
            token=_cell_token("b"),
            candidate_score=2.0,
            tiebreak_bit=0,
        )
        pool = coord._connection.pool  # type: ignore[attr-defined]
        assert (await pool.hget("test:archive", "a")) == "p-a"  # type: ignore[misc]
        assert (await pool.hget("test:archive", "b")) == "p-b"  # type: ignore[misc]


# ── token discipline ─────────────────────────────────────────────────


class TestTokenDiscipline:
    async def test_token_cell_mismatch_returns_err(self, coord: dp.DataPlane) -> None:
        wrong_token = _cell_token("other-cell")
        result = await coord.try_replace_elite(
            dp.CellKey("expected-cell"),
            dp.ProgramId("p"),
            token=wrong_token,
            candidate_score=1.0,
            tiebreak_bit=0,
        )
        assert isinstance(result, dp.Err)

    async def test_token_consumed_on_call(self, coord: dp.DataPlane) -> None:
        token = _cell_token("consume-me")
        await coord.try_replace_elite(
            dp.CellKey("consume-me"),
            dp.ProgramId("p"),
            token=token,
            candidate_score=1.0,
            tiebreak_bit=0,
        )
        assert token.consumed


# ── concurrency ──────────────────────────────────────────────────────


class TestConcurrency:
    async def test_concurrent_writers_best_wins(self, coord: dp.DataPlane) -> None:
        """N coroutines racing for the same cell — the highest-score
        candidate must end up the occupant."""
        scores = [(f"p-{i}", float(i)) for i in range(20)]

        async def attempt(pid: str, score: float) -> None:
            await coord.try_replace_elite(
                dp.CellKey("hot-cell"),
                dp.ProgramId(pid),
                token=_cell_token("hot-cell"),
                candidate_score=score,
                tiebreak_bit=0,
            )

        await asyncio.gather(*(attempt(p, s) for p, s in scores))
        pool = coord._connection.pool  # type: ignore[attr-defined]
        occupant = await pool.hget("test:archive", "hot-cell")  # type: ignore[misc]
        score = await pool.hget("test:archive:scores", "hot-cell")  # type: ignore[misc]
        assert occupant == "p-19"  # highest score
        assert float(score) == 19.0
