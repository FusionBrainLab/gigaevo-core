"""Exhaustive coverage of every ProgramState ``(from, to)`` pair.

The per-pair unit suite at ``test_transition_state.py`` covers
representative happy/illegal cases. This file enumerates the full
4×4 Cartesian product of :class:`ProgramState` so a future change to
``PROGRAM_STATE_TRANSITIONS`` cannot silently allow a pair that the
table forbids (or forbid a pair that the table allows) without one of
these parametrised tests flipping colour.

Invariants exercised per pair:

- ``(from, to)`` in the table → coordinator returns :class:`Ok` and
  the program's persisted state advances to ``to``.
- ``(from, to)`` not in the table → coordinator returns
  ``Err(TransitionError(kind="illegal", ...))`` and the program's
  state stays at ``from``.

Run together with ``test_transition_state.py`` they pin both the
mechanism and the table.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
import json

import fakeredis.aioredis
import pytest

import gigaevo.dataplane as dp
from gigaevo.dataplane.coordinator import ProgramPatch
from gigaevo.dataplane.transitions import (
    PROGRAM_STATE_TRANSITIONS,
    ProgramState,
    load_fsm_table,
)


@pytest.fixture
async def coord() -> AsyncIterator[dp.DataPlane]:
    """One coordinator per test — fakeredis isolated, FSM table loaded."""
    server = fakeredis.FakeServer()
    coord = dp.DataPlane("redis://embedded/0", key_prefix="fsm-exh")
    fake = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    coord._connection._pool = fake  # type: ignore[attr-defined]
    from gigaevo.dataplane.scripts import LuaRegistry

    lua = LuaRegistry(fake)
    coord._register_builtin_scripts(lua)  # type: ignore[attr-defined]
    await lua.load_all()
    await load_fsm_table(
        fake,
        key_prefix="fsm-exh",
        name="program_state",
        table=PROGRAM_STATE_TRANSITIONS,
    )
    coord._lua = lua  # type: ignore[attr-defined]
    coord._started = True  # type: ignore[attr-defined]
    try:
        yield coord
    finally:
        coord._started = False  # type: ignore[attr-defined]
        coord._lua = None  # type: ignore[attr-defined]
        coord._connection._pool = None  # type: ignore[attr-defined]
        await fake.aclose()  # type: ignore[attr-defined]


async def _put_program(coord: dp.DataPlane, pid: str, state: ProgramState) -> None:
    blob = {"id": pid, "state": state.value, "epoch": 0}
    await coord._connection.pool.set(  # type: ignore[misc]
        f"fsm-exh:program:{pid}", json.dumps(blob)
    )


def _all_pairs() -> list[tuple[ProgramState, ProgramState, bool]]:
    """Return every ``(from, to, is_legal)`` triple over the 4×4 product."""
    out: list[tuple[ProgramState, ProgramState, bool]] = []
    for src in ProgramState:
        legal = PROGRAM_STATE_TRANSITIONS[src]
        for dst in ProgramState:
            out.append((src, dst, dst in legal))
    return out


_PAIRS = _all_pairs()


def _idfn(pair: tuple[ProgramState, ProgramState, bool]) -> str:
    src, dst, legal = pair
    return f"{src.value}_to_{dst.value}_{'legal' if legal else 'illegal'}"


class TestEveryPairMatchesTable:
    """For each (from, to), the coordinator's verdict equals the table's."""

    @pytest.mark.parametrize("pair", _PAIRS, ids=_idfn)
    async def test_pair(
        self,
        coord: dp.DataPlane,
        pair: tuple[ProgramState, ProgramState, bool],
    ) -> None:
        src, dst, legal = pair
        pid = f"p-{src.value}-{dst.value}"
        await _put_program(coord, pid, src)

        result = await coord.transition_program_state(
            dp.ProgramId(pid),
            token=dp.mint_root(dp.ProgramId(pid)),
            expected_from=src,
            to=dst,
            patch=ProgramPatch(),
        )

        if legal:
            assert isinstance(result, dp.Ok), (
                f"{src.value}->{dst.value} should be legal but returned {result!r}"
            )
            assert result.value.value["state"] == dst.value
        else:
            assert isinstance(result, dp.Err), (
                f"{src.value}->{dst.value} should be illegal but returned {result!r}"
            )
            assert result.error.kind == "illegal"
            # Program must still be in the original state.
            raw = await coord._connection.pool.get(  # type: ignore[misc]
                f"fsm-exh:program:{pid}"
            )
            assert json.loads(raw)["state"] == src.value


class TestExpectedFromAny:
    """``expected_from=None`` accepts any current state but still gates on legality."""

    @pytest.mark.parametrize("pair", _PAIRS, ids=_idfn)
    async def test_any_pair_with_expected_from_none(
        self,
        coord: dp.DataPlane,
        pair: tuple[ProgramState, ProgramState, bool],
    ) -> None:
        src, dst, legal = pair
        pid = f"any-{src.value}-{dst.value}"
        await _put_program(coord, pid, src)

        result = await coord.transition_program_state(
            dp.ProgramId(pid),
            token=dp.mint_root(dp.ProgramId(pid)),
            expected_from=None,
            to=dst,
            patch=ProgramPatch(),
        )
        if legal:
            assert isinstance(result, dp.Ok)
        else:
            assert isinstance(result, dp.Err)
            assert result.error.kind == "illegal"
