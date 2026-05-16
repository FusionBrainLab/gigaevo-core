"""FSM transition tables.

State machines that the dataplane validates server-side in Lua. The
tables are loaded into Redis at ``DataPlane.startup`` so the
transition-checking Lua scripts can consult them via ``HGET``.

Mirrors of these tables stay in Python so callers can fast-fail at the
call site without a Redis round-trip — both layers reinforce each other.

The ProgramState table is the canonical mirror of
``gigaevo/programs/program_state.py``. Keeping the two in sync is
enforced by a parity test added with the program-FSM-API change.
"""

from __future__ import annotations

from enum import StrEnum

import redis.asyncio as aioredis

# ── ProgramState ──────────────────────────────────────────────────────


class ProgramState(StrEnum):
    """Mirror of :class:`gigaevo.programs.program_state.ProgramState`."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    DISCARDED = "DISCARDED"


PROGRAM_STATE_TRANSITIONS: dict[ProgramState, set[ProgramState]] = {
    ProgramState.QUEUED: {
        ProgramState.QUEUED,
        ProgramState.RUNNING,
        ProgramState.DISCARDED,
    },
    ProgramState.RUNNING: {
        ProgramState.RUNNING,
        ProgramState.DONE,
        ProgramState.DISCARDED,
    },
    ProgramState.DONE: {
        ProgramState.DONE,
        ProgramState.QUEUED,
        ProgramState.DISCARDED,
    },
    ProgramState.DISCARDED: {
        ProgramState.DISCARDED,
    },
}


# ── ClaimState ────────────────────────────────────────────────────────


class ClaimState(StrEnum):
    """Migration-bus / task-claim lifecycle."""

    UNCLAIMED = "UNCLAIMED"
    CLAIMED = "CLAIMED"
    ACKED = "ACKED"
    EXPIRED = "EXPIRED"


CLAIM_STATE_TRANSITIONS: dict[ClaimState, set[ClaimState]] = {
    ClaimState.UNCLAIMED: {ClaimState.CLAIMED},
    ClaimState.CLAIMED: {ClaimState.ACKED, ClaimState.EXPIRED},
    ClaimState.ACKED: set(),
    ClaimState.EXPIRED: {ClaimState.UNCLAIMED},
}


# ── LockState ─────────────────────────────────────────────────────────


class LockState(StrEnum):
    """Instance / role lease lifecycle.

    HELD and RENEWED are distinct only to surface "this lease has been
    refreshed at least once" in telemetry — the runtime transition is
    the same Lua call.
    """

    HELD = "HELD"
    RENEWED = "RENEWED"
    RELEASED = "RELEASED"
    LOST = "LOST"


LOCK_STATE_TRANSITIONS: dict[LockState, set[LockState]] = {
    LockState.HELD: {LockState.RENEWED, LockState.RELEASED, LockState.LOST},
    LockState.RENEWED: {LockState.RENEWED, LockState.RELEASED, LockState.LOST},
    LockState.RELEASED: set(),
    LockState.LOST: set(),
}


# ── helpers ───────────────────────────────────────────────────────────


def is_valid_transition(
    table: dict[StrEnum, set[StrEnum]],
    from_state: StrEnum,
    to_state: StrEnum,
) -> bool:
    """Client-side legality check. The Lua script re-validates server-side."""
    allowed = table.get(from_state)
    return allowed is not None and to_state in allowed


def encode_for_lua(table: dict[StrEnum, set[StrEnum]]) -> dict[str, str]:
    """Encode an FSM table as ``{from: "to1,to2,to3"}`` for ``HMSET``.

    The Lua side checks legality with a ``string.find`` on the
    comma-joined value. Comma is safe because all StrEnum values used
    by the dataplane are uppercase ASCII without commas.
    """
    return {
        from_state.value: ",".join(sorted(t.value for t in allowed))
        for from_state, allowed in table.items()
    }


def fsm_key(key_prefix: str, name: str) -> str:
    """Return the Redis key under which ``name``'s FSM table is stored.

    Prefixing prevents collisions when multiple runs share a Redis
    instance. The Lua scripts read from the same prefixed key so the
    convention is shared by both sides.
    """
    return f"{key_prefix}:fsm:{name}"


async def load_fsm_table(
    redis_client: aioredis.Redis,
    *,
    key_prefix: str,
    name: str,
    table: dict[StrEnum, set[StrEnum]],
) -> None:
    """HMSET the encoded table at ``{key_prefix}:fsm:{name}``.

    Called once per FSM during ``DataPlane.startup``. Idempotent —
    re-running overwrites with the same content. The redis client is
    passed directly rather than reaching into the coordinator's private
    pool.
    """
    encoded = encode_for_lua(table)
    if encoded:
        await redis_client.hset(fsm_key(key_prefix, name), mapping=encoded)  # type: ignore[misc]


__all__ = [
    "CLAIM_STATE_TRANSITIONS",
    "ClaimState",
    "LOCK_STATE_TRANSITIONS",
    "LockState",
    "PROGRAM_STATE_TRANSITIONS",
    "ProgramState",
    "encode_for_lua",
    "fsm_key",
    "is_valid_transition",
    "load_fsm_table",
]
