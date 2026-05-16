"""FSM transition tables.

State machines that the dataplane validates server-side in Lua. The
tables are loaded into Redis at ``DataPlane.startup`` so the
transition-checking Lua scripts can consult them via ``HGET``.

Mirrors of these tables stay in Python so callers can fast-fail at the
call site without a Redis round-trip — both layers reinforce each other.

The ProgramState table is the canonical mirror of
``gigaevo/programs/program_state.py``. Keeping the two in sync is
enforced by a parity test in the application-level test suite.
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


def is_valid_transition[S: StrEnum](
    table: dict[S, set[S]],
    from_state: S,
    to_state: S,
) -> bool:
    """Client-side legality check. The Lua script re-validates server-side.

    Generic over the StrEnum subtype so a ``ProgramState`` value cannot
    be looked up in the ``LockState`` table at the type level — mypy
    rejects mixed FSMs without any runtime cost.
    """
    allowed = table.get(from_state)
    return allowed is not None and to_state in allowed


def encode_for_lua[S: StrEnum](table: dict[S, set[S]]) -> dict[str, str]:
    """Encode an FSM table as ``{from: "to1,to2,to3"}`` for ``HSET``.

    The Lua side checks legality with a tokenised walk over the
    comma-joined value. State values must not contain ``","`` — a comma
    inside a value would split the encoded list at the wrong boundary
    and let a forged ``"X,Y"`` value satisfy a check for either ``X`` or
    ``Y`` alone. The encoder refuses such inputs at the call boundary
    so the Lua-side membership invariant stays inviolable.

    Each row is emitted under both its declared key and its lowercase
    alias; each row's target list carries both the declared form and
    the lowercase form of every target. A program blob persisted by an
    application-layer caller whose enum lowercases its ``.value`` (the
    pre-coordinator on-disk vocabulary) and a coordinator-native caller
    whose enum keeps the uppercase form share the same FSM hash and the
    same membership check. When a state's declared value already equals
    its lowercase form the duplicate write is a no-op overwrite.
    """
    encoded: dict[str, str] = {}
    for from_state, allowed in table.items():
        if "," in from_state.value:
            raise ValueError(
                f"FSM state value {from_state.value!r} contains ',' — "
                "comma is the on-wire separator and must be reserved"
            )
        for t in allowed:
            if "," in t.value:
                raise ValueError(
                    f"FSM state value {t.value!r} contains ',' — "
                    "comma is the on-wire separator and must be reserved"
                )
        # Build a target set that carries both case variants of every
        # member; the comma-list is then sorted for stable on-wire
        # output (tests pin the exact serialisation).
        target_variants: set[str] = set()
        for t in allowed:
            target_variants.add(t.value)
            target_variants.add(t.value.lower())
        joined = ",".join(sorted(target_variants))
        encoded[from_state.value] = joined
        encoded[from_state.value.lower()] = joined
    return encoded


def fsm_key(key_prefix: str, name: str) -> str:
    """Return the Redis key under which ``name``'s FSM table is stored.

    Prefixing prevents collisions when multiple runs share a Redis
    instance. The Lua scripts read from the same prefixed key so the
    convention is shared by both sides.
    """
    return f"{key_prefix}:fsm:{name}"


async def load_fsm_table[S: StrEnum](
    redis_client: aioredis.Redis,
    *,
    key_prefix: str,
    name: str,
    table: dict[S, set[S]],
) -> None:
    """Replace the FSM table at ``{key_prefix}:fsm:{name}`` with ``table``.

    Called once per FSM during ``DataPlane.startup``. The previous value
    (if any) is unconditionally deleted before the new mapping is
    written so removed states do not survive a schema change as stale
    HSET entries. The DELETE + HSET pair is issued as a single atomic
    pipeline so a concurrent reader cannot observe an empty key in the
    gap between the two writes.

    Idempotent under same-content re-runs. Generic over the StrEnum
    subtype so the caller cannot mix FSM tables (e.g. pass
    ``CLAIM_STATE_TRANSITIONS`` under the name ``"lock_state"`` with
    matching ``LockState`` annotations) by type-erasure accident.
    """
    encoded = encode_for_lua(table)
    key = fsm_key(key_prefix, name)
    pipe = redis_client.pipeline(transaction=True)
    pipe.delete(key)
    if encoded:
        pipe.hset(key, mapping=encoded)
    await pipe.execute()  # type: ignore[misc]


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
