"""FSM transition tables.

State machines validated server-side in Lua. The tables are loaded into
Redis at ``DataPlane.startup`` so the transition Lua scripts can HGET
them; the same Python tables let callers fast-fail without a round-trip.

ProgramState mirrors ``gigaevo/programs/program_state.py``; parity is
enforced by a test in the application-level suite.
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
    """Task-claim lifecycle."""

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
    """Client-side legality check; the Lua script re-validates server-side.

    Generic over StrEnum so mypy rejects mixed-FSM lookups statically.
    """
    allowed = table.get(from_state)
    return allowed is not None and to_state in allowed


def encode_for_lua[S: StrEnum](table: dict[S, set[S]]) -> dict[str, str]:
    """Encode an FSM table as ``{from: "to1,to2,to3"}`` for ``HSET``.

    The Lua membership check walks comma-separated targets, so state
    values containing ``","`` are rejected at the boundary (a forged
    ``"X,Y"`` value would otherwise satisfy a check for either alone).

    Each row is emitted under both its declared key and the lowercase
    alias, and each target list carries both case variants — application
    callers that lowercase enum values share the FSM hash with
    coordinator-native uppercase callers.
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
        # Sort target variants for stable on-wire output (tests pin it).
        target_variants: set[str] = set()
        for t in allowed:
            target_variants.add(t.value)
            target_variants.add(t.value.lower())
        joined = ",".join(sorted(target_variants))
        encoded[from_state.value] = joined
        encoded[from_state.value.lower()] = joined
    return encoded


def fsm_key(key_prefix: str, name: str) -> str:
    """Return the Redis key under which ``name``'s FSM table is stored."""
    return f"{key_prefix}:fsm:{name}"


async def load_fsm_table[S: StrEnum](
    redis_client: aioredis.Redis,
    *,
    key_prefix: str,
    name: str,
    table: dict[S, set[S]],
) -> None:
    """Replace the FSM table at ``{key_prefix}:fsm:{name}`` with ``table``.

    DELETE + HSET in one transactional pipeline so removed states do not
    survive a schema change and so a concurrent reader cannot observe an
    empty key between the two writes. Idempotent under same-content runs.
    Generic over StrEnum so the table type matches the named FSM.
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
