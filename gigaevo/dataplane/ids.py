"""Strong-ID NewType taxonomy.

Every identifier the dataplane handles is a NewType over its underlying
representation so mypy rejects accidental confusion (passing a program_id
where event_id was expected, etc.). Zero runtime cost — at runtime each
alias is identical to its base type.

Conventions:
    - ULIDs are encoded as 26-char Crockford base32 strings.
    - Stream entry ids are Redis's native millisecond-sequence strings.
    - Counters (Epoch, Generation, Step) are int.
    - Composite ids use a ``{a}:{b}`` packed string.
"""

from __future__ import annotations

from typing import NewType

# ── aggregate identity ────────────────────────────────────────────────
ProgramId = NewType("ProgramId", str)
AggregateId = NewType("AggregateId", str)
RunId = NewType("RunId", str)
WorkerId = NewType("WorkerId", str)
NodeId = NewType("NodeId", int)

# ── event identity ────────────────────────────────────────────────────
EventId = NewType("EventId", str)
CausationId = NewType("CausationId", str)
CorrelationId = NewType("CorrelationId", str)

# ── monotonic counters ────────────────────────────────────────────────
StepId = NewType("StepId", int)
EpochId = NewType("EpochId", int)
GenerationId = NewType("GenerationId", int)

# ── actor / role / lease ──────────────────────────────────────────────
ActorId = NewType("ActorId", str)
LeaseToken = NewType("LeaseToken", str)

# ── stream coordination ───────────────────────────────────────────────
StreamName = NewType("StreamName", str)
ConsumerGroup = NewType("ConsumerGroup", str)
ConsumerName = NewType("ConsumerName", str)

# ── key namespacing ───────────────────────────────────────────────────
KeyPrefix = NewType("KeyPrefix", str)
CellKey = NewType("CellKey", str)
CounterKey = NewType("CounterKey", str)

# ── content addressing + idempotency ──────────────────────────────────
ContentHash = NewType("ContentHash", bytes)
IdempotencyToken = NewType("IdempotencyToken", str)

# ── domain ids ────────────────────────────────────────────────────────
BanditArm = NewType("BanditArm", str)

# ── scripts ───────────────────────────────────────────────────────────
ScriptName = NewType("ScriptName", str)


# ── helpers ───────────────────────────────────────────────────────────


def make_actor_id(run_id: RunId, worker_id: WorkerId) -> ActorId:
    """Compose an :data:`ActorId` from its parts.

    The single helper exists so the ``{run_id}:{worker_id}`` convention
    is not duplicated at every caller and any future change to the
    composition (e.g. adding a node component) lands in one place.
    """
    return ActorId(f"{run_id}:{worker_id}")


__all__ = [
    "ActorId",
    "AggregateId",
    "BanditArm",
    "CausationId",
    "CellKey",
    "ConsumerGroup",
    "ConsumerName",
    "ContentHash",
    "CorrelationId",
    "CounterKey",
    "EpochId",
    "EventId",
    "GenerationId",
    "IdempotencyToken",
    "KeyPrefix",
    "LeaseToken",
    "NodeId",
    "ProgramId",
    "RunId",
    "ScriptName",
    "StepId",
    "StreamName",
    "WorkerId",
    "make_actor_id",
]
