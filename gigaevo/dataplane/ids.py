"""Strong-ID NewType taxonomy.

Every identifier the dataplane handles is a NewType over its underlying
representation so mypy rejects accidental confusion (passing a program_id
where event_id was expected, etc.). Zero runtime cost — at runtime each
alias is identical to its base type.

Conventions:
    - ULIDs are encoded as 26-char Crockford base32 strings.
    - Stream entry ids are Redis's native millisecond-sequence strings.
    - Counters (Epoch, Generation, Step) are int.
    - Composite ids surface as :class:`ActorIdentity` (typed dataclass)
      and pack to a ``{a}:{b}`` string only at wire boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
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


# ── ActorIdentity ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ActorIdentity:
    """Typed composite of ``(run_id, worker_id)`` that packs to ``ActorId``.

    The ``{run_id}:{worker_id}`` convention is a wire-format detail;
    Python call sites should pass :class:`ActorIdentity` so the parts
    are not stringly-typed and a typo (``"runworker"`` instead of
    ``"run:worker"``) cannot pass type-check. Use :meth:`pack` to obtain
    the wire-format :data:`ActorId` and :meth:`parse` to round-trip a
    received string back into a typed identity.

    Validation rejects empty parts and parts containing ``":"`` so the
    pack / parse round-trip is total: every constructible identity
    serialises to a unique string and parses back to the same identity.
    """

    run_id: RunId
    worker_id: WorkerId

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("ActorIdentity.run_id is empty")
        if not self.worker_id:
            raise ValueError("ActorIdentity.worker_id is empty")
        if ":" in self.run_id:
            raise ValueError(
                f"ActorIdentity.run_id contains ':' — would break pack/parse "
                f"round-trip: {self.run_id!r}"
            )
        if ":" in self.worker_id:
            raise ValueError(
                f"ActorIdentity.worker_id contains ':' — would break pack/parse "
                f"round-trip: {self.worker_id!r}"
            )

    def pack(self) -> ActorId:
        """Render the wire-format ``{run_id}:{worker_id}`` :data:`ActorId`."""
        return ActorId(f"{self.run_id}:{self.worker_id}")

    def __str__(self) -> str:
        return self.pack()

    @classmethod
    def parse(cls, actor_id: ActorId | str) -> ActorIdentity:
        """Inverse of :meth:`pack`.

        Splits on the *first* ``":"``; raises :class:`ValueError` if the
        input is missing the separator. Strict against the
        ``__post_init__`` contract so a malformed wire value never
        produces a malformed identity.
        """
        run, sep, worker = actor_id.partition(":")
        if not sep:
            raise ValueError(f"ActorId missing ':' separator: {actor_id!r}")
        return cls(run_id=RunId(run), worker_id=WorkerId(worker))


# ── helpers ───────────────────────────────────────────────────────────


def make_actor_id(run_id: RunId, worker_id: WorkerId) -> ActorId:
    """Compose an :data:`ActorId` from its parts.

    Convenience wrapper around :meth:`ActorIdentity.pack` for call sites
    that already work with raw NewType ids. New code should prefer
    :class:`ActorIdentity` directly — the typed dataclass surfaces the
    composition in the type system instead of hiding it behind a
    string-returning helper.
    """
    return ActorIdentity(run_id=run_id, worker_id=worker_id).pack()


_SCRIPT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def make_script_name(name: str) -> ScriptName:
    """Validate ``name`` and tag it as :data:`ScriptName`.

    Enforces a snake_case alphabet (lowercase ASCII letters, digits,
    underscores; leading char must be a letter) so the registry's
    logical names stay predictable in logs and Redis EVAL output.

    Raises :class:`ValueError` on any other shape — this is a startup-
    time error, not a runtime fault, so failing fast is the right thing.
    """
    if not _SCRIPT_NAME_RE.fullmatch(name):
        raise ValueError(
            f"ScriptName must match {_SCRIPT_NAME_RE.pattern!r} "
            f"(snake_case, leading letter): {name!r}"
        )
    return ScriptName(name)


__all__ = [
    "ActorId",
    "ActorIdentity",
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
    "make_script_name",
]
