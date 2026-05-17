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
from typing import Final, NewType

# Reject control bytes that have special meaning to downstream consumers:
# NUL truncates strings in many clients; CR/LF enable log injection;
# C0 carries ANSI ESC; U+2028/U+2029 act as line terminators in JS;
# U+202E and friends hide trailing content via bidi override. The
# validator runs on :class:`ActorIdentity` only; bare :data:`ActorId`
# NewType has no runtime check, so callers should go through the
# typed dataclass.
_FORBIDDEN_CONTROL_CODEPOINTS: Final[frozenset[int]] = frozenset(
    # C0 controls (NUL/CR/LF/BEL/ESC/etc.) + DEL (0x7F).
    list(range(0x00, 0x20)) + [0x7F]
)
_FORBIDDEN_UNICODE_DIRECTIONALS: Final[frozenset[str]] = frozenset(
    {
        "‪",  # LEFT-TO-RIGHT EMBEDDING
        "‫",  # RIGHT-TO-LEFT EMBEDDING
        "‬",  # POP DIRECTIONAL FORMATTING
        "‭",  # LEFT-TO-RIGHT OVERRIDE
        "‮",  # RIGHT-TO-LEFT OVERRIDE
        "⁦",  # LEFT-TO-RIGHT ISOLATE
        "⁧",  # RIGHT-TO-LEFT ISOLATE
        "⁨",  # FIRST STRONG ISOLATE
        "⁩",  # POP DIRECTIONAL ISOLATE
        " ",  # LINE SEPARATOR
        " ",  # PARAGRAPH SEPARATOR
    }
)


def _validate_identity_part(field_name: str, value: str) -> None:
    """Reject control bytes, separators, and bidirectional overrides.

    Used by :class:`ActorIdentity` so a wire value carrying a hostile
    payload (NUL, CR/LF, ANSI ESC, RLO) cannot pass type-check and reach
    a log line, Redis key, or Postgres column verbatim.
    """
    if not value:
        raise ValueError(f"ActorIdentity.{field_name} is empty")
    if ":" in value:
        raise ValueError(
            f"ActorIdentity.{field_name} contains ':' — would break pack/parse "
            f"round-trip: {value!r}"
        )
    for ch in value:
        cp = ord(ch)
        if cp in _FORBIDDEN_CONTROL_CODEPOINTS:
            raise ValueError(
                f"ActorIdentity.{field_name} contains control character "
                f"U+{cp:04X} — would break wire formats / log lines: {value!r}"
            )
        if ch in _FORBIDDEN_UNICODE_DIRECTIONALS:
            raise ValueError(
                f"ActorIdentity.{field_name} contains bidirectional override "
                f"U+{cp:04X} — would hide trailing content in renders: {value!r}"
            )


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

    The ``{run_id}:{worker_id}`` shape is a wire-format detail; call
    sites pass :class:`ActorIdentity` so a missing separator cannot pass
    type-check. Validation rejects empty parts and parts containing
    ``":"`` so pack/parse is total.
    """

    run_id: RunId
    worker_id: WorkerId

    def __post_init__(self) -> None:
        _validate_identity_part("run_id", self.run_id)
        _validate_identity_part("worker_id", self.worker_id)

    def pack(self) -> ActorId:
        """Render the wire-format ``{run_id}:{worker_id}`` :data:`ActorId`."""
        return ActorId(f"{self.run_id}:{self.worker_id}")

    def __str__(self) -> str:
        return self.pack()

    @classmethod
    def parse(cls, actor_id: ActorId | str) -> ActorIdentity:
        """Inverse of :meth:`pack`; raises :class:`ValueError` on missing ``":"``."""
        run, sep, worker = actor_id.partition(":")
        if not sep:
            raise ValueError(f"ActorId missing ':' separator: {actor_id!r}")
        return cls(run_id=RunId(run), worker_id=WorkerId(worker))


# ── helpers ───────────────────────────────────────────────────────────


def make_actor_id(run_id: RunId, worker_id: WorkerId) -> ActorId:
    """Compose an :data:`ActorId` from its parts (wrapper over
    :meth:`ActorIdentity.pack`; new code should prefer the dataclass).
    """
    return ActorIdentity(run_id=run_id, worker_id=worker_id).pack()


_SCRIPT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def make_script_name(name: str) -> ScriptName:
    """Validate ``name`` (snake_case, ASCII, leading letter) and tag it."""
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
