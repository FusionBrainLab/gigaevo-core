"""Typed coordination plane for every Redis interaction.

This package generalises four Redis-coordination primitives — atomic
claim with TTL, renewable heartbeat lease, durable stream queue, and
idempotent write — into a single typed coordinator
(:class:`DataPlane`). Every state-changing operation is validated and
executed server-side via Lua scripts; every value carries a freshness
witness (:class:`Versioned`); every multi-writer hazard is gated by a
move-only permission token (:class:`Token`).

The foundation in this module is additive — type vocabulary, Lua
script registry, connection pool, coordinator shell. Method bodies,
Lua scripts, and call-site migrations land separately.

The reference for the Lua-CAS pattern used here is
:mod:`gigaevo.infra.endpoint_pool`; the :class:`LuaRegistry` mechanism
generalises that module's script-load / SHA-cache / NOSCRIPT-reload
behaviour.
"""

from __future__ import annotations

__version__ = "0.1.0"


from .codec import (
    compute_content_hash,
    compute_content_hash_hex,
    decode_canonical,
    encode_canonical,
)
from .coordinator import DataPlane
from .crash import CrashEvent, CrashWatchedHandle, OneShotFlag, Recovered
from .errors import (
    CanonicalEncodingError,
    ContentHashMismatchError,
    DataPlaneError,
    DeadlineExceeded,
    LockHeld,
    LockLost,
    NotStartedError,
    SchemaVersionMissingError,
    ScriptLostError,
    ScriptNotRegisteredError,
    ShutdownError,
    StaleReadError,
    StartupError,
    TokenAlreadyConsumed,
    TokenNotPickleable,
    TransitionError,
    UpcasterMissingError,
    all_error_types,
)
from .ids import (
    ActorId,
    AggregateId,
    BanditArm,
    CausationId,
    CellKey,
    ConsumerGroup,
    ConsumerName,
    ContentHash,
    CorrelationId,
    CounterKey,
    EpochId,
    EventId,
    GenerationId,
    IdempotencyToken,
    KeyPrefix,
    LeaseToken,
    NodeId,
    ProgramId,
    RunId,
    ScriptName,
    StepId,
    StreamName,
    WorkerId,
    make_actor_id,
)
from .lattices import (
    BoolLattice,
    EpochLattice,
    GenerationLattice,
    Lattice,
    MonotoneLattice,
    ProductLattice,
)
from .models import (
    CachedValue,
    Err,
    ExternalValue,
    GossipedValue,
    HlcTimestamp,
    LocalValue,
    Monotonic,
    Ok,
    ReplayedValue,
    Result,
    SanitizedValue,
    Sourced,
    Versioned,
)
from .permissions import Token, mint_combine, mint_root, mint_split, mint_split_n
from .transitions import (
    CLAIM_STATE_TRANSITIONS,
    LOCK_STATE_TRANSITIONS,
    PROGRAM_STATE_TRANSITIONS,
    ClaimState,
    LockState,
    ProgramState,
    encode_for_lua,
    fsm_key,
    is_valid_transition,
)

__all__ = [
    "__version__",
    # Coordinator
    "DataPlane",
    # Codec
    "compute_content_hash",
    "compute_content_hash_hex",
    "decode_canonical",
    "encode_canonical",
    # Errors
    "CanonicalEncodingError",
    "ContentHashMismatchError",
    "DataPlaneError",
    "DeadlineExceeded",
    "LockHeld",
    "LockLost",
    "NotStartedError",
    "SchemaVersionMissingError",
    "ScriptLostError",
    "ScriptNotRegisteredError",
    "ShutdownError",
    "StaleReadError",
    "StartupError",
    "TokenAlreadyConsumed",
    "TokenNotPickleable",
    "TransitionError",
    "UpcasterMissingError",
    "all_error_types",
    # IDs
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
    # Lattices
    "BoolLattice",
    "EpochLattice",
    "GenerationLattice",
    "Lattice",
    "MonotoneLattice",
    "ProductLattice",
    # Models
    "CachedValue",
    "Err",
    "ExternalValue",
    "GossipedValue",
    "HlcTimestamp",
    "LocalValue",
    "Monotonic",
    "Ok",
    "ReplayedValue",
    "Result",
    "SanitizedValue",
    "Sourced",
    "Versioned",
    # Permissions
    "Token",
    "mint_combine",
    "mint_root",
    "mint_split",
    "mint_split_n",
    # Transitions
    "CLAIM_STATE_TRANSITIONS",
    "LOCK_STATE_TRANSITIONS",
    "PROGRAM_STATE_TRANSITIONS",
    "ClaimState",
    "LockState",
    "ProgramState",
    "encode_for_lua",
    "fsm_key",
    "is_valid_transition",
    # Crash
    "CrashEvent",
    "CrashWatchedHandle",
    "OneShotFlag",
    "Recovered",
]
