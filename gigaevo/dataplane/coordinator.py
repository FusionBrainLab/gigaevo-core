"""The :class:`DataPlane` — sole public surface for every Redis interaction.

This module ships as a *shell* with explicit lifecycle methods and
per-resource method stubs that raise :class:`NotImplementedError`. The
public contract — argument shapes, return shapes, error variants — is
fully expressed in the type system today so call sites can be staged
against the real signatures while the method bodies are still landing.

The coordinator owns:

    - one :class:`RedisConnection` (the connection pool)
    - one :class:`LuaRegistry` (script SHAs)
    - the FSM tables loaded into Redis via
      :func:`gigaevo.dataplane.transitions.load_fsm_table`
    - the engine-root :class:`Token`\\ s (split per subspace)

The FSM transition tables are NOT copied into the coordinator — the
single source of truth lives in :mod:`gigaevo.dataplane.transitions` and
the coordinator references those module-level constants by name during
:meth:`startup`. A copy would create a second place that drifts; a
property accessor would imply the coordinator owned mutable state. The
tables are intentionally immutable singletons.

Outside this module, no other code should import ``redis`` or
``redis.asyncio`` directly — the ruff config enforces this (see
``lints.toml``).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import math
import secrets
from time import monotonic
from typing import Final, Literal

from loguru import logger

from .codec import compute_content_hash_hex, decode_canonical, encode_canonical
from .connection import RedisConnection
from .crash import CrashEvent, CrashWatchedHandle, OneShotFlag
from .errors import (
    DataPlaneError,
    DeadlineExceeded,
    EliteInvalidError,
    LockHeld,
    LockLost,
    NotStartedError,
    ShutdownError,
    StaleReadError,
    TransitionError,
)
from .ids import (
    ActorIdentity,
    CellKey,
    CounterKey,
    KeyPrefix,
    LeaseToken,
    ProgramId,
    ScriptName,
    make_script_name,
)
from .models import (
    Err,
    Freshness,
    FreshnessAtLeast,
    FreshnessEventual,
    FreshnessStrict,
    HlcTimestamp,
    LocalValue,
    Ok,
    Result,
    Sourced,
    Versioned,
)
from .permissions import Token
from .scripts import LuaRegistry, load_lua_source
from .transitions import (
    CLAIM_STATE_TRANSITIONS,
    LOCK_STATE_TRANSITIONS,
    PROGRAM_STATE_TRANSITIONS,
    ProgramState,
    fsm_key,
    load_fsm_table,
)

_DEFAULT_KEY_PREFIX: Final[str] = "gigaevo"

# Cryptographically-random lease token length (bytes). Yields a 22-char
# urlsafe base64 string, comfortably uniqueness-safe for the lease
# lifetime even under millions of holders.
_LEASE_TOKEN_BYTES: Final[int] = 16

# Renewal cadence relative to TTL. A 1/3 ratio gives two safety
# attempts before the TTL would expire under perfect timing, with room
# for GC pauses or transient network hiccups.
_LOCK_RENEW_RATIO: Final[float] = 3.0

# Cap on per-batch item count. The batch is not atomic across items
# (see :meth:`DataPlane.transition_program_state_batch`); accepting a
# pathologically large batch would build a multi-megabyte outcome list
# in memory while holding no global progress invariant. The cap is loose
# enough for legitimate fan-out and tight enough to surface accidental
# unbounded batches at the boundary.
_BATCH_ITEM_CAP: Final[int] = 1024

# Control characters that are unsafe in any Redis key. NUL truncates
# the key on the RESP wire; CR / LF break framing for clients that
# share a connection with another protocol. Reject these universally
# regardless of whether the caller-supplied identifier is a single
# component or a hierarchical key path.
_FORBIDDEN_KEY_CONTROL_CHARS: Final[frozenset[str]] = frozenset({"\x00", "\r", "\n"})

# Additional separator forbidden for *atomic* (single-component) IDs
# spliced into ``{prefix}:{resource}:{id}`` patterns. Hierarchical keys
# (CounterKey, LWW register names) legitimately carry colons as
# sub-namespace separators and must opt out of this check.
_FORBIDDEN_ATOMIC_ID_SEPARATOR: Final[str] = ":"


# ── built-in script names ──────────────────────────────────────────────
#
# Each per-resource Lua script registers under a stable name. The names
# are fixed module constants so callers (and tests) refer to one symbol
# instead of a magic string repeated at every site.

_SCRIPT_COUNTER_INC: Final[ScriptName] = make_script_name("counter_inc")
_SCRIPT_LOCK_ACQUIRE: Final[ScriptName] = make_script_name("instance_lock_acquire")
_SCRIPT_LOCK_RENEW: Final[ScriptName] = make_script_name("instance_lock_renew")
_SCRIPT_LOCK_RELEASE: Final[ScriptName] = make_script_name("instance_lock_release")
_SCRIPT_LWWR_SET: Final[ScriptName] = make_script_name("lwwr_set")
_SCRIPT_TRANSITION_STATE: Final[ScriptName] = make_script_name("transition_state")
_SCRIPT_ARCHIVE_SWAP: Final[ScriptName] = make_script_name("archive_swap")
_SCRIPT_BOUNDED_LIST_PUSH: Final[ScriptName] = make_script_name("bounded_list_push")


# ── public contract dataclasses ─────────────────────────────────────────
#
# Every method on :class:`DataPlane` returns a typed shape. The
# dataclasses below are the public contract: callers can write the
# pattern-match arms today even though the method bodies still raise
# ``NotImplementedError``. Each is frozen + slotted so the wire / cache
# representation cannot drift.


@dataclass(slots=True, frozen=True)
class ProgramPatch:
    """Field-level patch applied during a program FSM transition.

    A ``ProgramPatch`` is a JSON-serialisable mapping that the server
    merges into the persisted program blob *before* it advances the
    state. The frozen-dataclass shape rather than a bare ``dict`` exists
    so a) the caller cannot mutate the patch after construction (which
    would race the in-flight Lua call), b) future fields (e.g. a JSON
    pointer / a content-hash floor) can be added without rewriting every
    call site, and c) tests can assert against an exact value.

    The ``fields`` payload is constrained to JSON-serialisable values at
    the codec boundary; this dataclass does not re-validate the shape.
    """

    fields: dict[str, object] = field(default_factory=dict)


# A placeholder for the future :class:`gigaevo.programs.program.Program`
# concrete type. The coordinator's Lua scripts treat the blob opaquely:
# they merge ``ProgramPatch.fields`` into the persisted JSON and advance
# the state. The Python-side shape is whatever the application layer
# decodes the blob into. When the Program module lands its concrete
# dataclass it will replace this alias in one edit.
type ProgramSnapshot = dict[str, object]


@dataclass(slots=True, frozen=True)
class BatchTransitionItem:
    """One element of a multi-program atomic transition batch.

    The server applies the whole batch as a single Lua call: either
    every transition succeeds or none do. Each item carries its own
    token because the underlying subspace is per-program — a single
    coordinator-root token cannot serve a multi-aggregate write.
    """

    program_id: ProgramId
    token: Token[ProgramId]
    expected_from: ProgramState | None
    to: ProgramState
    patch: ProgramPatch | None = None


@dataclass(slots=True, frozen=True)
class BatchTransitionOutcome:
    """Per-item outcome of a batch transition.

    ``items`` is the same length as the input batch and in the same
    order. Each entry is the post-transition :class:`Versioned` snapshot
    for the program; a per-item ``Err`` cannot occur because the batch
    is atomic at the server, so a partial failure surfaces as an
    :class:`Err` on the outer :data:`Result` instead.
    """

    items: tuple[Versioned[ProgramSnapshot], ...]


@dataclass(slots=True, frozen=True)
class InstanceLease:
    """A live, TTL-bounded lease over a key prefix.

    Returned from :meth:`DataPlane.acquire_instance_lock`. The
    ``token`` field is the opaque server-minted lease id; renew /
    release calls present the same token so the server can token-CAS
    and surface :class:`LockLost` if a different holder has taken over.

    ``expires_at_monotonic`` is captured at acquisition time from
    ``time.monotonic()`` plus ``ttl_s`` so a watchdog can detect a
    near-expiry condition without a Redis round-trip. ``flag`` is a
    one-shot signal owned by the holder; the recovery path (Sentinel
    failover, lease lost) sets it.
    """

    token: LeaseToken
    key: str
    ttl_s: float
    expires_at_monotonic: float
    flag: OneShotFlag = field(compare=False, repr=False)


@dataclass(slots=True, frozen=True)
class EliteInserted:
    """The cell was empty; the candidate became the elite."""

    kind: Literal["inserted"] = "inserted"


@dataclass(slots=True, frozen=True)
class EliteSwapped:
    """The candidate displaced the prior elite. ``displaced_id`` carries the loser."""

    displaced_id: ProgramId
    kind: Literal["swapped"] = "swapped"


@dataclass(slots=True, frozen=True)
class EliteRejected:
    """The candidate lost the comparison. ``occupant_id`` carries the survivor."""

    occupant_id: ProgramId
    kind: Literal["rejected"] = "rejected"


@dataclass(slots=True, frozen=True)
class LwwrValue:
    """A LWW-register's stored value alongside its HLC witness.

    Returned by :meth:`DataPlane.lwwr_get`. The ``value`` field is the
    caller's payload after canonical-JSON round-trip; the ``hlc`` field
    is the witness that proves causal ordering against other writers.
    Two LwwrValue instances compare equal when they hold identical
    value and HLC — useful for "did our write actually land" assertions.
    """

    value: object
    hlc: HlcTimestamp


type LwwrSetOutcome = Literal["replaced", "kept"]
"""Outcome of :meth:`DataPlane.lwwr_set`.

``replaced`` — the candidate HLC was strictly newer than the stored
HLC, so the stored value is now the candidate's.

``kept`` — the candidate HLC was older or equal; the stored value is
unchanged. The caller's write was silently rejected. This is the
normal eventual-consistency outcome for an out-of-order arrival; it
is **not** an error.
"""


type EliteSwapOutcome = EliteInserted | EliteSwapped | EliteRejected
"""Discriminated outcome of :meth:`DataPlane.try_replace_elite`.

Pattern-match on the variant::

    match outcome:
        case EliteInserted():
            ...
        case EliteSwapped(displaced_id=loser_id):
            ...
        case EliteRejected(occupant_id=winner_id):
            ...

The ``kind`` Literal field exists so a JSON / msgpack codec can
round-trip the variant tag without runtime reflection on the dataclass
class.
"""


class DataPlane:
    """The single object that talks to Redis.

    Constructed once per process. Wired through Hydra config (engine
    startup) or directly in tests. Every module that talks to Redis
    routes through this class's methods rather than building its own
    connection — see ``lints.toml`` for the policy.
    """

    def __init__(
        self,
        redis_url: str,
        *,
        key_prefix: str = _DEFAULT_KEY_PREFIX,
        max_connections: int = 64,
        socket_timeout_s: float = 30.0,
        socket_connect_timeout_s: float = 10.0,
    ) -> None:
        self._connection = RedisConnection(
            redis_url,
            key_prefix=key_prefix,
            max_connections=max_connections,
            socket_timeout_s=socket_timeout_s,
            socket_connect_timeout_s=socket_connect_timeout_s,
        )
        self._lua: LuaRegistry | None = None
        self._started: bool = False
        # Per-lease background renewers keyed by lease token. Populated
        # by :meth:`acquire_instance_lock`, drained by
        # :meth:`release_instance_lock` and :meth:`shutdown`.
        self._renewers: dict[LeaseToken, asyncio.Task[None]] = {}
        # Serialises the whole startup / shutdown sequence so two racing
        # callers cannot torn-init the coordinator's own ``_lua`` and
        # ``_started`` fields. The underlying :class:`RedisConnection`
        # has its own lifecycle lock for the pool itself; this lock
        # protects the coordinator-level state that wraps it. Lazily
        # built inside the running event loop so construction stays
        # cheap and loop-agnostic.
        self._lifecycle_lock: asyncio.Lock | None = None

    # ── lifecycle ────────────────────────────────────────────────────

    def _get_lifecycle_lock(self) -> asyncio.Lock:
        """Lazy-init the coordinator-level lifecycle lock inside the loop."""
        if self._lifecycle_lock is None:
            self._lifecycle_lock = asyncio.Lock()
        return self._lifecycle_lock

    async def startup(self) -> None:
        """Initialise the connection pool, load scripts, prime FSM tables.

        Idempotent: re-calling on a started coordinator is a no-op.
        Serialised: concurrent callers wait on a single lock so the
        pool-start / script-load / FSM-load sequence happens once,
        atomically. On partial failure (e.g. FSM table load fails) the
        connection pool is closed and the lua-registry handle is dropped
        before the exception propagates so the coordinator is left in
        the pre-startup state — the caller can retry without leaking
        sockets or holding a half-initialised registry.
        """
        # Cheap pre-check avoids contending on the lock when the
        # coordinator is already up. Re-checked under the lock below.
        if self._started:
            return
        async with self._get_lifecycle_lock():
            if self._started:
                return
            await self._connection.startup()
            try:
                lua = LuaRegistry(self._connection.pool)
                self._register_builtin_scripts(lua)
                await lua.load_all()
                for fsm_name, table in (
                    ("program_state", PROGRAM_STATE_TRANSITIONS),
                    ("claim_state", CLAIM_STATE_TRANSITIONS),
                    ("lock_state", LOCK_STATE_TRANSITIONS),
                ):
                    await load_fsm_table(
                        self._connection.pool,
                        key_prefix=self._connection.key_prefix,
                        name=fsm_name,
                        table=table,
                    )
            except BaseException:  # noqa: BLE001 - startup rollback, error re-raised verbatim
                # ``BaseException`` so a cancellation mid-FSM-load still
                # rolls the pool back. ``self._lua`` is reset to ``None``
                # so the invariant ``self._started == (self._lua is not
                # None)`` holds even after a partial script load
                # populated the local ``lua`` variable above.
                await self._connection.shutdown()
                self._lua = None
                raise
            self._lua = lua
            self._started = True
        logger.info("DataPlane started: prefix={}", self._connection.key_prefix)

    async def shutdown(self) -> None:
        """Release background tasks, close the connection pool.

        Idempotent and serialised against :meth:`startup` via the
        coordinator-level lifecycle lock. The pool is closed even if
        renewer cancellation surfaced an exception; that exception is
        re-wrapped into a :class:`ShutdownError` after the pool teardown
        has completed so no socket survives a noisy shutdown.
        """
        if not self._started:
            return
        async with self._get_lifecycle_lock():
            if not self._started:
                return
            cancel_exc: BaseException | None = None
            try:
                await self._cancel_renewers()
            except BaseException as exc:  # noqa: BLE001 - shutdown deferral, surfaced below
                # Defer the failure until after the pool is closed; a
                # renewer-cancel error must not leak a live connection
                # pool. Re-raised below wrapped in ShutdownError.
                cancel_exc = exc
            close_exc: BaseException | None = None
            try:
                await self._connection.shutdown()
            except BaseException as exc:  # noqa: BLE001 - shutdown boundary, wrapped below
                close_exc = exc
            self._lua = None
            self._started = False
        if cancel_exc is not None or close_exc is not None:
            primary = cancel_exc if cancel_exc is not None else close_exc
            raise ShutdownError(reason=repr(primary)) from primary

    @property
    def started(self) -> bool:
        return self._started

    @property
    def key_prefix(self) -> str:
        return self._connection.key_prefix

    # ── internals ────────────────────────────────────────────────────

    def _require_started(self, method: str) -> LuaRegistry:
        """Return the LuaRegistry or raise :class:`NotStartedError`.

        Every state-access method body that lands in subsequent changes
        must call this guard first; the ``method`` argument flows into
        the error's diagnostic field so a misbehaving caller is
        identified without a traceback walk.
        """
        lua = self._lua
        if lua is None:
            raise NotStartedError(method=method)
        return lua

    @staticmethod
    def _validate_key_component(
        value: str, *, method: str, field_name: str, allow_colon: bool = False
    ) -> None:
        """Reject identifiers that would break Redis key namespacing.

        The coordinator splices caller-supplied identifiers into Redis
        keys verbatim. NUL would silently truncate the key on the wire,
        CR / LF would corrupt RESP framing, and colon collides with the
        dataplane's own ``{prefix}:{resource}:{id}`` convention for
        atomic IDs. Hierarchical key paths (CounterKey, LWW register
        names) opt out of the colon check via ``allow_colon=True``.
        """
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"{method}: {field_name} must be a non-empty string, got {value!r}"
            )
        for ch in _FORBIDDEN_KEY_CONTROL_CHARS:
            if ch in value:
                raise ValueError(
                    f"{method}: {field_name} contains forbidden control "
                    f"character {ch!r}: {value!r}"
                )
        if not allow_colon and _FORBIDDEN_ATOMIC_ID_SEPARATOR in value:
            raise ValueError(
                f"{method}: {field_name} contains ':' which collides with the "
                f"dataplane key-namespacing convention: {value!r}"
            )

    @staticmethod
    def _validate_ttl(ttl_s: float, *, method: str) -> None:
        """Reject NaN / Inf / non-positive TTLs at the call boundary.

        ``int(NaN * 1000)`` raises ``ValueError`` deep in the renew path;
        we'd rather surface that as a clean ``ValueError`` at the entry
        point with the offending value attached. Non-positive TTLs would
        round to zero milliseconds and produce a lock that's immediately
        expired — refuse them up front.
        """
        if not isinstance(ttl_s, (int, float)) or isinstance(ttl_s, bool):
            raise ValueError(
                f"{method}: ttl_s must be a real number, got {type(ttl_s).__name__}"
            )
        if math.isnan(ttl_s) or math.isinf(ttl_s):
            raise ValueError(f"{method}: ttl_s must be finite, got {ttl_s!r}")
        if ttl_s <= 0:
            raise ValueError(f"{method}: ttl_s must be positive, got {ttl_s!r}")

    @staticmethod
    def _validate_floor(min_epoch: int, min_generation: int, *, method: str) -> None:
        """Reject negative freshness floors at the call boundary."""
        if min_epoch < 0:
            raise ValueError(
                f"{method}: min_epoch must be non-negative, got {min_epoch!r}"
            )
        if min_generation < 0:
            raise ValueError(
                f"{method}: min_generation must be non-negative, got {min_generation!r}"
            )

    def _check_deadline(self, deadline_monotonic: float | None, method: str) -> None:
        """Surface a typed :class:`DeadlineExceeded` if the budget is gone.

        Called at the top of every public method that accepts a
        deadline; we'd rather fail fast at the boundary than start a
        Redis call we know we cannot afford to finish.
        """
        if deadline_monotonic is None:
            return
        now = monotonic()
        if now >= deadline_monotonic:
            raise DeadlineExceeded(elapsed_s=now - deadline_monotonic, budget_s=0.0)

    # ── program FSM ──────────────────────────────────────────────────

    async def transition_program_state(
        self,
        program_id: ProgramId,
        *,
        token: Token[ProgramId],
        expected_from: ProgramState | None,
        to: ProgramState,
        patch: ProgramPatch | None = None,
        deadline_monotonic: float | None = None,
    ) -> Result[Versioned[ProgramSnapshot], DataPlaneError]:
        """Validated FSM transition for a program; one atomic Lua call.

        ``token`` is consumed on entry; the caller must mint a fresh
        token (via :func:`mint_split` from the engine-root token) per
        call. ``expected_from=None`` means any current state is
        acceptable; the FSM table still validates ``(from, to)``.
        ``patch`` is a typed :class:`ProgramPatch` merged into the
        persisted program blob before the state advance.

        Idempotency is derived: the wrapper hashes
        ``(pid, expected_from, to, patch)`` and the Lua script dedups on
        that hash for 5 minutes, so a network retry of the same logical
        call returns the previous outcome unchanged.

        Result variants:
            Ok(Versioned[ProgramSnapshot]) — transition applied (or the
                same logical call was retried and the prior outcome is
                being replayed); the value is the post-transition blob,
                ``epoch`` and ``generation`` are the global counter.
            Err(TransitionError.stale)   — program blob absent or
                ``expected_from`` mismatched the observed state.
            Err(TransitionError.illegal) — FSM table rejects (from, to).
            Err(TransitionError.unknown) — token-tag mismatch (caller
                bug) or unrecognised Lua status.
        """
        self._validate_key_component(
            program_id, method="transition_program_state", field_name="program_id"
        )
        try:
            self._check_deadline(deadline_monotonic, "transition_program_state")
        except DeadlineExceeded as exc:
            return Err(exc)
        lua = self._require_started("transition_program_state")
        tag = token.consume()
        if tag != program_id:
            return Err(
                TransitionError.unknown(
                    "token-tag-mismatch",
                    f"token tag {tag!r} does not match program_id {program_id!r}",
                )
            )
        prefix = self._connection.key_prefix
        program_key = f"{prefix}:program:{program_id}"
        events_stream = f"{prefix}:status_events"
        epoch_key = f"{prefix}:ts"
        fsm_table = fsm_key(prefix, "program_state")
        idem_hash = f"{prefix}:idem:program:{program_id}"
        patch_dict = patch.fields if patch is not None else {}
        patch_json = encode_canonical(patch_dict).decode("utf-8") if patch_dict else ""
        idempotency_tok = compute_content_hash_hex(
            {
                "pid": program_id,
                "from": expected_from.value if expected_from else "",
                "to": to.value,
                "patch": patch_dict,
            },
            schema_version=1,
        )
        try:
            raw = await lua.evalsha(
                _SCRIPT_TRANSITION_STATE,
                keys=[
                    program_key,
                    events_stream,
                    epoch_key,
                    fsm_table,
                    idem_hash,
                ],
                args=[
                    program_id,
                    expected_from.value if expected_from else "",
                    to.value,
                    patch_json,
                    idempotency_tok,
                    prefix,
                ],
            )
        except DataPlaneError as exc:
            return Err(exc)
        status, epoch_s, payload = raw
        epoch = int(epoch_s)
        if status in ("ok", "duplicate"):
            blob = decode_canonical(payload)
            return Ok(Versioned(value=blob, epoch=epoch, generation=epoch))
        if status == "stale":
            return Err(TransitionError.stale(payload))
        if status == "illegal":
            return Err(TransitionError.illegal(payload))
        if status == "invalid":
            return Err(TransitionError.invalid(payload))
        return Err(TransitionError.unknown(status, payload))

    async def transition_program_state_batch(
        self,
        items: tuple[BatchTransitionItem, ...],
        *,
        deadline_monotonic: float | None = None,
    ) -> Result[BatchTransitionOutcome, DataPlaneError]:
        """Apply a batch of program-FSM transitions.

        Each item is dispatched as an independent :meth:`transition_program_state`
        call; per-item atomicity is preserved (the underlying Lua
        script handles each one), but **the batch as a whole is not
        atomic**. If item *k* fails the preceding *k-1* items are
        already applied; the failure surfaces as the outer ``Err`` and
        the caller must compensate for the partial commit (or accept it).

        The order of :class:`BatchTransitionOutcome.items` matches the
        input ``items``. Each item's :class:`Token` is consumed on its
        own call — passing the same token across multiple items would
        raise :class:`TokenAlreadyConsumed` on the second use.

        Returns ``Ok(BatchTransitionOutcome(items=()))`` for an empty
        batch (vacuous success).
        """
        if not items:
            return Ok(BatchTransitionOutcome(items=()))
        if len(items) > _BATCH_ITEM_CAP:
            return Err(
                DataPlaneError(
                    f"transition_program_state_batch: item count {len(items)} "
                    f"exceeds cap {_BATCH_ITEM_CAP}"
                )
            )
        outcomes: list[Versioned[ProgramSnapshot]] = []
        for item in items:
            # Re-check the shared deadline before each per-item call so
            # a long batch surfaces :class:`DeadlineExceeded` after the
            # last successfully-committed item instead of pressing on
            # against a budget that's already gone.
            try:
                self._check_deadline(
                    deadline_monotonic, "transition_program_state_batch"
                )
            except DeadlineExceeded as exc:
                return Err(exc)
            result = await self.transition_program_state(
                item.program_id,
                token=item.token,
                expected_from=item.expected_from,
                to=item.to,
                patch=item.patch,
                deadline_monotonic=deadline_monotonic,
            )
            if isinstance(result, Err):
                return result
            outcomes.append(result.value)
        return Ok(BatchTransitionOutcome(items=tuple(outcomes)))

    async def read_program(
        self,
        program_id: ProgramId,
        *,
        freshness: Freshness | None = None,
        min_epoch: int = 0,
        min_generation: int = 0,
    ) -> Result[LocalValue[Versioned[ProgramSnapshot]] | None, DataPlaneError]:
        """Versioned program read with an explicit freshness declaration.

        ``freshness`` is the structural admission contract every reader
        must state. The variants are:

            * :class:`FreshnessEventual` — accept any persisted blob;
              the read never raises :class:`StaleReadError` on the
              freshness axis. The default; matches the legacy
              ``min_epoch=0, min_generation=0`` behaviour.
            * :class:`FreshnessAtLeast` — admission floor on the
              ``(epoch, generation)`` lattice. A blob below the floor
              raises :class:`StaleReadError` so a caller observing its
              own write (or a downstream replica's catch-up) fails loud
              rather than silently old.
            * :class:`FreshnessStrict` — re-read the global epoch
              counter and require the blob's epoch to match-or-exceed
              that snapshot. Costs one extra round-trip; use it only
              when the caller cannot supply a floor itself.

        ``min_epoch`` / ``min_generation`` remain accepted as a
        backwards-compat shim so existing callers keep working; a
        non-zero value (with ``freshness`` unset) constructs a
        :class:`FreshnessAtLeast` internally. Passing both an explicit
        ``freshness`` and a non-zero ``min_*`` raises
        :class:`ValueError` so a typo cannot mask one with the other.

        The successful return is wrapped in :data:`LocalValue` —
        the :class:`Sourced` phantom-tag alias for a fresh local read
        out of Redis. Cache-fronted readers, gossip-propagated readers,
        and replay loops carry their own provenance tags so a single
        ``rg "LocalValue\\|CachedValue\\|ReplayedValue"`` over the
        dataplane consumers surfaces every read-source distinction at
        the static-type level (bug class #13 — "stale cache returns
        as fresh authoritative" — is now a phantom-type discrimination
        at the function-signature boundary).

        Returns ``Ok(None)`` if the program is unknown (not yet
        written); ``Ok(LocalValue(Versioned(...)))`` if it exists and
        passes the freshness contract; ``Err(...)`` for decoding
        failures or a freshness-floor violation.
        """
        self._validate_key_component(
            program_id, method="read_program", field_name="program_id"
        )
        # Negative ``min_*`` raises :class:`ValueError` at the boundary
        # per the legacy contract (callers never expected this to return
        # an ``Err``). Conflicts between an explicit ``freshness`` and a
        # legacy ``min_*`` value still surface as ``Err`` because they
        # represent a genuine programming ambiguity rather than a
        # malformed numeric input.
        self._validate_floor(min_epoch, min_generation, method="read_program")
        try:
            effective_freshness = self._resolve_freshness(
                freshness, min_epoch, min_generation, method="read_program"
            )
        except ValueError as exc:
            return Err(DataPlaneError(str(exc)))
        self._require_started("read_program")
        prefix = self._connection.key_prefix
        program_key = f"{prefix}:program:{program_id}"

        # FreshnessStrict reads the live epoch counter first so the
        # subsequent blob GET admits only values stamped at or after
        # that snapshot. The two GETs are not transactional — a
        # concurrent transition between them is the exact race that
        # Strict catches: the blob's epoch will be the post-transition
        # value, which is >= our snapshot, so the read succeeds; the
        # earlier blob is invisible. The race that *fails* the floor is
        # the inverse — counter ahead of blob — which on a single-writer
        # engine cannot happen, but is the case Strict is designed for
        # cross-engine reads.
        floor_epoch: int
        floor_generation: int
        if isinstance(effective_freshness, FreshnessStrict):
            epoch_key = f"{prefix}:ts"
            raw_counter = await self._connection.pool.get(epoch_key)  # type: ignore[misc]
            try:
                floor_epoch = int(raw_counter) if raw_counter is not None else 0
            except (TypeError, ValueError) as exc:
                return Err(
                    DataPlaneError(
                        f"read_program: strict-freshness epoch counter "
                        f"unparseable: {exc!r}"
                    )
                )
            floor_generation = floor_epoch
        elif isinstance(effective_freshness, FreshnessAtLeast):
            floor_epoch = effective_freshness.epoch
            floor_generation = effective_freshness.generation
        else:
            floor_epoch = 0
            floor_generation = 0

        raw = await self._connection.pool.get(program_key)  # type: ignore[misc]
        if raw is None:
            return Ok(None)
        try:
            blob = decode_canonical(raw)
        except Exception as exc:  # noqa: BLE001 - coordinator boundary
            return Err(DataPlaneError(f"read_program: decode failed: {exc!r}"))
        if not isinstance(blob, dict):
            return Err(
                DataPlaneError(
                    f"read_program: decoded blob has wrong shape "
                    f"({type(blob).__name__}); expected dict"
                )
            )
        try:
            epoch = int(blob.get("epoch", 0))
        except (TypeError, ValueError) as exc:
            return Err(
                DataPlaneError(f"read_program: epoch field unparseable: {exc!r}")
            )
        versioned: Versioned[ProgramSnapshot] = Versioned(
            value=blob, epoch=epoch, generation=epoch
        )
        if not versioned.is_at_least(floor_epoch, floor_generation):
            return Err(
                StaleReadError(
                    observed_epoch=epoch,
                    observed_generation=epoch,
                    min_epoch=floor_epoch,
                    min_generation=floor_generation,
                )
            )
        # Wrap in :data:`LocalValue` — the freshness check has cleared
        # against the coordinator's own pool, so the provenance is
        # structurally local-fresh-read. Cache layers in front of the
        # coordinator (none today) would re-tag to ``CachedValue``;
        # gossip / replay paths would carry their own discriminator.
        local: LocalValue[Versioned[ProgramSnapshot]] = Sourced(value=versioned)
        return Ok(local)

    @staticmethod
    def _resolve_freshness(
        freshness: Freshness | None,
        min_epoch: int,
        min_generation: int,
        *,
        method: str = "read_program",
    ) -> Freshness:
        """Reconcile the new ``freshness`` arg with the legacy ``min_*`` shim.

        Resolution rules:

            * ``freshness`` explicit and non-default ``min_*`` ⇒
              :class:`ValueError`. The two channels would otherwise
              disagree silently.
            * ``freshness`` explicit ⇒ used verbatim.
            * ``freshness=None`` and ``min_*`` zero ⇒
              :class:`FreshnessEventual` (the default).
            * ``freshness=None`` and ``min_*`` non-zero ⇒
              :class:`FreshnessAtLeast(epoch, generation)` constructed
              from the kwargs (legacy compatibility).

        Negative ``min_*`` is rejected by the call-boundary
        :meth:`_validate_floor` before reaching this resolver.

        ``method`` is threaded into the ambiguity error so a caller that
        passes both channels sees which API they used in the message.
        """
        if freshness is not None:
            if min_epoch != 0 or min_generation != 0:
                raise ValueError(
                    f"{method}: pass either freshness= or "
                    "min_epoch=/min_generation= (legacy shim), not both"
                )
            return freshness
        if min_epoch == 0 and min_generation == 0:
            return FreshnessEventual()
        return FreshnessAtLeast(epoch=min_epoch, generation=min_generation)

    # ── instance lock ────────────────────────────────────────────────

    async def acquire_instance_lock(
        self,
        prefix: KeyPrefix,
        *,
        ttl_s: float,
        deadline_monotonic: float | None = None,
    ) -> Result[InstanceLease, DataPlaneError]:
        """Acquire a TTL-bounded lease over a key prefix.

        Returns a typed :class:`InstanceLease` (carried via :class:`Ok`)
        on success, or a typed :class:`LockHeld` on contention. The lease
        spawns a background renewal task on a ``ttl_s / 3`` cadence; if
        renewal ever fails the task signals the lease's
        :class:`OneShotFlag` so the caller's hot path can short-circuit
        without a blocking call.

        ``deadline_monotonic`` is honoured at the call boundary: an
        already-expired deadline returns :class:`DeadlineExceeded`
        without issuing the EVALSHA. Once the call is in flight the
        single round-trip runs to completion; finer-grained deadline
        propagation lands with the connection-pool wait-timeout.
        """
        self._validate_key_component(
            prefix,
            method="acquire_instance_lock",
            field_name="prefix",
            allow_colon=True,
        )
        self._validate_ttl(ttl_s, method="acquire_instance_lock")
        try:
            self._check_deadline(deadline_monotonic, "acquire_instance_lock")
        except DeadlineExceeded as exc:
            return Err(exc)
        lua = self._require_started("acquire_instance_lock")
        lock_key = f"{self._connection.key_prefix}:lock:{prefix}"
        lease_token = LeaseToken(secrets.token_urlsafe(_LEASE_TOKEN_BYTES))
        try:
            result = await lua.evalsha(
                _SCRIPT_LOCK_ACQUIRE,
                keys=[lock_key],
                args=[lease_token, max(1, int(ttl_s * 1000))],
            )
        except DataPlaneError as exc:
            return Err(exc)
        if int(result) != 1:
            # Best-effort holder lookup for diagnostics. The holder
            # field is redacted in str(err) so we don't leak the token.
            # A failed lookup must not mask the underlying LockHeld
            # outcome — degrade gracefully to a holder-less Err.
            try:
                holder = await self._connection.pool.get(lock_key)  # type: ignore[misc]
            except Exception:  # noqa: BLE001 - diagnostic-only lookup, degrade silently
                holder = None
            return Err(LockHeld(key=lock_key, holder=holder))
        flag = OneShotFlag()
        lease = InstanceLease(
            token=lease_token,
            key=lock_key,
            ttl_s=ttl_s,
            expires_at_monotonic=monotonic() + ttl_s,
            flag=flag,
        )
        # Spawn the background renewer. The task captures ``lease`` by
        # value; if a future call constructs a fresh lease (e.g. for
        # ttl_s change) it must call release + acquire, not mutate.
        task = asyncio.create_task(
            self._renew_lease_loop(lease), name=f"dataplane.renewer:{lock_key}"
        )
        self._renewers[lease_token] = task
        return Ok(lease)

    async def renew_instance_lock(
        self,
        lease: InstanceLease,
        *,
        ttl_s: float,
    ) -> Result[InstanceLease, DataPlaneError]:
        """Token-CAS EXPIRE; fails with :class:`LockLost` on token mismatch.

        On success returns a fresh :class:`InstanceLease` with the
        updated ``ttl_s`` / ``expires_at_monotonic`` and the same
        :class:`OneShotFlag`. The input lease becomes logically stale
        but remains hashable / equal to the new one for the fields that
        matter (``compare=False`` on ``flag`` and ``expires_at_monotonic``
        below the dataclass scope so renewals are transparent to
        callers holding the lease by value).
        """
        self._validate_ttl(ttl_s, method="renew_instance_lock")
        lua = self._require_started("renew_instance_lock")
        try:
            result = await lua.evalsha(
                _SCRIPT_LOCK_RENEW,
                keys=[lease.key],
                args=[lease.token, max(1, int(ttl_s * 1000))],
            )
        except DataPlaneError as exc:
            return Err(exc)
        if int(result) != 1:
            return Err(LockLost(key=lease.key))
        return Ok(
            InstanceLease(
                token=lease.token,
                key=lease.key,
                ttl_s=ttl_s,
                expires_at_monotonic=monotonic() + ttl_s,
                flag=lease.flag,
            )
        )

    async def release_instance_lock(self, lease: InstanceLease) -> None:
        """Token-CAS DEL; no-op if the lease was already released or lost.

        Cancels the lease's renewal task AND awaits its exit before
        issuing the DEL so a renewer that's mid-Redis-call cannot race
        the release. Without the wait the renewer's pending EXPIRE
        could land after our DEL, leaving the lock un-bounded for one
        renew interval. Coordinator must be started; an already-shut-
        down coordinator silently no-ops.
        """
        task = self._renewers.pop(lease.token, None)
        if task is not None:
            task.cancel()
            # ``return_exceptions=True`` so an unrelated renewer
            # exception (e.g. flag-signalled mid-renew) does not mask
            # the release path. The renewer can still race with DEL on
            # a renewing EXPIRE iff Redis processes its request before
            # our cancellation; the Lua script's token-CAS handles
            # that — the EXPIRE on a missing key returns 0.
            await asyncio.gather(task, return_exceptions=True)
        if not self._started or self._lua is None:
            return
        lua = self._lua
        try:
            await lua.evalsha(
                _SCRIPT_LOCK_RELEASE,
                keys=[lease.key],
                args=[lease.token],
            )
        except DataPlaneError:
            # Release is idempotent; swallow transient connection
            # errors so callers can use this in a finally block safely.
            pass

    def wrap_lease(
        self, lease: InstanceLease
    ) -> CrashWatchedHandle[InstanceLease, str, InstanceLease]:
        """Wrap a lease in a :class:`CrashWatchedHandle` for typed loss recovery.

        Callers invoke lease-scoped operations through the returned
        handle and pattern-match the result::

            handle = dp.wrap_lease(lease)
            match await handle.call(lambda l: dp.renew_instance_lock(l, ttl_s=10)):
                case (result, None):
                    ...  # normal path; ``result`` is the wrapped op's Result
                case (None, evt):
                    ...  # evt: CrashEvent[str, InstanceLease], evt.peer == lock key,
                         # evt.resource is the lost lease itself.

        The wrap is opt-in convenience. Unwrapped calls
        (:meth:`renew_instance_lock`, :meth:`release_instance_lock`)
        retain their default semantics — a lease whose flag has been
        signalled by the background renewer continues to return
        :class:`Err` (:class:`LockLost`) on direct invocation. The
        handle adds a short-circuit branch *before* the underlying call,
        so once the renewer has signalled loss every subsequent
        ``handle.call`` resolves to ``(None, CrashEvent)`` without
        another Redis round-trip.

        ``peer`` is the canonical lock key (``{prefix}:lock:{caller-prefix}``);
        ``resource`` is the lost lease, surfaced verbatim so the caller
        can route compensation by token or key. ``survivor_tokens`` is
        the empty tuple — the instance lock sits beneath any permission
        subsystem at this stage, so there are no survivor permissions
        to mint.
        """

        async def _recover(
            lost: InstanceLease,
        ) -> CrashEvent[str, InstanceLease]:
            return CrashEvent(peer=lost.key, resource=lost, survivor_tokens=())

        return CrashWatchedHandle(lease, lease.flag, _recover)

    async def _renew_lease_loop(self, lease: InstanceLease) -> None:
        """Background renewal loop. Signals the lease's flag on loss.

        Runs forever until either (a) the task is cancelled (typical
        path: :meth:`release_instance_lock` or :meth:`shutdown`), or (b)
        renewal fails — token mismatch (real loss), a typed
        :class:`DataPlaneError`, or any unexpected exception. Cases
        other than cancellation signal the lease's flag so the holder
        observes the loss on its next dataplane call.

        Cadence is ``ttl_s / _LOCK_RENEW_RATIO`` with a floor of 1 ms so
        pathologically tiny TTLs (test fixtures) still make forward
        progress.
        """
        interval = max(lease.ttl_s / _LOCK_RENEW_RATIO, 0.001)
        while True:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                # Cooperative cancellation from release / shutdown; the
                # caller does not treat this as a lock loss.
                return
            try:
                result = await self.renew_instance_lock(lease, ttl_s=lease.ttl_s)
            except asyncio.CancelledError:
                # Cancellation racing the in-flight renew: propagate so
                # the gather() in :meth:`_cancel_renewers` observes it.
                # The flag stays unset — cancellation is not loss.
                raise
            except Exception:  # noqa: BLE001 - watchdog boundary, surfaced via flag
                # Any other failure (NotStartedError after shutdown,
                # programming bug, transient client error not wrapped
                # as DataPlaneError) is escalated to the holder via
                # the one-shot flag. The exception is swallowed because
                # nobody is awaiting this task in steady state.
                lease.flag.signal()
                return
            if isinstance(result, Err):
                lease.flag.signal()
                return

    async def _cancel_renewers(self) -> None:
        """Cancel every active renewer; wait for them to exit.

        Called from :meth:`shutdown`. We gather the tasks with
        ``return_exceptions=True`` so a stuck or already-failed renewer
        does not block the shutdown path.
        """
        tasks = list(self._renewers.values())
        self._renewers.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ── archive ──────────────────────────────────────────────────────

    async def try_replace_elite(
        self,
        cell: CellKey,
        candidate_id: ProgramId,
        *,
        token: Token[CellKey],
        candidate_score: float,
        tiebreak_bit: int,
        deadline_monotonic: float | None = None,
    ) -> Result[EliteSwapOutcome, DataPlaneError]:
        """Atomic archive cell swap.

        Comparison reduces to ``(candidate_score, tiebreak_bit)`` —
        both computed Python-side from the caller's ``is_better``
        predicate before the Lua call. The Lua script atomically reads
        the occupant's score, compares, and either swaps, inserts (if
        the cell was empty), or rejects. The discriminated
        :data:`EliteSwapOutcome` carries the loser / survivor id so the
        caller can route follow-up work (e.g. discarding the displaced
        program) without a second read.

        ``tiebreak_bit`` controls equal-score behaviour: ``1`` means
        the candidate wins ties (favours the latest write — useful when
        equal-score programs should rotate), ``0`` means the occupant
        wins ties (favours stability — useful when equal-score programs
        should be left undisturbed).
        """
        self._validate_key_component(
            cell, method="try_replace_elite", field_name="cell"
        )
        self._validate_key_component(
            candidate_id, method="try_replace_elite", field_name="candidate_id"
        )
        if math.isnan(candidate_score) or math.isinf(candidate_score):
            return Err(
                EliteInvalidError(
                    detail=(f"candidate_score must be finite, got {candidate_score!r}")
                )
            )
        if int(tiebreak_bit) not in (0, 1):
            return Err(
                EliteInvalidError(
                    detail=f"tiebreak_bit must be 0 or 1, got {tiebreak_bit!r}"
                )
            )
        try:
            self._check_deadline(deadline_monotonic, "try_replace_elite")
        except DeadlineExceeded as exc:
            return Err(exc)
        lua = self._require_started("try_replace_elite")
        tag = token.consume()
        if tag != cell:
            return Err(
                DataPlaneError(
                    f"try_replace_elite: token tag {tag!r} does not match cell {cell!r}"
                )
            )
        prefix = self._connection.key_prefix
        archive_key = f"{prefix}:archive"
        reverse_key = f"{prefix}:archive:reverse"
        scores_key = f"{prefix}:archive:scores"
        try:
            raw = await lua.evalsha(
                _SCRIPT_ARCHIVE_SWAP,
                keys=[archive_key, reverse_key, scores_key],
                args=[
                    cell,
                    candidate_id,
                    repr(float(candidate_score)),
                    str(int(tiebreak_bit)),
                ],
            )
        except DataPlaneError as exc:
            return Err(exc)
        status, displaced_or_occupant = raw
        if status == "inserted":
            return Ok(EliteInserted())
        if status == "swapped":
            return Ok(EliteSwapped(displaced_id=ProgramId(displaced_or_occupant)))
        if status == "rejected":
            return Ok(EliteRejected(occupant_id=ProgramId(displaced_or_occupant)))
        if status == "invalid":
            # The Lua script rejected the candidate at the input boundary
            # (non-finite score, empty cell or candidate id). The payload
            # carries the reason verbatim from the script.
            return Err(
                EliteInvalidError(detail=str(displaced_or_occupant)),
            )
        return Err(
            DataPlaneError(f"try_replace_elite: unexpected Lua status {status!r}")
        )

    # ── CRDT counters ────────────────────────────────────────────────

    async def crdt_inc(
        self,
        key: CounterKey,
        *,
        actor: ActorIdentity,
        delta: int = 1,
        token: Token[CounterKey] | None = None,
        deadline_monotonic: float | None = None,
    ) -> Result[int, DataPlaneError]:
        """Increment a per-actor G-counter; consensus value is sum across actors.

        Returns the post-increment per-actor value (not the cross-actor
        sum). Use :meth:`crdt_read` for the consensus sum.

        ``delta`` may be negative; the per-actor sub-count is signed. The
        G-counter merge invariant only requires that *each actor* writes
        monotonically — callers that decrement should hold a token for
        that actor's subspace.

        ``token`` is the move-only permission witness for the
        ``CounterKey`` subspace. When supplied, the wrapper consumes it
        before the Lua call and verifies the tag matches ``key`` — the
        same single-writer discipline that :meth:`transition_program_state`
        and :meth:`try_replace_elite` apply to their subspaces. ``None``
        preserves the legacy ledger-only contract used by callers that
        have not yet been wired to the engine-root token bundle; new
        production call sites SHOULD supply a token split from the
        engine's counter root.
        """
        self._validate_key_component(
            key, method="crdt_inc", field_name="key", allow_colon=True
        )
        try:
            self._check_deadline(deadline_monotonic, "crdt_inc")
        except DeadlineExceeded as exc:
            return Err(exc)
        lua = self._require_started("crdt_inc")
        if token is not None:
            tag = token.consume()
            if tag != key:
                return Err(
                    DataPlaneError(
                        f"crdt_inc: token tag {tag!r} does not match key {key!r}"
                    )
                )
        counts_key, gen_key, epoch_key = self._counter_keys(key)
        try:
            raw = await lua.evalsha(
                _SCRIPT_COUNTER_INC,
                keys=[counts_key, gen_key, epoch_key],
                args=[actor.pack(), int(delta)],
            )
        except DataPlaneError as exc:
            return Err(exc)
        # Lua returns ASCII-encoded ints under decode_responses=True.
        new_count = int(raw[0])
        return Ok(new_count)

    async def crdt_read(
        self,
        key: CounterKey,
        *,
        freshness: Freshness | None = None,
        min_epoch: int = 0,
        min_generation: int = 0,
    ) -> Result[Versioned[int], DataPlaneError]:
        """Read a G-counter as a :class:`Versioned` sum across all actors.

        Per-actor reads are not exposed: a G-counter's only meaningful
        observable is the cross-actor sum. The ``freshness`` argument
        declares the admission contract:

            * :class:`FreshnessEventual` — accept any persisted view
              (the default; matches ``min_epoch=0, min_generation=0``).
            * :class:`FreshnessAtLeast` — admission floor on
              ``(epoch, generation)``. A view below the floor returns
              :class:`StaleReadError` so a caller observing its own
              prior increment fails loud rather than silently old.
            * :class:`FreshnessStrict` — snapshot the live epoch counter
              first; the pipeline's view must clear that snapshot. Costs
              one extra round-trip and is the right choice for
              cross-engine readers that cannot supply their own floor.

        ``min_epoch`` / ``min_generation`` remain accepted as a
        backwards-compat shim mapping to :class:`FreshnessAtLeast`.
        Passing both an explicit ``freshness`` and a non-zero ``min_*``
        surfaces as :class:`Err(DataPlaneError)` so the two channels
        cannot silently disagree.

        Three Redis commands run as a single non-transactional pipeline
        (one round-trip). The G-counter's eventual-consistency model
        does not require the three reads to be atomic — a HGETALL that
        misses the last increment merely yields a slightly-older
        ``Versioned`` whose epoch / generation reflect that staleness,
        which is exactly what the freshness floor catches.
        """
        self._validate_key_component(
            key, method="crdt_read", field_name="key", allow_colon=True
        )
        self._validate_floor(min_epoch, min_generation, method="crdt_read")
        try:
            effective_freshness = self._resolve_freshness(
                freshness, min_epoch, min_generation, method="crdt_read"
            )
        except ValueError as exc:
            return Err(DataPlaneError(str(exc)))
        self._require_started("crdt_read")
        counts_key, gen_key, epoch_key = self._counter_keys(key)
        redis = self._connection.pool

        # FreshnessStrict pre-snapshots the live epoch counter so a
        # concurrent writer that bumps it after our snapshot is detected
        # as a stale view. The pipeline reads ``epoch_key`` (which IS
        # the live counter at ``{prefix}:ts``) again as part of the
        # pipeline; the floor is ``max(pre-snapshot, the value the
        # pipeline observes)`` so a counter advance between snapshot and
        # pipeline closes the same race the ``read_program`` Strict path
        # closes.
        strict_floor: int = 0
        if isinstance(effective_freshness, FreshnessStrict):
            raw_counter = await redis.get(epoch_key)  # type: ignore[misc]
            try:
                strict_floor = int(raw_counter) if raw_counter is not None else 0
            except (TypeError, ValueError) as exc:
                return Err(
                    DataPlaneError(
                        f"crdt_read: strict-freshness epoch counter "
                        f"unparseable: {exc!r}"
                    )
                )

        pipe = redis.pipeline(transaction=False)
        pipe.hgetall(counts_key)
        pipe.get(gen_key)
        pipe.get(epoch_key)
        counts_map, gen_raw, epoch_raw = await pipe.execute()  # type: ignore[misc]
        generation = int(gen_raw) if gen_raw is not None else 0
        epoch = int(epoch_raw) if epoch_raw is not None else 0
        total = sum(int(v) for v in counts_map.values()) if counts_map else 0
        versioned = Versioned(value=total, epoch=epoch, generation=generation)

        floor_epoch: int
        floor_generation: int
        if isinstance(effective_freshness, FreshnessStrict):
            floor_epoch = strict_floor
            floor_generation = strict_floor
        elif isinstance(effective_freshness, FreshnessAtLeast):
            floor_epoch = effective_freshness.epoch
            floor_generation = effective_freshness.generation
        else:
            floor_epoch = 0
            floor_generation = 0

        if not versioned.is_at_least(floor_epoch, floor_generation):
            return Err(
                StaleReadError(
                    observed_epoch=epoch,
                    observed_generation=generation,
                    min_epoch=floor_epoch,
                    min_generation=floor_generation,
                )
            )
        return Ok(versioned)

    # ── LWW register (HLC-tiebreak last-write-wins) ──────────────────

    async def lwwr_set(
        self,
        key: str,
        value: object,
        *,
        hlc: HlcTimestamp,
    ) -> Result[LwwrSetOutcome, DataPlaneError]:
        """Atomically write a single-value LWW-register guarded by HLC.

        ``key`` is the logical register name (the coordinator prefixes
        it with ``{key_prefix}:lwwr:`` to namespace). ``value`` is any
        canonical-JSON-encodable Python object; the wrapper encodes it
        via :func:`gigaevo.dataplane.codec.encode_canonical`. ``hlc`` is
        the witness — writes with a strictly-newer HLC replace; ties
        and older writes are rejected silently as ``Ok("kept")``.

        Returns ``Ok("replaced")`` on a successful write or
        ``Ok("kept")`` if the stored register's HLC was already at or
        beyond the candidate's. ``Err(DataPlaneError)`` surfaces only on
        connection / script-load failures.
        """
        self._validate_key_component(
            key, method="lwwr_set", field_name="key", allow_colon=True
        )
        lua = self._require_started("lwwr_set")
        register_key = f"{self._connection.key_prefix}:lwwr:{key}"
        encoded_value = encode_canonical(value).decode("utf-8")
        try:
            outcome = await lua.evalsha(
                _SCRIPT_LWWR_SET,
                keys=[register_key],
                args=[encoded_value, hlc.pack_hex()],
            )
        except DataPlaneError as exc:
            return Err(exc)
        # Lua returns 'replaced' or 'kept' (both Literal members of
        # LwwrSetOutcome); narrow defensively.
        if outcome == "replaced":
            return Ok("replaced")
        if outcome == "kept":
            return Ok("kept")
        return Err(DataPlaneError(f"lwwr_set: unexpected Lua return value {outcome!r}"))

    async def lwwr_get(
        self,
        key: str,
    ) -> Result[LwwrValue | None, DataPlaneError]:
        """Read the current LWW-register value plus HLC.

        Returns ``Ok(None)`` if the register has never been written.
        Returns ``Ok(LwwrValue(...))`` with the decoded value and its
        HLC witness otherwise. ``Err(DataPlaneError)`` on connection /
        decoding failures.
        """
        self._validate_key_component(
            key, method="lwwr_get", field_name="key", allow_colon=True
        )
        self._require_started("lwwr_get")
        register_key = f"{self._connection.key_prefix}:lwwr:{key}"
        redis = self._connection.pool
        # HMGET is one round-trip and returns [value, hlc] in order.
        raw = await redis.hmget(register_key, ["value", "hlc"])  # type: ignore[misc]
        if not raw or raw[0] is None or raw[1] is None:
            return Ok(None)
        value_raw, hlc_hex = raw
        decoded = decode_canonical(value_raw)
        return Ok(LwwrValue(value=decoded, hlc=HlcTimestamp.unpack_hex(hlc_hex)))

    # ── bounded recency list ────────────────────────────────────────
    #
    # A bounded recency list holds the newest N entries of a value
    # stream — pure LWW semantics, no per-actor merge. Writers push
    # at the head and the script trims to the cap atomically so
    # concurrent pushers cannot leave the list briefly over-cap.
    # Reads return the slice ``[0, count)`` in insertion-newest-first
    # order. Used by the prompt-fitness window (§5.6) as a lighter
    # alternative to a CRDT counter for "last N samples".

    async def bounded_list_push(
        self,
        key: str,
        value: object,
        *,
        cap: int,
        deadline_monotonic: float | None = None,
    ) -> Result[int, DataPlaneError]:
        """Atomically LPUSH ``value`` at the head and trim to ``cap`` entries.

        ``key`` is the fully-qualified Redis list key (the coordinator
        prefixes it with ``{key_prefix}:list:`` to namespace). ``value``
        is encoded via :func:`gigaevo.dataplane.codec.encode_canonical`,
        so any canonical-JSON-encodable Python value round-trips. The
        ``cap`` cap is enforced server-side by the Lua script (positive
        integer); a non-positive cap returns
        :class:`DataPlaneError` rather than silently becoming a
        no-op trim.

        Returns ``Ok(new_length)`` on success — the post-trim length of
        the list, capped at ``cap``. Concurrent writers race only on
        ordering of their values; each push is itself atomic.
        """
        self._validate_key_component(
            key, method="bounded_list_push", field_name="key", allow_colon=True
        )
        if not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0:
            return Err(
                DataPlaneError(
                    f"bounded_list_push: cap must be a positive int, got {cap!r}"
                )
            )
        try:
            self._check_deadline(deadline_monotonic, "bounded_list_push")
        except DeadlineExceeded as exc:
            return Err(exc)
        lua = self._require_started("bounded_list_push")
        list_key = f"{self._connection.key_prefix}:list:{key}"
        encoded = encode_canonical(value).decode("utf-8")
        try:
            raw = await lua.evalsha(
                _SCRIPT_BOUNDED_LIST_PUSH,
                keys=[list_key],
                args=[encoded, int(cap)],
            )
        except DataPlaneError as exc:
            return Err(exc)
        return Ok(int(raw[0]))

    async def bounded_list_range(
        self,
        key: str,
        *,
        count: int,
    ) -> Result[list[object], DataPlaneError]:
        """Read the first ``count`` entries of a bounded list, newest-first.

        Returns the canonical-decoded values in the order Redis returns
        them (LPUSH puts the newest at index 0, so the result is
        most-recent-first). An empty or unknown list yields ``Ok([])``.
        Each entry is decoded via :func:`decode_canonical`; a malformed
        entry surfaces as an :class:`Err` with the offending index in
        the message rather than silently dropping the entry.
        """
        self._validate_key_component(
            key, method="bounded_list_range", field_name="key", allow_colon=True
        )
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            return Err(
                DataPlaneError(
                    f"bounded_list_range: count must be a positive int, got {count!r}"
                )
            )
        self._require_started("bounded_list_range")
        list_key = f"{self._connection.key_prefix}:list:{key}"
        try:
            raw = await self._connection.pool.lrange(list_key, 0, count - 1)  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001 - coordinator boundary
            return Err(DataPlaneError(f"bounded_list_range: lrange failed: {exc!r}"))
        out: list[object] = []
        for i, entry in enumerate(raw):
            try:
                out.append(decode_canonical(entry))
            except Exception as exc:  # noqa: BLE001 - coordinator boundary
                return Err(
                    DataPlaneError(
                        f"bounded_list_range: entry index {i} failed to decode: {exc!r}"
                    )
                )
        return Ok(out)

    # ── small-set directory ──────────────────────────────────────────
    #
    # A directory of opaque string members. Members are deduplicated
    # server-side by SADD; reads return the unordered set. The size of
    # the set is unbounded — callers that need a cap should size their
    # member alphabet, not the set. Used for the per-prompt metric-name
    # registry that feeds the stats reader.

    async def set_add(
        self,
        key: str,
        member: str,
    ) -> Result[int, DataPlaneError]:
        """Add ``member`` to the named set; idempotent under duplicate adds.

        Returns ``Ok(1)`` when a new member was added, ``Ok(0)`` when
        the member was already present. SADD is atomic in Redis, so no
        Lua script is needed.
        """
        self._validate_key_component(
            key, method="set_add", field_name="key", allow_colon=True
        )
        if not isinstance(member, str) or not member:
            return Err(
                DataPlaneError(
                    f"set_add: member must be a non-empty string, got {member!r}"
                )
            )
        self._require_started("set_add")
        set_key = f"{self._connection.key_prefix}:set:{key}"
        try:
            added = await self._connection.pool.sadd(set_key, member)  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001 - coordinator boundary
            return Err(DataPlaneError(f"set_add: sadd failed: {exc!r}"))
        return Ok(int(added))

    async def set_members(
        self,
        key: str,
    ) -> Result[frozenset[str], DataPlaneError]:
        """Read every member of the named set; unordered, deduplicated.

        Returns ``Ok(frozenset())`` for an unknown / empty set. The
        frozenset shape signals to the caller that the return is a
        snapshot — concurrent writers may have added new members
        between the read and the caller's use; a stale-cache witness
        is not provided because the set has no canonical version.
        """
        self._validate_key_component(
            key, method="set_members", field_name="key", allow_colon=True
        )
        self._require_started("set_members")
        set_key = f"{self._connection.key_prefix}:set:{key}"
        try:
            raw = await self._connection.pool.smembers(set_key)  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001 - coordinator boundary
            return Err(DataPlaneError(f"set_members: smembers failed: {exc!r}"))
        return Ok(frozenset(str(m) for m in raw))

    # ── raw-key access (cross-namespace) ─────────────────────────────
    #
    # The dataplane's typed primitives bake the coordinator's
    # ``key_prefix`` into every key. Some callers — notably the prompt
    # co-evolution path, which reads keys written by an independently
    # configured "main run" with its own prefix — need to address keys
    # *outside* the coordinator's own namespace. These primitives accept
    # the fully-qualified Redis key and skip prefix prepending. They are
    # the only sanctioned escape hatch; new code should still prefer the
    # typed primitives above.

    async def raw_hash_get(
        self,
        key: str,
        field: str,
    ) -> Result[str | None, DataPlaneError]:
        """HGET ``field`` from a fully-qualified hash key.

        Returns ``Ok(None)`` when the hash or field is missing,
        ``Ok(str)`` when present. The dataplane's connection mandates
        ``decode_responses=True`` so the value is always a Python
        ``str``.
        """
        self._validate_key_component(
            key, method="raw_hash_get", field_name="key", allow_colon=True
        )
        self._validate_key_component(
            field, method="raw_hash_get", field_name="field", allow_colon=True
        )
        self._require_started("raw_hash_get")
        try:
            raw = await self._connection.pool.hget(key, field)  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001 - coordinator boundary
            return Err(DataPlaneError(f"raw_hash_get: hget failed: {exc!r}"))
        if raw is None:
            return Ok(None)
        return Ok(str(raw))

    async def raw_hash_values(
        self,
        key: str,
    ) -> Result[list[str], DataPlaneError]:
        """HVALS for a fully-qualified hash key; empty list when missing."""
        self._validate_key_component(
            key, method="raw_hash_values", field_name="key", allow_colon=True
        )
        self._require_started("raw_hash_values")
        try:
            raw = await self._connection.pool.hvals(key)  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001 - coordinator boundary
            return Err(DataPlaneError(f"raw_hash_values: hvals failed: {exc!r}"))
        return Ok([str(v) for v in raw])

    async def raw_get(
        self,
        key: str,
    ) -> Result[str | None, DataPlaneError]:
        """GET a fully-qualified string key; ``Ok(None)`` when missing."""
        self._validate_key_component(
            key, method="raw_get", field_name="key", allow_colon=True
        )
        self._require_started("raw_get")
        try:
            raw = await self._connection.pool.get(key)  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001 - coordinator boundary
            return Err(DataPlaneError(f"raw_get: get failed: {exc!r}"))
        if raw is None:
            return Ok(None)
        return Ok(str(raw))

    # ── built-in-script bookkeeping ──────────────────────────────────

    def _register_builtin_scripts(self, lua: LuaRegistry) -> None:
        """Load every per-resource Lua source into the registry.

        Called once during :meth:`startup` before :meth:`LuaRegistry.load_all`
        so the SHA cache is primed in one round-trip. Each script lives
        as a ``.lua`` file under ``gigaevo/dataplane/scripts/``; the name
        used here matches the file stem.
        """
        for name in (
            _SCRIPT_COUNTER_INC,
            _SCRIPT_LOCK_ACQUIRE,
            _SCRIPT_LOCK_RENEW,
            _SCRIPT_LOCK_RELEASE,
            _SCRIPT_LWWR_SET,
            _SCRIPT_TRANSITION_STATE,
            _SCRIPT_ARCHIVE_SWAP,
            _SCRIPT_BOUNDED_LIST_PUSH,
        ):
            lua.register(name, load_lua_source(name))

    def _counter_keys(self, key: CounterKey) -> tuple[str, str, str]:
        """Resolve the three Redis keys backing a CRDT counter.

        Returns ``(counts_hash, generation_counter, epoch_counter)``.
        The epoch counter is shared across every coordinator op (any
        write bumps it); the generation counter is per-counter so two
        independent counters do not race on each other's freshness
        witness.
        """
        prefix = self._connection.key_prefix
        return (
            f"{prefix}:{key}:counts",
            f"{prefix}:{key}:gen",
            f"{prefix}:ts",
        )


__all__ = [
    "BatchTransitionItem",
    "BatchTransitionOutcome",
    "DataPlane",
    "EliteInserted",
    "EliteRejected",
    "EliteSwapOutcome",
    "EliteSwapped",
    "InstanceLease",
    "LwwrSetOutcome",
    "LwwrValue",
    "ProgramPatch",
    "ProgramSnapshot",
]
