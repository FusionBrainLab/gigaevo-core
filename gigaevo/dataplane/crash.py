"""Crash-event recovery primitives.

The pattern: a hot-path consumer peeks a one-shot flag on every call,
and on the rare true path the caller's next dataplane invocation
returns ``(None, CrashEvent)`` instead of the normal ``(value, None)``
pair — recovery becomes a control-flow branch, not an exception unwind.

The flag is backed by :class:`asyncio.Event`. Hot-path cost is one
non-blocking flag read per call, which is acceptable at the coarse
granularity of a network round-trip.

Asyncio-only contract
=====================
:class:`OneShotFlag` is **not** thread-safe. It wraps
``asyncio.Event``, which is documented to be safe only within the
single event-loop thread that created it. Calling :meth:`signal` from
:func:`asyncio.to_thread` *appears* to work because ``Event.set`` does
not itself await — but a coroutine that's parked on :meth:`wait`
running in the same loop will not be woken until the loop schedules it,
and there is no guarantee the worker thread's modification is visible
to the loop thread without an explicit ``loop.call_soon_threadsafe``
fence. Signalling from off-loop threads is forbidden; cross-thread
signalling must round-trip through ``call_soon_threadsafe`` (or use a
different primitive altogether). The dataplane's own watchdogs are
async coroutines, so this never comes up in production; the rule is
documented here so future contributors do not introduce it by accident.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

# ── OneShotFlag ───────────────────────────────────────────────────────


class OneShotFlag:
    """Single-direction "the peer is dead" / "the lock is lost" signal.

    The flag is produced by exactly one party (the watchdog that detects
    the crash) and consumed by any number of readers. Once set, it stays
    set — the recovery handler runs once, mints survivor permissions,
    and the owning resource is replaced.

    Bound to a single asyncio loop. See the module docstring for the
    cross-thread contract; the short version is: don't.
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def signal(self) -> None:
        """Mark the flag as set. Idempotent."""
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> None:
        """Block until the flag is set."""
        await self._event.wait()


# ── CrashEvent ────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class CrashEvent[PeerTag, Resource]:
    """Typed recovery payload returned in place of a normal value.

    Carries:
        - the dead peer's tag (so handlers can route to a recovery policy)
        - the recovered resource (e.g. a fresh Redis connection after
          Sentinel failover, or a fresh lease after re-acquisition)
        - a tuple of survivor permission tokens minted at the crash
          boundary; subsequent calls require these as proof of
          legitimate post-crash operation.

    Why ``survivor_tokens: tuple[object, ...]`` and not a typed tuple
    --------------------------------------------------------------
    A single :class:`CrashEvent` carries a heterogeneous tuple — a
    program-FSM token, a cell-swap token, and a CRDT-actor token can
    all be minted for the same crash. The element types vary per crash
    class and are not known to ``CrashEvent`` itself. A
    ``tuple[Token[ProgramId] | Token[CellKey] | ...]`` would either
    spuriously narrow (when one of the variants is absent) or recreate
    the same ``object``-typed escape hatch behind a longer signature.
    Callers narrow at the consumption site via ``isinstance`` /
    pattern-matching; the documentation is the contract.
    """

    peer: PeerTag
    resource: Resource
    survivor_tokens: tuple[object, ...] = field(default_factory=tuple)


# ── Recovered alias ───────────────────────────────────────────────────


type Recovered[T, PeerTag, Resource] = (
    tuple[T, None] | tuple[None, CrashEvent[PeerTag, Resource]]
)
"""Return shape of a crash-watched call.

Callers match::

    match await handle.call(op):
        case (value, None):
            ...  # normal path
        case (None, evt):
            ...  # recovery path; evt is CrashEvent[PeerTag, Resource]
"""


# ── CrashWatchedHandle ────────────────────────────────────────────────


class CrashWatchedHandle[T, PeerTag, Resource]:
    """Wraps an inner resource plus a :class:`OneShotFlag`.

    Every call peeks the flag first. If the flag is set, the call short-
    circuits to a recovery path that mints survivors and returns the
    crash event. Otherwise the inner method runs and its result is
    returned as ``(value, None)``.

    After a recovery event, callers replace the wrapped resource via
    :meth:`replace_inner`. The :class:`OneShotFlag` may be cleared by
    the caller or replaced by minting a fresh flag for the new resource;
    the dataplane's convention is to mint a fresh flag.

    The reconnect / survivor-minting policy is injected via
    ``recover_fn`` so this class stays mechanism-only.
    """

    __slots__ = ("_inner", "_flag", "_recover")

    def __init__(
        self,
        inner: T,
        flag: OneShotFlag,
        recover_fn: Callable[[T], Awaitable[CrashEvent[PeerTag, Resource]]],
    ) -> None:
        self._inner: T = inner
        self._flag = flag
        self._recover = recover_fn

    @property
    def inner(self) -> T:
        return self._inner

    @property
    def flag(self) -> OneShotFlag:
        return self._flag

    def replace_inner(self, new_inner: T, new_flag: OneShotFlag | None = None) -> None:
        """Swap in the recovered resource (and optionally a fresh flag).

        Called by the caller after handling a :class:`CrashEvent`. If
        ``new_flag`` is omitted the existing flag stays — callers should
        only re-use the old flag if they've cleared its asyncio.Event,
        otherwise every subsequent call short-circuits to recovery.
        """
        self._inner = new_inner
        if new_flag is not None:
            self._flag = new_flag

    async def call[R](
        self,
        method: Callable[[T], Awaitable[R]],
    ) -> Recovered[R, PeerTag, Resource]:
        """Invoke ``method`` on the inner handle with crash interception.

        If the flag is set before the call, recovery runs and a
        :class:`CrashEvent` is returned. Otherwise the method runs to
        completion and its result is returned as ``(value, None)``.

        A flag set DURING the method's execution is not observed by
        this call (we accept stale-but-completed work). The next call
        observes it.
        """
        if self._flag.is_set():
            event = await self._recover(self._inner)
            return (None, event)
        value = await method(self._inner)
        return (value, None)


__all__ = [
    "CrashEvent",
    "CrashWatchedHandle",
    "OneShotFlag",
    "Recovered",
]
