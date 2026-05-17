"""Crash-event recovery primitives.

A hot-path consumer peeks a one-shot flag on every call; on the rare
true path the next dataplane call returns ``(None, CrashEvent)`` instead
of ``(value, None)``. Recovery is a control-flow branch, not an
exception unwind. Hot-path cost is one non-blocking flag read per call.

Asyncio-only contract: :class:`OneShotFlag` wraps :class:`asyncio.Event`
and is single-loop only. Cross-thread signalling must round-trip via
``loop.call_soon_threadsafe``; the dataplane's own watchdogs are
coroutines so this never comes up in production.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

# ── OneShotFlag ───────────────────────────────────────────────────────


class OneShotFlag:
    """Single-direction "the peer is dead" / "the lock is lost" signal.

    Produced by one watchdog, consumed by any number of readers; once
    set, it stays set. Single-loop only (see module docstring).
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

    Carries the dead peer's tag (for routing to a recovery policy), the
    recovered resource (e.g. a fresh lease after re-acquisition), and a
    tuple of survivor permission tokens minted at the crash boundary.

    ``survivor_tokens`` is ``tuple[object, ...]`` because a single event
    may mix heterogeneous tokens (program / cell / counter). Callers
    narrow at the consumption site.
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

    Each call peeks the flag first; a set flag short-circuits to
    ``recover_fn`` and returns the crash event. Callers swap in the
    recovered resource via :meth:`replace_inner` (minting a fresh flag
    by convention). The recovery policy is caller-injected so this class
    stays mechanism-only.
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

        If ``new_flag`` is omitted, the existing flag is retained — a
        still-set flag will short-circuit every subsequent call, so the
        usual choice is to pass a fresh flag.
        """
        self._inner = new_inner
        if new_flag is not None:
            self._flag = new_flag

    async def call[R](
        self,
        method: Callable[[T], Awaitable[R]],
    ) -> Recovered[R, PeerTag, Resource]:
        """Invoke ``method`` on the inner handle with crash interception.

        Flag set before the call ⇒ recovery runs, returns
        ``(None, CrashEvent)``. Otherwise the method runs and returns
        ``(value, None)``. A flag set *during* execution is observed only
        on the next call. A raising ``recover_fn`` propagates unchanged
        — the supervisory layer must catch it.
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
