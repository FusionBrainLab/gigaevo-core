"""Parent refresh — the only post-seed DONE→QUEUED path under the
steady-state engine.

A producer task selects parents from the archive, then asks the
:class:`ParentRefresher` to:

1. Expand the parent set via a :class:`ParentRefreshSelector` (default:
   direct parents only; future implementations may walk lineage).
2. Flip every selected program from DONE → QUEUED (in one batch transition,
   so no producer sees a half-flipped parent bundle).
3. Wait until every flipped program is DONE again (re-evaluated by the
   DAG runner).
4. Return the freshly-evaluated :class:`Program` objects.

Concurrent producers that happen to select overlapping parents are
serialised on a per-parent-id :class:`asyncio.Lock` so a parent is never
double-flipped.

Failure semantics: if any parent ends up DISCARDED or vanishes during the
refresh wait, the helper raises :class:`ValueError`; the caller aborts
that mutant (releases its in-flight slot) rather than fall back to stale
state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass, field
import time
import weakref

from loguru import logger

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS, Program
from gigaevo.programs.program_state import ProgramState

_REFRESH_POLL_S = 0.25


@dataclass
class ParentRefreshTicket:
    """Receipt for a successful refresh that the caller will own past return.

    The ticket carries:
      * ``refreshed``: the freshly-evaluated parent programs (state=DONE).
      * Per-parent-id ``asyncio.Lock`` instances, held in deterministic
        sorted-by-id order at the time the ticket was issued.

    Ownership contract: whoever holds the ticket MUST call :meth:`release`
    exactly once. ``release()`` is idempotent — calling it twice is a no-op
    rather than an error — which lets handoff points (mutant_task →
    ingestor) double-release defensively without corrupting the lock state.

    The locks are released in reverse acquisition order to avoid waking
    contention on later locks before earlier ones, matching the unwind
    order of the original ``_acquire_all`` context manager.

    This object is hand-off cooperative: a producer task can refresh,
    register the resulting child in ``_in_flight``, and store the ticket
    on the engine for the ingestor to release once the child reaches
    DONE/DISCARDED. That extends the lock-hold scope from "refresh only"
    to "refresh + mutate + child-DAG", which is the invariant that
    prevents another producer from seeing this parent's children mid-DAG.
    """

    refreshed: list[Program]
    _locks: list[asyncio.Lock] = field(default_factory=list)
    _released: bool = False

    def release(self) -> None:
        """Release all held locks. Idempotent."""
        if self._released:
            return
        self._released = True
        for lk in reversed(self._locks):
            lk.release()


class ParentRefreshSelector(ABC):
    """Expand a producer's parent pick into the full refresh target set.

    The producer selects which programs will mutate into a new mutant.
    Before the mutation runs, those parents may need a fresh fitness
    evaluation against the current Hall-of-Fame / adversarial set. This
    selector decides *which* programs are pulled into that refresh sweep.

    Default — :class:`DirectParentsSelector` — refreshes only the directly
    selected parents. Future implementations may walk lineage up to
    depth-k (grandparents, great-grandparents, …) and order the refresh
    in depth-batched waves so the deepest ancestors finish before the
    nearest parents flip.
    """

    @abstractmethod
    async def select(self, parents: list[Program]) -> list[Program]:
        """Return the full list of programs that must be refreshed.

        Implementations must return the input parents *plus* any
        additional programs they choose to include. Returning an empty
        list short-circuits :meth:`ParentRefresher.refresh` to a no-op.
        """


class DirectParentsSelector(ParentRefreshSelector):
    """Refresh only the directly selected parents.

    This is the canonical default: every mutant flips its own parents
    DONE → QUEUED → DONE before mutation, and touches nothing else in
    the lineage tree.
    """

    async def select(self, parents: list[Program]) -> list[Program]:
        return list(parents)


class ParentRefresher:
    """Per-parent-id locked DONE→QUEUED→DONE refresh helper.

    Pipeline:
      1. Expand selected parents via the configured :class:`ParentRefreshSelector`.
      2. Acquire per-id locks in deterministic (sorted) order.
      3. Flip DONE-state targets to QUEUED in one batch transition.
      4. Poll storage until every target is DONE again.
      5. Return the refreshed :class:`Program` objects.

    Args:
        storage: ProgramStorage backing the refresh.
        selector: Strategy for expanding the parent set. Defaults to
            :class:`DirectParentsSelector`.
        poll_interval: Seconds between storage polls while awaiting DONE.
        timeout_seconds: Overall wait budget per refresh, in seconds.
            Defaults to 600s (10 minutes). The engine-level stopper does
            NOT bound this individual wait — without a finite timeout, a
            DAG-runner crash that strands a parent in QUEUED would block
            the calling mutant task forever, leaking its in-flight slot.
            Pass an explicit value for tests that intentionally race the
            timeout; ``None`` is permitted but discouraged outside tests.
    """

    def __init__(
        self,
        *,
        storage: ProgramStorage,
        selector: ParentRefreshSelector | None = None,
        poll_interval: float = _REFRESH_POLL_S,
        timeout_seconds: float | None = 600.0,
    ) -> None:
        self._storage = storage
        self._selector = selector or DirectParentsSelector()
        self._poll_interval = poll_interval
        self._timeout_seconds = timeout_seconds
        # WeakValueDictionary: locks are retained only while at least one
        # active refresh holds a strong reference. Once all callers for a
        # parent id finish and drop their reference, the lock is GC'd —
        # preventing unbounded growth on multi-day runs that touch tens of
        # thousands of distinct parent ids.
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )
        self._registry_lock = asyncio.Lock()

    async def refresh(self, parents: list[Program]) -> list[Program]:
        """Flip and re-await all targets selected from ``parents``.

        Back-compat short form: acquires the per-parent-id locks, performs
        the DONE→QUEUED→DONE flip + await, and releases the locks before
        returning. Callers that need to hold the locks past return — e.g.
        the steady-state producer that must keep parents pinned through
        the entire child-DAG evaluation — should use
        :meth:`refresh_with_ticket` instead and release the returned ticket
        once they no longer require the lock guarantee.

        Duplicate ids in the selector output are folded to a single entry
        (keeping the first occurrence). Without this, ``_acquire_all`` would
        call ``acquire()`` twice on the same non-reentrant :class:`asyncio.Lock`
        and the mutant task would hang indefinitely, holding its in-flight slot.
        """
        ticket = await self.refresh_with_ticket(parents)
        try:
            return ticket.refreshed
        finally:
            ticket.release()

    async def refresh_with_ticket(self, parents: list[Program]) -> ParentRefreshTicket:
        """Same as :meth:`refresh` but returns a :class:`ParentRefreshTicket`
        whose lifetime the caller owns.

        Until the caller calls ``ticket.release()``, the per-parent-id
        locks remain held — any concurrent ``refresh()`` /
        ``refresh_with_ticket()`` for the same id BLOCKS. This is the
        primitive the steady-state engine uses to enforce
        "parents are not refreshed while their children are in flight":
        the producer holds the ticket until the ingestor confirms the
        resulting child reached DONE/DISCARDED.

        Failure paths (selector returns empty, ``_do_refresh`` raises)
        release any partially-acquired locks before re-raising — no
        ticket is returned in those paths, so the caller cannot leak.
        An empty parents list returns an empty-lock ticket that is still
        safe to ``release()``.
        """
        if not parents:
            return ParentRefreshTicket(refreshed=[], _locks=[])
        targets = await self._selector.select(parents)
        if not targets:
            return ParentRefreshTicket(refreshed=[], _locks=[])
        # Dedupe by program id (preserve first-seen order); see docstring.
        by_id: dict[str, Program] = {}
        for p in targets:
            by_id.setdefault(p.id, p)
        ordered = sorted(by_id.values(), key=lambda p: p.id)
        locks = [await self._get_lock(p.id) for p in ordered]
        acquired: list[asyncio.Lock] = []
        try:
            for lk in locks:
                await lk.acquire()
                acquired.append(lk)
            refreshed = await self._do_refresh(ordered)
            return ParentRefreshTicket(refreshed=refreshed, _locks=acquired)
        except BaseException:
            # Release every lock we managed to grab — otherwise the parent
            # ids stay stranded behind a permanent lock and any future
            # caller for them deadlocks.
            for lk in reversed(acquired):
                lk.release()
            raise

    async def _get_lock(self, pid: str) -> asyncio.Lock:
        async with self._registry_lock:
            lock = self._locks.get(pid)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[pid] = lock
            return lock

    async def _do_refresh(self, targets: list[Program]) -> list[Program]:
        for p in targets:
            if p.state == ProgramState.DISCARDED:
                raise ValueError(
                    f"ParentRefresher: parent {p.short_id} is DISCARDED; refusing to flip"
                )

        done_ids = [p.id for p in targets if p.state == ProgramState.DONE]
        if done_ids:
            await self._storage.batch_transition_by_ids(
                done_ids,
                ProgramState.DONE.value,
                ProgramState.QUEUED.value,
            )
            logger.debug(
                "ParentRefresher: flipped {} parents DONE->QUEUED",
                len(done_ids),
            )

        return await self._await_done([p.id for p in targets])

    async def _await_done(self, pids: list[str]) -> list[Program]:
        deadline = (
            time.monotonic() + self._timeout_seconds
            if self._timeout_seconds is not None
            else None
        )
        while True:
            programs = await self._storage.mget(pids, exclude=EXCLUDE_STAGE_RESULTS)
            found_ids = {p.id for p in programs}
            missing = [pid for pid in pids if pid not in found_ids]
            if missing:
                raise ValueError(
                    f"ParentRefresher: {len(missing)} parents vanished during refresh"
                )

            done: list[Program] = []
            still_active = 0
            for p in programs:
                if p.state == ProgramState.DONE:
                    done.append(p)
                elif p.state in (ProgramState.QUEUED, ProgramState.RUNNING):
                    still_active += 1
                elif p.state == ProgramState.DISCARDED:
                    raise ValueError(
                        f"ParentRefresher: parent {p.short_id} became DISCARDED "
                        "during refresh"
                    )

            if still_active == 0 and len(done) == len(pids):
                return done

            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(
                    f"ParentRefresher: timed out waiting for {still_active} parents"
                )

            await asyncio.sleep(self._poll_interval)


__all__ = [
    "DirectParentsSelector",
    "ParentRefresher",
    "ParentRefreshSelector",
    "ParentRefreshTicket",
]
