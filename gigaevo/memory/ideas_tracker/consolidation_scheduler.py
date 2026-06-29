"""ConsolidationScheduler: throttled background near-duplicate consolidation.

Counts cards written across sweeps and dispatches exactly one background
consolidation pass per ``every_n``. The pass runs under the shared write lock so
it can never interleave with a live write sweep, yet dispatch is non-blocking so
the sweep that triggered it returns immediately.
"""

from __future__ import annotations

import asyncio

from loguru import logger

from gigaevo.memory.core.events import emit_memory_event
from gigaevo.memory.ideas_tracker.consolidation import consolidate
from gigaevo.memory.ideas_tracker.write_stack import LibrarianWriteStack

# Grace given to a cancelled final pass to unwind before it is abandoned as an
# orphan, so a pass that swallows CancelledError cannot wedge engine teardown.
_CANCEL_GRACE_S = 1.0


class ConsolidationScheduler:
    """Throttles and serializes background bank consolidation."""

    def __init__(
        self,
        *,
        stack: LibrarianWriteStack,
        run_lock: asyncio.Lock,
        every_n: int,
        eps: float,
        k: int,
    ) -> None:
        self._stack = stack
        self._run_lock = run_lock
        self._every_n = every_n
        self._eps = eps
        self._k = k
        self._writes_since = 0
        self._failures = 0
        self._task: asyncio.Task | None = None

    @property
    def writes_since(self) -> int:
        return self._writes_since

    @property
    def failures(self) -> int:
        return self._failures

    @property
    def task(self) -> asyncio.Task | None:
        return self._task

    def note_writes(self, written: int) -> None:
        """Accumulate cards written and schedule one consolidation pass per
        ``every_n``. The cadence counter is consumed only when a pass is actually
        dispatched, so a dispatch that cannot run (un-built stack, a pass already
        in flight, or no running loop) leaves the writes pending for a later
        sweep rather than silently disabling consolidation."""
        if self._every_n <= 0 or written <= 0:
            return
        self._writes_since += written
        if self._writes_since >= self._every_n and self.schedule():
            self._writes_since = 0

    def schedule(self) -> bool:
        """Dispatch one background consolidation pass. Returns True iff a task was
        actually created; False when the stack is un-built, a pass is in flight,
        or there is no running loop."""
        if self._stack.store is None or self._stack.gate is None:
            return False
        if self._task is not None and not self._task.done():
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False  # no running loop (sync context); defer to a later increment
        self._task = loop.create_task(self._run())
        return True

    async def drain(self, *, timeout: float | None = None) -> None:
        """Await an in-flight consolidation pass so a pass scheduled by the final
        post-run sweep completes before the event loop is torn down — ``asyncio.run``
        cancels pending tasks on exit, which would otherwise silently drop the last
        consolidation. Bounded by ``timeout`` so a stalled memory-LLM call in that
        final pass cannot hang engine shutdown; on overrun the pass is cancelled."""
        task = self._task
        if task is None or task.done():
            return
        # asyncio.wait (not wait_for) for the cancel step: a pass that swallows
        # CancelledError is abandoned as an orphan after a short grace instead of
        # blocking teardown forever waiting for the cancel to be honoured.
        _, pending = await asyncio.wait({task}, timeout=timeout)
        if not pending:
            return
        task.cancel()
        _, still_pending = await asyncio.wait({task}, timeout=_CANCEL_GRACE_S)
        if still_pending:
            logger.warning(
                "[Memory][IdeaTracker] final consolidation pass ignored cancel "
                "within {}s grace; abandoned as orphan at shutdown",
                _CANCEL_GRACE_S,
            )
        else:
            logger.warning(
                "[Memory][IdeaTracker] final consolidation pass exceeded {}s; "
                "cancelled at shutdown",
                timeout,
            )

    async def _run(self) -> None:
        # Consolidation rewrites the bank, so it runs under the same write lock as
        # a sweep — never interleaved with one — but is dispatched in the
        # background so the triggering sweep is not blocked waiting for it.
        neighbors = self._stack.neighbors
        if neighbors is None:
            return  # un-built stack (schedule() guards this) — nothing to fold
        async with self._run_lock:
            try:
                merged = await consolidate(
                    store=self._stack.store,
                    gate=self._stack.gate,
                    neighbors=neighbors,
                    agent=self._stack.consolidation_agent,
                    eps=self._eps,
                    k=self._k,
                )
            except Exception as exc:
                self._failures += 1
                logger.warning(
                    "[Memory][IdeaTracker] consolidation pass failed ({}); skipping",
                    exc,
                )
                emit_memory_event(
                    component="consolidation",
                    event_type="consolidation.failed",
                    payload={"error": str(exc), "failures": self._failures},
                    level="WARNING",
                )
                return
            if merged:
                logger.info(
                    "[Memory][IdeaTracker] consolidation merged {} near-dup cards",
                    merged,
                )
            emit_memory_event(
                component="consolidation",
                event_type="consolidation.pass",
                payload={"merged": merged},
            )
