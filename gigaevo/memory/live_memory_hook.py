"""Live mid-run refresh hook for the IdeaTracker / external-memory pipeline.

Wraps an :class:`IdeaTracker` instance as a ``post_step_hook`` so the global
memory bank is rebuilt mid-run instead of only once after ``on_run_complete``.

Cadence is counted in *ingestor sweeps that landed at least one program* —
the steady-state engine has no notion of "generations", and the ingestor only
fires its post-step hook when ``added > 0`` (see
``gigaevo/evolution/engine/ingestor.py``). One sweep ≈ one ingest of a fresh
batch of mutants, so this approximates "every K newly-evaluated programs".

The hook signature matches what :class:`EvolutionEngine` expects::

    Callable[[], Awaitable[None]]

Fault isolation, wall-clock bounds and cancel-grace are provided by the
engine's ``_run_bounded_post_step_hook`` wrapper. This class logs each refresh
failure on its own channel (with a consecutive-failure count, so a silently
dying memory bank is attributable from the run log) and re-raises — the engine
keeps owning fault isolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from gigaevo.evolution.engine.hooks import IncrementalPostRunHook
from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS

if TYPE_CHECKING:
    from gigaevo.database.program_storage import ProgramStorage


class LiveMemoryRefreshHook:
    """Cadence-gated wrapper around :meth:`IdeaTracker.run_increment`.

    Args:
        tracker: The :class:`IncrementalPostRunHook` (in practice an
            :class:`IdeaTracker`) whose state should be refreshed live.
        storage: ProgramStorage to pull the current program set from.
        refresh_every: Number of post-step invocations between refreshes.
            ``1`` refreshes on every ingestor sweep that lands a program.
            Defaults to ``10``.
        max_programs_per_sweep: If set, cap the LLM *analyzer* window to the N
            newest programs by ``created_at`` so each sweep stays inside the
            engine's time budget; ``None`` (default) is unbounded. The injection
            posterior always sees the full program set regardless of this cap —
            it needs intact lineage to resolve child→parent improvement events.
    """

    def __init__(
        self,
        *,
        tracker: IncrementalPostRunHook,
        storage: ProgramStorage,
        refresh_every: int = 10,
        max_programs_per_sweep: int | None = None,
    ) -> None:
        if not isinstance(tracker, IncrementalPostRunHook):
            raise TypeError(
                "LiveMemoryRefreshHook needs an IncrementalPostRunHook (got "
                f"{type(tracker).__name__}); pipeline=intra_extra_memory requires "
                "the co-overrides `ideas_tracker=default memory=local`"
            )
        self._tracker = tracker
        self._storage = storage
        self._refresh_every = max(1, int(refresh_every))
        self._max_programs_per_sweep = (
            None
            if max_programs_per_sweep is None
            else max(1, int(max_programs_per_sweep))
        )
        self._sweep_counter = 0
        self._last_refresh_sweep = 0
        self._consecutive_failures = 0

    @property
    def consecutive_failures(self) -> int:
        """Refresh failures since the last successful refresh."""
        return self._consecutive_failures

    async def __call__(self) -> None:
        self._sweep_counter += 1
        if self._sweep_counter - self._last_refresh_sweep < self._refresh_every:
            return
        programs = await self._storage.get_all(exclude=EXCLUDE_STAGE_RESULTS)
        if not programs:
            logger.debug(
                "[Memory][LiveRefresh] sweep={} storage empty, skipping refresh",
                self._sweep_counter,
            )
            self._last_refresh_sweep = self._sweep_counter
            return
        total = len(programs)
        window = programs
        if (
            self._max_programs_per_sweep is not None
            and total > self._max_programs_per_sweep
        ):
            window = sorted(programs, key=lambda p: p.created_at, reverse=True)[
                : self._max_programs_per_sweep
            ]
        logger.info(
            "[Memory][LiveRefresh] sweep={} programs={}/{} refreshing memory bank",
            self._sweep_counter,
            len(window),
            total,
        )
        try:
            await self._tracker.run_increment(window, posterior_programs=programs)
        except BaseException as exc:
            # BaseException so CancelledError is counted and logged too.
            # _last_refresh_sweep is deliberately NOT advanced: the next
            # landing sweep retries instead of waiting a full cadence window.
            self._consecutive_failures += 1
            logger.error(
                "[Memory][LiveRefresh] refresh FAILED at sweep {} ({} consecutive): {}",
                self._sweep_counter,
                self._consecutive_failures,
                repr(exc),
            )
            raise
        self._consecutive_failures = 0
        self._last_refresh_sweep = self._sweep_counter
