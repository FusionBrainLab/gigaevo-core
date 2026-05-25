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
engine's ``_run_bounded_post_step_hook`` wrapper, so this class deliberately
does no try/except of its own — failures propagate and the engine handles
them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS

if TYPE_CHECKING:
    from gigaevo.database.program_storage import ProgramStorage
    from gigaevo.memory.ideas_tracker.ideas_tracker import IdeaTracker


class LiveMemoryRefreshHook:
    """Cadence-gated wrapper around :meth:`IdeaTracker.run_increment`.

    Args:
        tracker: The :class:`IdeaTracker` whose state should be refreshed live.
        storage: ProgramStorage to pull the current program set from.
        refresh_every: Number of post-step invocations between refreshes.
            ``1`` refreshes on every ingestor sweep that lands a program.
            Defaults to ``10``.
        max_programs_per_sweep: If set, cap each refresh to the N newest
            programs by ``created_at``; ``None`` (default) is unbounded.
    """

    def __init__(
        self,
        *,
        tracker: IdeaTracker,
        storage: ProgramStorage,
        refresh_every: int = 10,
        max_programs_per_sweep: int | None = None,
    ) -> None:
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
        if (
            self._max_programs_per_sweep is not None
            and total > self._max_programs_per_sweep
        ):
            programs = sorted(programs, key=lambda p: p.created_at, reverse=True)[
                : self._max_programs_per_sweep
            ]
        logger.info(
            "[Memory][LiveRefresh] sweep={} programs={}/{} refreshing memory bank",
            self._sweep_counter,
            len(programs),
            total,
        )
        await self._tracker.run_increment(programs)
        self._last_refresh_sweep = self._sweep_counter
