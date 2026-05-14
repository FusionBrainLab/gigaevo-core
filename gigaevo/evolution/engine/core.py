from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import contextlib
import time
from typing import Any

from loguru import logger

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.evolution.engine.config import EngineConfig
from gigaevo.evolution.engine.hooks import NullPostRunHook, PostRunHook
from gigaevo.evolution.engine.metrics import EngineMetrics
from gigaevo.evolution.engine.snapshot import (
    ENGINE_SNAPSHOT_KEY,
    EngineSnapshot,
    load_engine_snapshot,
    set_current_snapshot,
)
from gigaevo.evolution.engine.stopper import (
    EvolutionStopper,
    StopContext,
)
from gigaevo.evolution.mutation.base import MutationOperator
from gigaevo.evolution.mutation.mutation_operator import (
    LLMMutationOperator,
)
from gigaevo.evolution.strategies.base import EvolutionStrategy
from gigaevo.llm.bandit import BanditModelRouter, MutationOutcome
from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS, Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.utils.metrics_collector import start_metrics_collector
from gigaevo.utils.metrics_tracker import MetricsTracker
from gigaevo.utils.trackers.base import LogWriter


class EvolutionEngine:
    """Abstract base for evolution engines.

    Provides shared helpers (snapshot, metrics, idle wait, hooks, stop
    context) consumed by :class:`SteadyStateEvolutionEngine`. The concrete
    loop is provided by the subclass; this base intentionally raises
    ``NotImplementedError`` from ``run()`` so it cannot be instantiated as
    a standalone engine.

    See ``docs/superpowers/specs/2026-05-12-steady-state-engine-audit-and-redesign.md``.
    """

    def __init__(
        self,
        storage: ProgramStorage,
        strategy: EvolutionStrategy,
        mutation_operator: MutationOperator,
        config: EngineConfig,
        writer: LogWriter,
        metrics_tracker: MetricsTracker,
        pre_step_hook: Callable[[], Awaitable[None]] | None = None,
        post_run_hook: PostRunHook | None = None,
        post_step_hook: Callable[[], Awaitable[None]] | None = None,
    ):
        self.storage = storage
        self.strategy = strategy
        self.mutation_operator = mutation_operator
        self.config = config
        self._writer = writer.bind(path=["evolution_engine"])

        self._running = False
        self._last_pending_dags_counts: tuple[int, int] | None = None

        self._task: asyncio.Task | None = None
        self._metrics_collector_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # ETA tracking — set at the start of run()
        self._run_start_time: float | None = None

        self.metrics = EngineMetrics()
        self.state = ProgramStateManager(self.storage)
        self._metrics_tracker = metrics_tracker
        self._pre_step_hook = pre_step_hook
        self._post_step_hook = post_step_hook
        self._post_run_hook = post_run_hook or NullPostRunHook()

        self._snapshot: EngineSnapshot = EngineSnapshot()
        # Serialises _write_snapshot so the version+Redis writes from
        # concurrent mutant tasks land in monotone order. Without this lock,
        # T1 may compute v=N+1 and await save; T2 may compute v=N+2 and
        # award save; if T2's save lands first, Redis ends at v=N+1 with
        # stale fields and a crash resume rehydrates a snapshot older than
        # the in-memory mirror.
        self._snapshot_lock: asyncio.Lock = asyncio.Lock()

        logger.info(
            "[EvolutionEngine] Init | strategy={}, acceptor={}, stopper={}",
            type(self.strategy).__name__,
            type(self.config.program_acceptor).__name__,
            type(self.stopper).__name__,
        )

    def start(self) -> None:
        """Start the evolution engine in a background task."""
        if self._task and not self._task.done():
            return
        self._loop = asyncio.get_running_loop()
        self._running = True
        self._task = asyncio.create_task(self.run(), name="evolution-engine")
        self._metrics_tracker.start(self._loop)

        async def _collect_metrics() -> dict[str, Any]:
            out = self.metrics.model_dump(mode="json")
            strategy_metrics = await self.strategy.get_metrics()
            if strategy_metrics:
                out.update(strategy_metrics.to_dict())
            if isinstance(self.mutation_operator, LLMMutationOperator) and isinstance(
                self.mutation_operator.llm_wrapper, BanditModelRouter
            ):
                out["bandit"] = self.mutation_operator.llm_wrapper.get_bandit_stats()
            return out

        self._metrics_collector_task = start_metrics_collector(
            writer=self._writer,
            collect_fn=_collect_metrics,
            interval=self.config.metrics_collection_interval,
            stop_flag=lambda: not self._running,
            task_name="evolution-metrics-collector",
        )
        logger.info("[EvolutionEngine] Task started")

    async def stop(self) -> None:
        """Stop the evolution engine and await task completion."""
        self._running = False
        task = self._task
        self._task = None
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        if self._metrics_collector_task:
            # Await the cancel — without this, the collector may still be
            # mid `await storage.<call>` when `storage.close()` fires below,
            # raising ConnectionClosedError into a coroutine that has no
            # caller to surface it. Bound the wait so a wedged collector
            # cannot indefinitely block shutdown.
            self._metrics_collector_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(self._metrics_collector_task, timeout=2.0)
            self._metrics_collector_task = None

        if self._metrics_tracker:
            await self._metrics_tracker.stop()

        await self.storage.close()

    @property
    def task(self) -> asyncio.Task | None:
        return self._task

    async def run(self) -> None:
        """Abstract — subclasses provide the loop.

        Use :class:`SteadyStateEvolutionEngine` (the only concrete engine).
        """
        raise NotImplementedError(
            "EvolutionEngine.run() is abstract — use SteadyStateEvolutionEngine."
        )

    async def _await_idle(self) -> None:
        """Block until there are no programs in QUEUED or RUNNING."""
        t0 = time.monotonic()
        ghost_checked = False
        while True:
            has_active = await self._has_active_dags()
            if not has_active:
                break

            elapsed = time.monotonic() - t0
            if elapsed > 30 and int(elapsed) % 60 < self.config.loop_interval:
                logger.info(
                    "[EvolutionEngine] mutants={} Waiting for idle ({:.0f}s elapsed)",
                    self.metrics.iteration,
                    elapsed,
                )
            # Ghost safety: after 30s, verify counts with full fetch (once)
            if elapsed > 30 and not ghost_checked:
                ghost_checked = True
                real_q = len(
                    await self.storage.get_all_by_status(
                        ProgramState.QUEUED.value,
                        exclude=EXCLUDE_STAGE_RESULTS,
                    )
                )
                real_r = len(
                    await self.storage.get_all_by_status(
                        ProgramState.RUNNING.value,
                        exclude=EXCLUDE_STAGE_RESULTS,
                    )
                )
                if real_q == 0 and real_r == 0:
                    # Clean up ghost IDs from status sets
                    queued_ids = await self.storage.get_ids_by_status(
                        ProgramState.QUEUED.value
                    )
                    running_ids = await self.storage.get_ids_by_status(
                        ProgramState.RUNNING.value
                    )
                    if queued_ids:
                        await self.storage.remove_ids_from_status_set(
                            ProgramState.QUEUED.value, queued_ids
                        )
                    if running_ids:
                        await self.storage.remove_ids_from_status_set(
                            ProgramState.RUNNING.value, running_ids
                        )
                    ghost_count = len(queued_ids) + len(running_ids)
                    logger.warning(
                        "[EvolutionEngine] Ghost IDs detected — SCARD says active "
                        "but no real programs found. Cleaned {} ghost ID(s) from "
                        "status sets. Breaking idle wait.",
                        ghost_count,
                    )
                    break
            await asyncio.sleep(self.config.loop_interval)

    async def _select_parents_for_mutation(self) -> list[Program]:
        # In steady-state, every iteration mutates exactly one parent group, so
        # we ask the strategy for ``num_parents`` elites and treat the response
        # as the parent set. May return fewer when the archive is smaller than
        # ``num_parents`` (early run, aggressive rejection) — the mutation
        # operator decides whether single-parent mutation is acceptable.
        num_parents = self.config.parent_selector.num_parents
        parents = await self.strategy.select_elites(total=num_parents)
        logger.debug(
            "[EvolutionEngine] mutants={} Parents selected: {}",
            self.metrics.iteration,
            len(parents),
        )
        self.metrics.elites_selected += len(parents)
        return parents

    async def _ingest_completed_programs(self) -> None:
        """Validate and hand over any DONE programs to the strategy.

        Programs already in the archive stay DONE (they arrived from a
        refresh DAG). New programs are added if accepted, otherwise
        discarded.
        """
        # Fetch only IDs first (SMEMBERS — no deserialization), then filter
        # out archive programs before doing the expensive mget+deserialize.
        done_ids = await self.storage.get_ids_by_status(ProgramState.DONE.value)
        if not done_ids:
            logger.debug(
                "[EvolutionEngine] mutants={} No completed programs to ingest",
                self.metrics.iteration,
            )
            return

        archive_program_ids = set(await self.strategy.get_program_ids())
        new_ids = [pid for pid in done_ids if pid not in archive_program_ids]

        if not new_ids:
            logger.debug(
                "[EvolutionEngine] mutants={} {} DONE programs all in archive, skipping",
                self.metrics.iteration,
                len(done_ids),
            )
            return

        # Only deserialize the new (non-archive) programs.
        # Exclude stage_results (~10% of payload) — ingestion only needs
        # metrics, state, metadata, and lineage.  The merge strategy in
        # storage.update() preserves existing stage_results from Redis.
        completed = await self.storage.mget(new_ids, exclude=EXCLUDE_STAGE_RESULTS)
        # Filter to actual DONE state (mget may return stale status)
        completed = [p for p in completed if p.state == ProgramState.DONE]

        if not completed:
            return

        logger.info(
            "[EvolutionEngine] mutants={} Ingest {} program(s) ({} in archive skipped)",
            self.metrics.iteration,
            len(completed),
            len(done_ids) - len(new_ids),
        )
        logger.debug(
            "[EvolutionEngine] Program IDs: {}",
            [p.short_id for p in completed[:8]]
            + (["..."] if len(completed) > 8 else []),
        )

        added = 0
        rej_valid = 0
        rej_strategy = 0

        # Collect IDs of rejected programs for a single batch transition
        # at the end, instead of one Redis write per reject.
        reject_ids: list[str] = []

        for prog in completed:
            try:
                if not self.config.program_acceptor.is_accepted(prog):
                    # rejected by basic checks
                    rej_valid += 1
                    logger.info(
                        "[EvolutionEngine] Program {} REJECTED by acceptor (metrics={})",
                        prog.short_id,
                        prog.metrics,
                    )
                    await self._notify_hook(prog, MutationOutcome.REJECTED_ACCEPTOR)
                    reject_ids.append(prog.id)
                elif await self.strategy.add(prog):
                    # accepted by strategy — stays DONE until next refresh
                    added += 1
                    await self._notify_hook(prog, MutationOutcome.ACCEPTED)
                    logger.debug(
                        "[EvolutionEngine] Program {} added to strategy (metrics={})",
                        prog.short_id,
                        prog.metrics,
                    )
                else:
                    # rejected by strategy / validation
                    rej_strategy += 1
                    logger.debug(
                        "[EvolutionEngine] Program {} rejected by strategy (metrics={})",
                        prog.short_id,
                        prog.metrics,
                    )
                    await self._notify_hook(prog, MutationOutcome.REJECTED_STRATEGY)
                    reject_ids.append(prog.id)
            except Exception as e:
                # Isolate per-program failures: log and discard the offending program
                # so the remaining programs in this batch are still processed.
                logger.error(
                    "[EvolutionEngine] Ingestion failed for program {}: {} — discarding",
                    prog.short_id,
                    e,
                )
                reject_ids.append(prog.id)

        # Batch DONE → DISCARDED (raw JSON patch, no Pydantic serialization).
        # Also update in-memory state so any downstream code sees DISCARDED.
        if reject_ids:
            reject_set = set(reject_ids)
            for prog in completed:
                if prog.id in reject_set:
                    prog.state = ProgramState.DISCARDED
            try:
                await self.storage.batch_transition_by_ids(
                    reject_ids,
                    ProgramState.DONE.value,
                    ProgramState.DISCARDED.value,
                )
            except Exception as e:
                logger.error(
                    "[EvolutionEngine] batch discard failed for {} programs: {}",
                    len(reject_ids),
                    e,
                )

        self.metrics.programs_processed += added
        self.metrics.record_ingestion_metrics(added, rej_valid, rej_strategy)
        logger.info(
            "[EvolutionEngine] mutants={} Ingest done | added={}, rejected_validation={}, rejected_strategy={}",
            self.metrics.iteration,
            added,
            rej_valid,
            rej_strategy,
        )

    async def _has_active_dags(self) -> bool:
        """True if any programs are QUEUED or RUNNING (i.e., engine not idle).

        Uses count_by_status (SCARD, O(1)) for the fast path.  Falls back to
        the expensive get_all_by_status after 30s of continuous waiting to
        detect ghost IDs that would otherwise stall _await_idle forever.
        """
        queued, running = await asyncio.gather(
            self.storage.count_by_status(ProgramState.QUEUED.value),
            self.storage.count_by_status(ProgramState.RUNNING.value),
        )

        if queued or running:
            current_counts = (queued, running)
            if self._last_pending_dags_counts != current_counts:
                logger.debug(
                    "[EvolutionEngine] Pending DAGs: queued={}, running={}",
                    queued,
                    running,
                )
                self._last_pending_dags_counts = current_counts
            return True

        self._last_pending_dags_counts = None
        return False

    async def _notify_hook(self, prog: Program, outcome: MutationOutcome) -> None:
        """Call on_program_ingested with fault isolation.

        Hook failures are non-fatal: they must never cause a program that was
        already accepted by the strategy to be discarded (which would create a
        ghost entry in the archive).
        """
        try:
            await self.mutation_operator.on_program_ingested(
                prog, self.storage, outcome=outcome
            )
        except Exception as exc:
            logger.warning(
                "[EvolutionEngine] on_program_ingested hook failed for {}: {} "
                "(non-fatal, program state unchanged)",
                prog.short_id,
                exc,
            )

    async def _write_snapshot(self, **updates: Any) -> None:
        """Merge fields into the snapshot, bump version, persist to Redis,
        and mirror into the process-wide sync cache.

        Last-writer-wins — the engine is single-process async with a single
        writer coroutine.

        Persist-then-mirror ordering: if ``save_run_state`` raises, the
        in-memory mirror keeps its current version so the next call retries
        the same version number — Redis never sees a version skip. The
        mirror is thus always ``≤`` Redis, which is fine because Redis is
        the source of truth on resume.
        """
        # EngineSnapshot is frozen; rebuild via model_copy instead of in-place mutation.
        # Lock serialises concurrent writes so the Redis-persisted version
        # matches the in-memory mirror — without it, races between two mutant
        # tasks can leave Redis on an older version than the mirror, and a
        # crash resume would rehydrate a snapshot that has lost updates.
        async with self._snapshot_lock:
            next_snapshot = self._snapshot.model_copy(
                update={**updates, "version": self._snapshot.version + 1}
            )
            await self.storage.save_run_state(
                ENGINE_SNAPSHOT_KEY, next_snapshot.model_dump_json()
            )
            # Only mirror after Redis confirms — on failure the line above
            # raises and the mirror keeps the prior version, so the next
            # call retries cleanly.
            self._snapshot = next_snapshot
            set_current_snapshot(next_snapshot)

    async def _load_snapshot_on_resume(self) -> None:
        """Hydrate ``self._snapshot`` from Redis during engine startup."""
        self._snapshot = await load_engine_snapshot(self.storage)
        set_current_snapshot(self._snapshot)

    async def restore_state(self) -> None:
        """Restore iteration and programs_processed from storage after a resume.

        Note: EngineSnapshot still spells the persisted field ``total_mutants``;
        the rename to ``iteration`` will land in a follow-up slice (#232).
        """
        await self._load_snapshot_on_resume()
        self.metrics.iteration = self._snapshot.total_mutants
        self.metrics.programs_processed = self._snapshot.programs_processed
        logger.info(
            "[EvolutionEngine] Restored iteration={} programs_processed={}",
            self._snapshot.total_mutants,
            self._snapshot.programs_processed,
        )

    @property
    def stopper(self) -> EvolutionStopper:
        return self.config.stopper

    def _build_stop_context(self) -> StopContext:
        elapsed = (
            time.monotonic() - self._run_start_time
            if self._run_start_time is not None
            else 0.0
        )
        return StopContext(
            total_mutants=self.metrics.iteration,
            elapsed_seconds=elapsed,
            best_fitness=self._metrics_tracker.get_best_fitness(),
            programs_processed=self.metrics.programs_processed,
        )

    def _reached_mutant_cap(self) -> bool:
        return self.stopper.should_stop(self._build_stop_context()).stop
