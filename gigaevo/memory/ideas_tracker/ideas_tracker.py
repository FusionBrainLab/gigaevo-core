"""
IdeaTracker: PostRunHook that authors memory cards from a completed
evolutionary run via the librarian write path (one card per mutation diff,
plus a clean card for each top-fitness exemplar).

The hook is a thin orchestration shell over four collaborators, mirroring the
reader's modular pipeline: a :class:`ProgramRecordExtractor` (eligible records +
dedup bookkeeping), a :class:`LibrarianWriteStack` (lazy store/gate/librarian +
task summary), a :class:`CardStatsUpdater` (gain attribution + restamp + harm
sweep), and a :class:`ConsolidationScheduler` (throttled background dedup).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from gigaevo.evolution.engine.hooks import IncrementalPostRunHook
from gigaevo.llm.models import MultiModelRouter
from gigaevo.memory.core.protocols import Evictor
from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.ideas_tracker.card_stats import CardStatsUpdater
from gigaevo.memory.ideas_tracker.consolidation_scheduler import ConsolidationScheduler
from gigaevo.memory.ideas_tracker.dedup_policy import DedupPolicy
from gigaevo.memory.ideas_tracker.fitness import select_top_programs
from gigaevo.memory.ideas_tracker.record_extractor import (
    ProgramRecordExtractor,
    record_note,
)
from gigaevo.memory.ideas_tracker.write_stack import LibrarianWriteStack
from gigaevo.memory.shared_memory.models import ProgramCard
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS, Program

if TYPE_CHECKING:
    from gigaevo.database.program_storage import ProgramStorage


class IdeaTracker(IncrementalPostRunHook):
    """
    PostRunHook that authors memory cards from a completed evolutionary run via
    the librarian write path: each eligible mutation diff is reconciled into
    clean idea cards, each top-fitness exemplar gets a program card, and one
    harm-eviction sweep runs per increment.

    Instantiated by Hydra. ``MemorySystem`` threads in the shared backend, llm
    router, and evictor so the writer and reader share one card bank.

    Args:
        llm: Memory LLM router for the librarian agents (Hydra
            ``memory/common/llm`` group, threaded in by ``MemorySystem``).
            Required when
            ``memory_write_enabled`` is True.
        memory_write_enabled: If True, run the librarian write path.
        memory_write_best_programs_percent: Share of top-fitness programs
            authored into program cards.
        ingest_call_timeout_s: Per-call wall-clock bound on each librarian LLM
            call (reconcile/author); a stalled call past this is skipped and the
            record retried on a later increment.
        consolidation_every_n: After this many cards are written across sweeps,
            schedule one background near-duplicate consolidation pass over the
            whole bank. 0 (default) disables it; ``librarian.yaml`` sets it on.
        dedup_policy: Near-duplicate thresholds for both the online pre-gate and
            the batch consolidation pass; defaults to ``DedupPolicy()``.
        backend: Backend builder (the Hydra ``memory/common/backend`` ``_partial_`` over
            ``build_local_backend``) used to build the card bank. Required
            whenever ``memory_write_enabled`` is True — ``MemorySystem`` threads
            in the same llm-bound partial it gave the read provider, so the
            writer and reader share one card bank.
        checkpoint_dir: Pins per-run memory cards under the Hydra output dir.
        task_description: Human-readable description of the current task; supplied
            by ``librarian.yaml`` from the run's problem context.
        prompts_dir: Optional prompts directory for the librarian agents (e.g.
            ``config.prompts.dir``); None uses the package default prompts.
        fitness_key: Metric key to use as fitness (default "fitness").
        fitness_higher_is_better: Sort direction for the exemplar slice and the
            sign of gain attribution.
        metrics_context: When wired, programs whose fitness equals the
            metric's sentinel value are excluded from records and posteriors.
        evictor: Harm evictor shared with the read provider; defaults to a
            reputation-backed ``HarmEvictor`` when not threaded in.
        reputation: Reputation model backing the default evictor.
    """

    def __init__(
        self,
        *,
        llm: MultiModelRouter | None = None,
        memory_write_enabled: bool = True,
        memory_write_best_programs_percent: float = 5.0,
        ingest_call_timeout_s: float = 300.0,
        consolidation_every_n: int = 0,
        dedup_policy: DedupPolicy | None = None,
        backend: Callable[..., Any] | None = None,
        checkpoint_dir: str | Path | None = None,
        task_description: str = "",
        prompts_dir: str | Path | None = None,
        fitness_key: str = "fitness",
        fitness_higher_is_better: bool = True,
        metrics_context: MetricsContext | None = None,
        evictor: Evictor | None = None,
        reputation: BetaBinomialReputation | None = None,
    ) -> None:
        if memory_write_enabled and backend is None:
            raise ValueError(
                "IdeaTracker: memory_write_enabled=True requires an explicit "
                "backend builder. Enable the writer via `memory=writer` or "
                "`memory=full` (MemorySystem assembles the backend and threads "
                "it in), or pass memory_write_enabled=False."
            )
        if memory_write_enabled and llm is None:
            raise ValueError(
                "IdeaTracker: memory_write_enabled=True requires an LLM router "
                "for the librarian. Enable the writer via `memory=writer` or "
                "`memory=full` (MemorySystem threads in the `memory/common/llm` "
                "router; swap it with `memory/common/llm=qwen_instruct`)."
            )

        self._memory_write_enabled = memory_write_enabled
        self._best_programs_percent = memory_write_best_programs_percent
        self._ingest_call_timeout_s = ingest_call_timeout_s
        self._fitness_key = fitness_key
        self._fitness_higher_is_better = fitness_higher_is_better
        self._metrics_context = metrics_context
        self._task_description = task_description
        policy = dedup_policy if dedup_policy is not None else DedupPolicy()

        # the live hook and the post-run hook share one tracker; overlapping
        # sweeps would interleave store writes and dedup bookkeeping
        self._run_lock = asyncio.Lock()
        self._stack = LibrarianWriteStack(
            backend=backend,
            llm=llm,
            task_description=self._task_description,
            evictor=evictor,
            reputation=reputation,
            checkpoint_dir=checkpoint_dir,
            dedup_policy=policy,
            prompts_dir=prompts_dir,
        )
        self._extractor = ProgramRecordExtractor(
            task_description=self._task_description,
            fitness_key=fitness_key,
            metrics_context=metrics_context,
        )
        self._stats = CardStatsUpdater(
            fitness_key=fitness_key,
            higher_is_better=fitness_higher_is_better,
            metrics_context=metrics_context,
        )
        self._consolidation = ConsolidationScheduler(
            stack=self._stack,
            run_lock=self._run_lock,
            every_n=consolidation_every_n,
            eps=policy.consolidation_eps,
            k=policy.consolidation_k,
        )

    # ------------------------------------------------------------------
    # PostRunHook interface
    # ------------------------------------------------------------------

    async def on_run_complete(self, storage: ProgramStorage) -> None:
        """Called by EvolutionEngine after the generation loop finishes."""
        programs = await storage.get_all(exclude=EXCLUDE_STAGE_RESULTS)
        if not programs:
            logger.warning("[Memory][IdeaTracker] no programs in storage, skipping.")
            return
        await self.run_increment(programs)
        # The final sweep may schedule a background consolidation pass; await it
        # here (the run lock is now free) so it completes before the event loop
        # is torn down rather than being cancelled at shutdown.
        await self._consolidation.drain(timeout=self._ingest_call_timeout_s)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    async def run_increment(
        self,
        programs: list[Program],
        *,
        posterior_programs: list[Program] | None = None,
    ) -> None:
        """Full pipeline: filter eligible records → reconcile each diff into
        cards → author exemplar cards → stamp gain events → harm-evict.

        ``programs`` feeds the expensive librarian agents and may be a bounded
        window (the live hook caps it to keep each sweep inside the engine's
        time budget). ``posterior_programs`` feeds the cheap, pure
        gain-event attribution, which needs the full program set so
        child→parent lineage resolves; a capped window would sever lineage and
        collapse the intro-event population. Defaults to ``programs`` when not
        supplied.
        """
        async with self._run_lock:
            await self._run_increment_locked(
                programs, posterior_programs=posterior_programs
            )

    async def _run_increment_locked(
        self,
        programs: list[Program],
        *,
        posterior_programs: list[Program] | None,
    ) -> None:
        if not self._memory_write_enabled:
            return
        await self._stack.ensure()
        summary = self._stack.task_description_summary
        records = self._extractor.extract(
            programs,
            task_description_summary=summary,
            posterior_programs=posterior_programs,
        )

        librarian = self._stack.require_librarian()
        cards_written = 0
        for rec in records:
            try:
                written = await asyncio.wait_for(
                    librarian.ingest_idea(
                        base_parent_code=rec.parent_code,
                        child_id=rec.id,
                        child_code=rec.code,
                        note=record_note(rec),
                    ),
                    timeout=self._ingest_call_timeout_s,
                )
                cards_written += len(written)
            except TimeoutError:
                # A stalled memory-LLM call must not starve the rest of the
                # sweep (sibling ingests, exemplars, harm eviction). Drop this
                # record so it is retried on a later increment, and continue.
                logger.warning(
                    "[Memory][IdeaTracker] ingest of {} timed out after {}s; "
                    "skipping record for retry next sweep",
                    rec.id,
                    self._ingest_call_timeout_s,
                )
                self._extractor.forget({rec.id})
                continue
            except BaseException:
                # CancelledError included: the record was marked seen before
                # ingest; without rollback the window's idea is lost.
                self._extractor.forget({rec.id})
                raise

        pool = programs if posterior_programs is None else posterior_programs
        await self._author_exemplars(pool, summary)

        # re-stamping (per-card store writes) and harm eviction are blocking
        # I/O; keep them off the event loop so in-flight mutations don't stall
        await asyncio.to_thread(
            self._stats.update,
            pool,
            store=self._stack.store,
            gate=self._stack.gate,
        )
        self._consolidation.note_writes(cards_written)

    async def _author_exemplars(self, pool: list[Program], summary: str) -> None:
        """Author a clean ProgramCard for each top-fitness exemplar.

        ``author_program`` is cached on ``program-<id>`` so a re-selected
        exemplar never re-pays the LLM; the gate re-admits the card (its
        gain events are restamped immediately after from the full pool).
        """
        selected = select_top_programs(
            pool,
            best_programs_percent=self._best_programs_percent,
            fitness_key=self._fitness_key,
            higher_is_better=self._fitness_higher_is_better,
            metrics_context=self._metrics_context,
        )
        librarian = self._stack.require_librarian()
        for prog, fitness in selected:
            try:
                authored = await asyncio.wait_for(
                    librarian.author_program(
                        program_id=prog.id,
                        code=prog.code,
                        fitness=fitness,
                    ),
                    timeout=self._ingest_call_timeout_s,
                )
            except TimeoutError:
                logger.warning(
                    "[Memory][IdeaTracker] authoring exemplar {} timed out "
                    "after {}s; skipping",
                    prog.id,
                    self._ingest_call_timeout_s,
                )
                continue
            except Exception as exc:
                # A non-timeout authoring failure (API/schema error) must
                # degrade per-exemplar like the idea path, not abort the
                # increment before the stats restamp and harm sweep. Cancellation
                # (BaseException) still propagates.
                logger.warning(
                    "[Memory][IdeaTracker] authoring exemplar {} failed ({}); skipping",
                    prog.id,
                    exc,
                )
                continue
            librarian.admit_program(
                ProgramCard(
                    id=f"program-{prog.id}",
                    program_id=prog.id,
                    task_description=self._task_description,
                    task_description_summary=summary,
                    description=authored.description,
                    explanation_summary=authored.explanation_summary,
                    fitness=fitness,
                    code=prog.code,
                    keywords=authored.keywords or [],
                ),
                higher_is_better=self._fitness_higher_is_better,
            )
