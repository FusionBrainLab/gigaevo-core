"""MemoryWriter: PostRunHook that authors memory cards from a completed run.

The hook is a thin orchestration shell over four collaborators, mirroring the
reader's modular pipeline: a :class:`ProgramRecordExtractor` (eligible records +
dedup bookkeeping), a :class:`LibrarianWriteStack` (the shared store + a lazy
gate/librarian + task summary), a :class:`CardStatsUpdater` (gain attribution +
restamp + configured eviction), and a :class:`ConsolidationScheduler` (throttled
background dedup).
There is no enable flag — when the writer is off the config wires
``NullPostRunHook`` instead.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from loguru import logger

from gigaevo.evolution.engine.hooks import IncrementalPostRunHook
from gigaevo.llm.agents.consolidate_cards import ConsolidateAgent
from gigaevo.llm.agents.factories import (
    create_consolidate_agent,
    create_novelty_admission_agent,
    create_program_author_agent,
    create_reconcile_agent,
    create_task_summary_agent,
)
from gigaevo.llm.models import MultiModelRouter
from gigaevo.memory.cards import Card, CardKind
from gigaevo.memory.context.no_card import NoCardEvidenceRecorder
from gigaevo.memory.prior_evidence import EvictedEvidenceSink
from gigaevo.memory.selection_leases import InFlightSelectionRegistry
from gigaevo.memory.storage.base import MemoryStore
from gigaevo.memory.write.admission import (
    CardAdmissionGate,
    WriteLedger,
    WriteOutcome,
    WriteResult,
)
from gigaevo.memory.write.consolidation import ConsolidationScheduler
from gigaevo.memory.write.crediting import EffectEstimator
from gigaevo.memory.write.eviction import Evictor
from gigaevo.memory.write.extraction import ProgramRecordExtractor, record_note
from gigaevo.memory.write.librarian import Librarian, NeighborSource, code_sha256
from gigaevo.memory.write.merge import DedupPolicy, ProgramExemplarPolicy
from gigaevo.memory.write.stats import CardStatsUpdater, NoCardBaselineEstimator
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS, Program

if TYPE_CHECKING:
    from gigaevo.database.program_storage import ProgramStorage
    from gigaevo.memory.ope.reporter import MemoryOpeReporter


@runtime_checkable
class CardEvidenceUpdater(Protocol):
    """Writer-side maintenance after content proposals have landed."""

    def update(
        self, pool: list[Program], *, store: MemoryStore, gate: CardAdmissionGate
    ) -> None: ...


async def _shielded_to_thread(func, *args, cancel_log: str, **kwargs):
    """Run blocking work off-loop, but do not release caller locks mid-write.

    Cancelling ``asyncio.to_thread`` cancels only the awaiter; the worker thread
    keeps running. The memory writer holds an async run lock around store
    mutations, so on cancellation we wait for the thread to finish before
    propagating ``CancelledError`` and releasing that lock.
    """
    task = asyncio.create_task(asyncio.to_thread(func, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        logger.warning("{}; waiting for blocking memory work to finish", cancel_log)
        try:
            await task
        except Exception as exc:
            logger.warning(
                "[Memory][Writer] blocking memory work failed after cancellation: {}",
                exc,
            )
        raise


class LibrarianWriteStack:
    """Builds and holds the write-path components over the shared store.

    A thin holder over the one ``MemoryStore`` the whole run shares (injected —
    the reader reads the same instance, so a write is visible to the next read
    with no cross-view reconciliation). It builds the admission gate, neighbor
    source, librarian, and consolidation agent once, off the event loop, on
    first use. It also condenses the task description into the one-line summary
    stamped on every card — folded into :meth:`ensure` so there is no
    summary-before-stack ordering rule for the orchestrator to honour.
    """

    def __init__(
        self,
        *,
        llm: MultiModelRouter,
        evictor: Evictor,
        store: MemoryStore,
        checkpoint_dir: str | Path,
        task_key: str = "",
        task_description: str = "",
        dedup_policy: DedupPolicy | None = None,
        prompts_dir: str | Path | None = None,
        novelty_admission_gate: bool = False,
        selection_leases: InFlightSelectionRegistry | None = None,
        min_effective_events: float = 0.0,
        max_task_cards: int | None = None,
        evicted_evidence: EvictedEvidenceSink | None = None,
    ) -> None:
        self._llm = llm
        self._evictor = evictor
        self._store = store
        self._checkpoint_dir = Path(checkpoint_dir)
        self._task_key = task_key
        self._task_description = task_description
        self._dedup_policy = dedup_policy if dedup_policy is not None else DedupPolicy()
        self._prompts_dir = prompts_dir
        self._novelty_admission_gate = novelty_admission_gate
        self._selection_leases = selection_leases
        self._min_effective_events = min_effective_events
        self._max_task_cards = max_task_cards
        self._preserve_survivor_payload = self._dedup_policy.preserve_survivor_payload
        self._evicted_evidence = evicted_evidence
        self._gate: CardAdmissionGate | None = None
        self._librarian: Librarian | None = None
        self._neighbors: NeighborSource | None = None
        self._consolidation_agent: ConsolidateAgent | None = None
        # genuine LLM-condensed one-liner, produced once per run; None until then
        # so the call is memoised.
        self._summary: str | None = None
        self._build_lock = asyncio.Lock()

    @property
    def store(self) -> MemoryStore | None:
        return self._store

    def require_store(self) -> MemoryStore:
        if self._store is None:
            raise RuntimeError(
                "LibrarianWriteStack.require_store() called before ensure(); "
                "await the build before writing."
            )
        return self._store

    @property
    def gate(self) -> CardAdmissionGate | None:
        return self._gate

    def require_gate(self) -> CardAdmissionGate:
        if self._gate is None:
            raise RuntimeError(
                "LibrarianWriteStack.require_gate() called before ensure(); "
                "await the build before writing."
            )
        return self._gate

    @property
    def librarian(self) -> Librarian | None:
        return self._librarian

    def require_librarian(self) -> Librarian:
        if self._librarian is None:
            raise RuntimeError(
                "LibrarianWriteStack.require_librarian() called before ensure(); "
                "await the build before writing."
            )
        return self._librarian

    @property
    def neighbors(self) -> NeighborSource | None:
        return self._neighbors

    @property
    def consolidation_agent(self) -> ConsolidateAgent | None:
        return self._consolidation_agent

    @property
    def task_description_summary(self) -> str:
        return self._summary or ""

    async def ensure(self) -> None:
        """Build the store, gate, neighbors, librarian, and consolidation agent
        once. The build loads the embedding model (seconds of blocking I/O) so it
        runs off the event loop; the lock collapses a concurrent first-write race
        down to a single build."""
        if self._librarian is not None:
            return
        async with self._build_lock:
            if self._librarian is not None:
                return
            summary = await self.ensure_summary()
            await _shielded_to_thread(
                self._build,
                summary,
                cancel_log="[Memory][Writer] stack build was cancelled",
            )

    async def ensure_summary(self) -> str:
        """Condense the task description into a one-line summary, once per run.

        Falls back to the full task description on any LLM failure (and to the
        empty string when there is no task text), so a memory-LLM hiccup can
        never block the write path.
        """
        if self._summary is not None:
            return self._summary
        if not self._task_description:
            self._summary = ""
            return self._summary
        try:
            agent = create_task_summary_agent(self._llm, prompts_dir=self._prompts_dir)
            resp = await agent.arun(task_description=self._task_description)
            self._summary = resp.summary.strip() or self._task_description
        except Exception as exc:
            logger.warning(
                "[Memory][Writer] task-summary LLM failed ({}); falling back "
                "to the full task description",
                exc,
            )
            self._summary = self._task_description
        return self._summary

    def _build(self, summary: str) -> None:
        policy = self._dedup_policy
        store = self._store
        gate = CardAdmissionGate(
            store=store,
            evictor=self._evictor,
            ledger=WriteLedger(self._checkpoint_dir / "write_ledger.jsonl"),
            selection_leases=self._selection_leases,
            task_key=self._task_key,
            min_effective_events=self._min_effective_events,
            max_task_cards=self._max_task_cards,
            preserve_survivor_payload=self._preserve_survivor_payload,
            evicted_evidence_sink=self._evicted_evidence,
        )
        # Optional novelty-admission judge: gates freshly-authored idea cards on
        # novelty against the mutator's prior. Off unless the arm turns it on
        # (the extra LLM hop per authored card is not free).
        admission_judge = (
            create_novelty_admission_agent(
                self._llm, self._task_description, prompts_dir=self._prompts_dir
            )
            if self._novelty_admission_gate
            else None
        )
        # the store IS the neighbor source: its nearest-card contract method
        # feeds the reconcile agent's context, exemplar twin dedup, and the
        # batch consolidation pass.
        librarian = Librarian(
            agent=create_reconcile_agent(
                self._llm, self._task_description, prompts_dir=self._prompts_dir
            ),
            program_author=create_program_author_agent(
                self._llm, self._task_description, prompts_dir=self._prompts_dir
            ),
            gate=gate,
            store=store,
            neighbors=store,
            top_k=policy.online_top_k,
            max_cards=policy.max_cards_per_diff,
            task_key=self._task_key,
            task_description=self._task_description,
            task_description_summary=summary,
            admission_judge=admission_judge,
        )
        self._store = store
        self._gate = gate
        self._neighbors = store
        self._librarian = librarian
        self._consolidation_agent = create_consolidate_agent(
            self._llm, self._task_description, prompts_dir=self._prompts_dir
        )


class MemoryWriter(IncrementalPostRunHook):
    """PostRunHook that authors memory cards from a completed evolutionary run
    via the librarian write path: each eligible mutation diff is reconciled into
    clean idea cards, each top-fitness exemplar gets a program card, and one
    configured eviction sweep runs per increment.

    Instantiated by Hydra; the config declares the evictor and llm once and
    shares them by reference.

    Args:
        llm: Memory LLM router for the librarian agents.
        evictor: Eviction policy consulted by the admission gate — the default
            config wires a composite over the read side's reputation;
            ``memory/evictor=none`` uses ``NullEvictor``.
        store: The one ``MemoryStore`` the run shares (``${ref:memory.store}``);
            the reader reads the same instance, so a write is visible to the
            next read with no cross-view sync.
        checkpoint_dir: Pins the write ledger under the shared checkpoint dir
            alongside the bank.
        metrics_context: Validity/sentinel semantics for record eligibility and
            gain attribution; also the single source of the fitness direction.
        task_description: Human-readable description of the current task.
        fitness_key: Metric key to use as fitness; empty (the default) resolves
            to the task's primary metric key from ``metrics_context``.
        best_programs_percent: Share of top-fitness programs authored into
            program cards.
        ingest_call_timeout_s: Per-call wall-clock bound on each librarian LLM
            call (reconcile/author); a stalled call past this is skipped and the
            record retried on a later increment.
        consolidation_every_n: After this many cards are written across sweeps,
            schedule one background near-duplicate consolidation pass over the
            whole bank. 0 disables it.
        dedup_policy: Dedup knobs of the librarian write path (reconcile-agent
            context width, consolidation candidate width); defaults to
            ``DedupPolicy()``.
        program_exemplars: Program exemplar caps/dedup policy; defaults to
            ``ProgramExemplarPolicy()``.
        prompts_dir: Optional prompts directory for the librarian agents (e.g.
            ``config.prompts.dir``); None uses the package default prompts.
        novelty_admission_gate: When true, a novelty-admission LLM judge gates
            each freshly-authored idea card, rejecting levers a strong model
            would already reach for unprompted (the bank's binding constraint is
            an excess of prior-known cards). Off by default; the extra hop per
            authored card is not free, and it fails open on any judge error.
    """

    def __init__(
        self,
        *,
        llm: MultiModelRouter,
        evictor: Evictor,
        store: MemoryStore,
        checkpoint_dir: str | Path,
        metrics_context: MetricsContext,
        stats_updater: CardEvidenceUpdater | None = None,
        baseline_estimator: NoCardBaselineEstimator | None = None,
        effect_estimator: EffectEstimator | None = None,
        no_card_recorder: NoCardEvidenceRecorder | None = None,
        task_key: str = "",
        task_description: str = "",
        fitness_key: str = "",
        best_programs_percent: float = 5.0,
        ingest_call_timeout_s: float = 300.0,
        consolidation_every_n: int = 0,
        dedup_policy: DedupPolicy | None = None,
        program_exemplars: ProgramExemplarPolicy | None = None,
        prompts_dir: str | Path | None = None,
        novelty_admission_gate: bool = False,
        selection_leases: InFlightSelectionRegistry | None = None,
        min_effective_events: float = 0.0,
        max_task_cards: int | None = None,
        evicted_evidence: EvictedEvidenceSink | None = None,
        ope_reporter: MemoryOpeReporter | None = None,
        require_archive_or_positive_gain: bool = False,
    ) -> None:
        # Default to the task's primary metric, not a literal "fitness": on a
        # task whose primary key differs, a hardcoded key would resolve to no
        # metric and silently zero every gain event (reputation never warms).
        fitness_key = fitness_key or metrics_context.get_primary_key()
        self._best_programs_percent = best_programs_percent
        self._ingest_call_timeout_s = ingest_call_timeout_s
        self._fitness_key = fitness_key
        self._metrics_context = metrics_context
        self._higher_is_better = metrics_context.is_higher_better(fitness_key)
        self._task_key = task_key
        self._task_description = task_description
        policy = dedup_policy if dedup_policy is not None else DedupPolicy()
        self._program_exemplars = (
            program_exemplars
            if program_exemplars is not None
            else ProgramExemplarPolicy()
        )

        # the live hook and the post-run hook share one writer; overlapping
        # sweeps would interleave store writes and dedup bookkeeping
        self._run_lock = asyncio.Lock()
        self._stack = LibrarianWriteStack(
            llm=llm,
            evictor=evictor,
            store=store,
            checkpoint_dir=checkpoint_dir,
            task_key=task_key,
            task_description=task_description,
            dedup_policy=policy,
            prompts_dir=prompts_dir,
            novelty_admission_gate=novelty_admission_gate,
            selection_leases=selection_leases,
            min_effective_events=min_effective_events,
            max_task_cards=max_task_cards,
            evicted_evidence=evicted_evidence,
        )
        self._extractor = ProgramRecordExtractor(
            task_description=task_description,
            task_key=task_key,
            fitness_key=fitness_key,
            metrics_context=metrics_context,
            require_archive_or_positive_gain=require_archive_or_positive_gain,
        )
        if stats_updater is not None and any(
            value is not None
            for value in (baseline_estimator, effect_estimator, no_card_recorder)
        ):
            raise ValueError(
                "a custom card evidence updater cannot be combined with legacy "
                "baseline/effect/no-card estimators"
            )
        self._stats: CardEvidenceUpdater = (
            stats_updater
            if stats_updater is not None
            else CardStatsUpdater(
                fitness_key=fitness_key,
                higher_is_better=self._higher_is_better,
                metrics_context=metrics_context,
                baseline_estimator=baseline_estimator,
                effect_estimator=effect_estimator,
                no_card_recorder=no_card_recorder,
                task_key=task_key,
                selection_leases=selection_leases,
            )
        )
        self._consolidation = ConsolidationScheduler(
            stack=self._stack,
            run_lock=self._run_lock,
            every_n=consolidation_every_n,
            k=policy.consolidation_k,
        )
        self._ope_reporter = ope_reporter

    async def on_run_complete(self, storage: ProgramStorage) -> None:
        """Called by EvolutionEngine after the generation loop finishes."""
        programs = await storage.get_all(exclude=EXCLUDE_STAGE_RESULTS)
        if not programs:
            logger.warning("[Memory][Writer] no programs in storage, skipping.")
            return
        await self.run_increment(programs)
        # The final sweep may schedule a background consolidation pass; await it
        # here (the run lock is now free) so it completes before the event loop
        # is torn down rather than being cancelled at shutdown.
        await self._consolidation.drain(timeout=self._ingest_call_timeout_s)

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
        # After the lock releases (both live sweeps and the final on_run_complete
        # route through here), refresh the probe-ITT summary off the loop so tau
        # lands beside the ledger in-progress. Read-only; never raises.
        if self._ope_reporter is not None:
            await asyncio.to_thread(self._ope_reporter.refresh)

    async def _run_increment_locked(
        self,
        programs: list[Program],
        *,
        posterior_programs: list[Program] | None,
    ) -> None:
        await self._stack.ensure()
        summary = self._stack.task_description_summary
        records = self._extractor.extract(
            programs,
            task_description_summary=summary,
            posterior_programs=posterior_programs,
        )

        librarian = self._stack.require_librarian()
        results: list[WriteResult] = []
        for i, rec in enumerate(records, 1):
            # The sink mirrors the ingest's return value result-by-result:
            # on timeout the cancelled coroutine's return is lost, but cards
            # it already banked must still count below (inline consolidation
            # subset + cadence), else accounting under-counts exactly on the
            # slowest calls. Read only on timeout — success uses the return.
            partial: list[WriteResult] = []
            try:
                written = await asyncio.wait_for(
                    librarian.ingest_idea(
                        base_parent_code=rec.parent_code,
                        child_id=rec.id,
                        child_code=rec.code,
                        note=record_note(rec),
                        founding_gain=rec.founding_gain,
                        sink=partial,
                    ),
                    timeout=self._ingest_call_timeout_s,
                )
                results.extend(written)
            except TimeoutError:
                # A stalled memory-LLM call must not starve the rest of the
                # sweep (sibling ingests, exemplars, harm eviction). Drop this
                # record so it is retried on a later increment, and continue.
                logger.warning(
                    "[Memory][Writer] ingest of {} ({}/{}) timed out after "
                    "{}s with {} card(s) already banked; skipping record for "
                    "retry next sweep",
                    rec.id,
                    i,
                    len(records),
                    self._ingest_call_timeout_s,
                    sum(1 for r in partial if r.landed),
                )
                results.extend(partial)
                self._extractor.forget({rec.id})
                continue
            except BaseException:
                # CancelledError included: the record was marked seen before
                # ingest; without rollback the window's idea is lost.
                self._extractor.forget({rec.id})
                raise

        pool = programs if posterior_programs is None else posterior_programs
        await self._author_exemplars(pool, summary)

        # re-stamping (per-card store writes) and eviction are blocking
        # I/O; keep them off the event loop so in-flight mutations don't stall
        await _shielded_to_thread(
            self._stats.update,
            pool,
            store=self._stack.require_store(),
            gate=self._stack.require_gate(),
            cancel_log="[Memory][Writer] stats restamp was cancelled",
        )
        # Intra-batch dedup: fold this increment's own same-lever idea cards now,
        # inline under the run lock, so a duplicate can't be injected into the
        # mutator before the periodic whole-bank pass' cadence trips. Bounded so
        # its consolidate-agent calls can't stall the sweep. Only freshly ADDED
        # cards can be the unseen half of a same-batch duplicate pair — a
        # MERGED/DUPLICATE target was already arbitrated this increment, and a
        # rejected card never landed; partners still come from the whole bank.
        landed = [r.card_id for r in results if r.landed]
        added = {r.card_id for r in results if r.outcome is WriteOutcome.ADDED}
        await self._consolidation.consolidate_written(
            added, timeout=self._ingest_call_timeout_s
        )
        self._consolidation.note_writes(len(landed))

    async def _author_exemplars(self, pool: list[Program], summary: str) -> None:
        """Author a clean program card for each top-fitness exemplar.

        ``author_program`` is cached on ``program-<id>`` so a re-selected
        exemplar never re-pays the LLM; the gate re-admits the card (its
        gain events are restamped immediately after from the full pool).
        A harm-evicted exemplar is skipped outright — the eviction deleted the
        cache, so re-authoring would re-pay the LLM only for the gate's
        tombstone to reject the card.
        """
        policy = self._program_exemplars
        if not policy.enabled:
            return
        selected = self._metrics_context.top_valid_programs(
            pool,
            key=self._fitness_key,
            percent=self._best_programs_percent,
        )
        if policy.top_k_per_refresh is not None:
            selected = selected[: policy.top_k_per_refresh]
        librarian = self._stack.require_librarian()
        gate = self._stack.require_gate()
        for prog, fitness in selected:
            if gate.is_tombstoned(f"program-{prog.id}"):
                continue
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
                    "[Memory][Writer] authoring exemplar {} timed out "
                    "after {}s; skipping",
                    prog.id,
                    self._ingest_call_timeout_s,
                )
                continue
            except Exception as exc:
                # A non-timeout authoring failure (API/schema error) must
                # degrade per-exemplar like the idea path, not abort the
                # increment before the stats restamp and eviction sweep.
                # Cancellation (BaseException) still propagates.
                logger.warning(
                    "[Memory][Writer] authoring exemplar {} failed ({}); skipping",
                    prog.id,
                    exc,
                )
                continue
            try:
                librarian.admit_program(
                    Card(
                        kind=CardKind.PROGRAM,
                        id=f"program-{prog.id}",
                        task_key=self._task_key,
                        program_id=prog.id,
                        task_description=self._task_description,
                        task_description_summary=summary,
                        description=authored.description,
                        explanation_summary=authored.explanation_summary,
                        fitness=fitness,
                        code=prog.code if policy.store_code else "",
                        code_sha256=code_sha256(prog.code),
                        keywords=tuple(authored.keywords),
                    ),
                    higher_is_better=self._higher_is_better,
                    min_fitness_gap=policy.min_fitness_gap,
                )
            except Exception as exc:
                # Card construction (validation) or the gate's store write can
                # fail for one exemplar (e.g. a persist hiccup); degrade it like
                # the authoring path so the remaining exemplars, the stats
                # restamp, and the eviction sweep still run.
                logger.warning(
                    "[Memory][Writer] admitting exemplar {} failed ({}); skipping",
                    prog.id,
                    exc,
                )
        self._prune_program_exemplars()

    def _prune_program_exemplars(self) -> None:
        policy = self._program_exemplars
        if not policy.enabled:
            return
        store = self._stack.require_store()
        exemplars = [
            c
            for c in store.snapshot()
            if c.kind is CardKind.PROGRAM and c.task_key == self._task_key
        ]
        excess = len(exemplars) - policy.max_cards
        if excess <= 0:
            return

        def rank(card: Card) -> float:
            if card.fitness is None:
                return float("-inf") if self._higher_is_better else float("inf")
            return card.fitness

        ordered = sorted(
            exemplars,
            key=lambda c: (rank(c), c.id),
            reverse=self._higher_is_better,
        )
        gate = self._stack.require_gate()
        for card in ordered[policy.max_cards :]:
            gate.retire_exemplar(
                card,
                reason=f"program exemplar pruned by max_cards={policy.max_cards}",
            )
