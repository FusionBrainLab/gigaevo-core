"""
IdeaTracker: PostRunHook that authors memory cards from a completed
evolutionary run via the librarian write path (one card per mutation diff,
plus a clean card for each top-fitness exemplar).
"""

from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from gigaevo.evolution.engine.hooks import IncrementalPostRunHook
from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
    MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY,
    MUTATION_OUTPUT_METADATA_KEY,
)

# LLM-first write path (librarian): authors clean cards from a mutation diff and
# routes every verdict through the single admission gate.
from gigaevo.llm.agents.factories import (
    create_consolidate_agent,
    create_program_author_agent,
    create_reconcile_agent,
)
from gigaevo.llm.agents.task_summary import TaskSummaryAgent
from gigaevo.llm.models import MultiModelRouter
from gigaevo.memory.backend_factory import MemoryBackendFactory
from gigaevo.memory.context import ContextualGain
from gigaevo.memory.core.admission_gate import CardAdmissionGate
from gigaevo.memory.core.events import emit_memory_event
from gigaevo.memory.core.evictor import HarmEvictor
from gigaevo.memory.core.protocols import (
    Evictor,
    ReputationModel,
)
from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.core.write_ledger import WriteLedger
from gigaevo.memory.efficacy.stamping import CardStatsStamper
from gigaevo.memory.ideas_tracker.consolidation import consolidate
from gigaevo.memory.ideas_tracker.librarian import Librarian
from gigaevo.memory.ideas_tracker.librarian_retriever import ChromaNeighborSource
from gigaevo.memory.ideas_tracker.models import (
    ProgramRecord,
    program_to_record,
)
from gigaevo.memory.shared_memory.injection_posterior import (
    InjectionOutcome,
    compute_contextual_gains,
)
from gigaevo.memory.shared_memory.models import ProgramCard
from gigaevo.programs.metrics.context import VALIDITY_KEY, MetricsContext
from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS, Program

if TYPE_CHECKING:
    from gigaevo.database.program_storage import ProgramStorage

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _hf_cache_dir_usable(path: Path) -> bool:
    try:
        path = path.expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".gigaevo_hf_write_probe"
        probe.write_text("ok", encoding="ascii")
        probe.unlink()
        return True
    except OSError:
        return False


def _ensure_writable_hf_cache() -> None:
    """
    Hugging Face / sentence-transformers follow HF_HOME and related env vars.
    Shared clusters often set these to NFS roots that are not writable for this
    user, which breaks embedding downloads. Clear bad entries and use ~/.cache.
    """
    fallback = Path.home() / ".cache" / "huggingface"
    keys = (
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "SENTENCE_TRANSFORMERS_HOME",
    )
    for key in keys:
        raw = os.environ.get(key)
        if not raw or not str(raw).strip():
            continue
        if not _hf_cache_dir_usable(Path(raw)):
            logger.warning(
                "[Memory][IdeaTracker] Clearing unwritable {}={!r}", key, raw
            )
            os.environ.pop(key, None)

    hf = os.environ.get("HF_HOME")
    if hf and _hf_cache_dir_usable(Path(hf)):
        return

    fallback.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(fallback)
    hub = fallback / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub)
    os.environ["TRANSFORMERS_CACHE"] = str(hub)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(fallback)
    logger.warning("[Memory][IdeaTracker] HF cache directory -> {}", fallback)


def _load_task_description(redis_prefix: str, package_path: Path) -> str:
    """Load task_description.txt from the matching problems/ directory."""
    prefix = (redis_prefix or "").replace("/", "_")
    if not prefix:
        return "No description available"
    problems_root = package_path.parents[3] / "problems"
    try:
        for root, dirs, _ in os.walk(problems_root):
            if "initial_programs" in dirs:
                leaf = Path(root)
                split = leaf.parts.index("problems") + 1
                name = "_".join(leaf.parts[split:])
                if name == prefix:
                    candidate = leaf / "task_description.txt"
                    if candidate.is_file():
                        return candidate.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.warning(
            "[Memory][IdeaTracker] Failed to read task description for prefix {!r}: {}",
            prefix,
            exc,
        )
    return "No description available"


def _valid_fitness(
    prog: Program,
    fitness_key: str,
    metrics_context: MetricsContext | None = None,
) -> float | None:
    """Fitness under ``fitness_key``, or ``None`` for invalid programs.

    ``is_valid`` is a contract: every evaluated program carries it, so a missing
    flag is treated as invalid — the same strict semantics record eligibility
    uses. Invalid programs carry a sentinel floor fitness; treating it as a real
    value would manufacture catastrophic harm (invalid child) or phantom
    improvement (invalid parent baseline). When ``metrics_context`` is wired, a
    fitness equal to the metric's sentinel is rejected even if the program
    claims validity. Contrast: :meth:`MetricsContext.is_valid` defaults a
    missing flag to valid — correct for metric aggregation, not for stat
    tracking.
    """
    is_valid = prog.metrics.get(VALIDITY_KEY)
    if is_valid is None or is_valid <= 0:
        return None
    fit = prog.metrics.get(fitness_key)
    if fit is None or not math.isfinite(fit):
        return None
    if metrics_context is not None and metrics_context.is_sentinel(fitness_key, fit):
        return None
    return float(fit)


def _evaluated_invalid(
    prog: Program,
    fitness_key: str,
    metrics_context: MetricsContext | None = None,
) -> bool:
    """True iff the program was evaluated and judged invalid — a real negative
    outcome the posterior must count as harm, unlike a program that simply never
    produced a fitness (missing ``is_valid``: not evaluated, no signal)."""
    is_valid = prog.metrics.get(VALIDITY_KEY)
    if is_valid is None:
        return False
    if is_valid <= 0:
        return True
    fit = prog.metrics.get(fitness_key)
    if fit is None or not math.isfinite(fit):
        return False
    return metrics_context is not None and metrics_context.is_sentinel(fitness_key, fit)


def _base_fitness(
    base_metrics: dict[str, float],
    fitness_key: str,
    metrics_context: MetricsContext | None,
) -> float | None:
    """Base parent's reward baseline from its frozen metrics, mirroring the
    validity/sentinel semantics of :func:`_valid_fitness`."""
    is_valid = base_metrics.get(VALIDITY_KEY)
    if is_valid is None or is_valid <= 0:
        return None
    fit = base_metrics.get(fitness_key)
    if fit is None or not math.isfinite(fit):
        return None
    if metrics_context is not None and metrics_context.is_sentinel(fitness_key, fit):
        return None
    return float(fit)


def _card_ids_used(prog: Program) -> list[str]:
    """Card ids the mutator declared it applied, from the stamped structured output."""
    out = prog.get_metadata(MUTATION_OUTPUT_METADATA_KEY)
    if isinstance(out, dict):
        return list(out.get("card_ids_used", []) or [])
    return []


def _base_selected_ids(prog: Program) -> list[str]:
    """Cards selected for the mutator's named base parent, frozen at birth."""
    ids = prog.get_metadata(MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY)
    return list(ids) if isinstance(ids, list) else []


def _base_metrics(prog: Program) -> dict[str, float]:
    """The base parent's metric dict, frozen at birth."""
    metrics = prog.get_metadata(MUTATION_MEMORY_BASE_METRICS_METADATA_KEY)
    return dict(metrics) if isinstance(metrics, dict) else {}


def _card_gain_events_from_programs(
    programs: list[Program],
    *,
    fitness_key: str,
    higher_is_better: bool,
    metrics_context: MetricsContext | None = None,
) -> dict[str, list[ContextualGain]]:
    """Use-attributed base-relative gain events per card id, from live programs.

    Credits a card only when it was both selected for the mutator's named base
    parent (``memory_base_selected_idea_ids``) and declared used
    (``mutation_output.card_ids_used``); the base fitness baseline is resolved
    here from the frozen base metrics under ``fitness_key``.
    """
    rows = []
    for prog in programs:
        base_metrics = _base_metrics(prog)
        rows.append(
            InjectionOutcome(
                id=prog.id,
                parents=prog.lineage.parents,
                fitness=_valid_fitness(prog, fitness_key, metrics_context),
                invalid=_evaluated_invalid(prog, fitness_key, metrics_context),
                base_selected_ids=_base_selected_ids(prog),
                base_metrics=base_metrics,
                base_fitness=_base_fitness(base_metrics, fitness_key, metrics_context),
                card_ids_used=_card_ids_used(prog),
            )
        )
    return compute_contextual_gains(rows, higher_is_better=higher_is_better)


# ---------------------------------------------------------------------------
# Librarian write-path helpers
# ---------------------------------------------------------------------------


def _record_note(record: ProgramRecord) -> str:
    """One-line mutation note from a record's normalised improvements."""
    note = "; ".join(
        imp.description.strip()
        for imp in record.improvements
        if imp.description.strip()
    )
    return note or "Unspecified change"


def _select_top_programs(
    programs: list[Program],
    *,
    best_programs_percent: float,
    fitness_key: str,
    higher_is_better: bool,
    metrics_context: MetricsContext | None,
) -> list[tuple[Program, float]]:
    """The top-fitness slice of valid programs, as (program, fitness) pairs."""
    if best_programs_percent <= 0:
        return []
    scored = [
        (prog, fit)
        for prog in programs
        for fit in (_valid_fitness(prog, fitness_key, metrics_context),)
        if fit is not None
    ]
    if not scored:
        return []
    scored.sort(key=lambda pair: (pair[1], pair[0].id), reverse=higher_is_better)
    count = max(1, math.ceil(len(scored) * best_programs_percent / 100.0))
    return scored[:count]


# ---------------------------------------------------------------------------
# IdeaTracker
# ---------------------------------------------------------------------------


class IdeaTracker(IncrementalPostRunHook):
    """
    PostRunHook that authors memory cards from a completed evolutionary run via
    the librarian write path: each eligible mutation diff is reconciled into
    clean idea cards, each top-fitness exemplar gets a program card, and one
    harm-eviction sweep runs per increment.

    Instantiated by Hydra. ``MemorySystem`` threads in the shared backend, llm
    router, and evictor so the writer and reader share one card bank.

    Args:
        llm: Memory LLM router for the librarian agents (Hydra ``memory/llm``
            group, threaded in by ``MemorySystem``). Required when
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
        consolidation_eps: Cosine-distance threshold below which two bank cards
            are folded into one during a consolidation pass.
        backend: Memory backend factory (Hydra ``memory/backend`` group) used to
            build the card bank. Required whenever ``memory_write_enabled`` is
            True — ``MemorySystem`` threads in the shared backend it built for the
            read provider, so the writer and reader share one card bank.
        checkpoint_dir: Pins per-run memory cards under the Hydra output dir.
        task_description: Human-readable description of the current task. If empty,
            loaded from the matching problems/ directory using redis_prefix.
        redis_prefix: Redis key prefix (e.g. "chains/hotpotqa/static") used to
            locate the task_description.txt file when task_description is empty.
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
        consolidation_eps: float = 0.05,
        backend: MemoryBackendFactory | None = None,
        checkpoint_dir: str | Path | None = None,
        task_description: str = "",
        redis_prefix: str = "",
        fitness_key: str = "fitness",
        fitness_higher_is_better: bool = True,
        metrics_context: MetricsContext | None = None,
        evictor: Evictor | None = None,
        reputation: ReputationModel | None = None,
        **extras: Any,
    ) -> None:
        if extras:
            logger.warning(
                "[Memory][IdeaTracker] ignoring extra instantiate kwargs: {}", extras
            )
        if memory_write_enabled and backend is None:
            raise ValueError(
                "IdeaTracker: memory_write_enabled=True requires an explicit "
                "backend factory. Enable the writer via `memory=writer` or "
                "`memory=full` (MemorySystem assembles the backend and threads "
                "it in), or pass memory_write_enabled=False."
            )
        if memory_write_enabled and llm is None:
            raise ValueError(
                "IdeaTracker: memory_write_enabled=True requires an LLM router "
                "for the librarian. Enable the writer via `memory=writer` or "
                "`memory=full` (MemorySystem threads in the `memory/llm` router; "
                "swap it with `memory/llm=qwen_instruct`)."
            )

        _ensure_writable_hf_cache()
        self._llm = llm
        self._fitness_key = fitness_key
        self._fitness_higher_is_better = fitness_higher_is_better
        self._metrics_context = metrics_context
        self._memory_write_enabled = memory_write_enabled
        self._best_programs_percent = memory_write_best_programs_percent
        self._ingest_call_timeout_s = ingest_call_timeout_s
        self._consolidation_every_n = consolidation_every_n
        self._consolidation_eps = consolidation_eps
        self._writes_since_consolidation = 0
        self._consolidation_task: asyncio.Task | None = None
        self._consolidation_agent: Any | None = None
        self._consolidation_neighbors: Any | None = None
        self._backend = backend
        self._checkpoint_dir = checkpoint_dir
        self._evictor = evictor
        self._reputation = (
            reputation if reputation is not None else BetaBinomialReputation()
        )
        self._all_records: list[ProgramRecord] = []
        self._seen_ids: set[str] = set()
        # the live hook and the post-run hook share one tracker; overlapping
        # sweeps would interleave store writes and _seen_ids bookkeeping
        self._run_lock = asyncio.Lock()
        # built lazily on the first sweep: backend.build is heavy I/O and needs
        # the resolved per-run checkpoint dir
        self._store: Any | None = None
        self._gate: CardAdmissionGate | None = None
        self._librarian: Librarian | None = None

        if task_description:
            self._task_description = task_description
        else:
            self._task_description = _load_task_description(
                redis_prefix, Path(__file__).resolve()
            )
        # genuine LLM-condensed one-liner, produced once per run on the first
        # write sweep; None until then so the call is memoised.
        self._task_description_summary: str | None = None

    async def _ensure_task_summary(self) -> None:
        """Condense the task description into a one-line summary, once per run.

        Falls back to the full task description on any LLM failure (and to the
        empty string when there is no task text), so a memory-LLM hiccup can
        never block the write path.
        """
        if self._task_description_summary is not None:
            return
        if not self._task_description:
            self._task_description_summary = ""
            return
        try:
            resp = await TaskSummaryAgent(self._llm).arun(
                task_description=self._task_description
            )
            self._task_description_summary = (
                resp.summary.strip() or self._task_description
            )
        except Exception as exc:
            logger.warning(
                "[Memory][IdeaTracker] task-summary LLM failed ({}); falling back "
                "to the full task description",
                exc,
            )
            self._task_description_summary = self._task_description

    def _ensure_write_stack(self) -> None:
        """Build the store, admission gate, and librarian once, lazily."""
        if self._librarian is not None:
            return
        store = self._backend.build(
            checkpoint_dir=self._checkpoint_dir, evictor=self._evictor
        )
        self._store = store
        self._gate = CardAdmissionGate(
            store=store,
            evictor=self._evictor
            if self._evictor is not None
            else HarmEvictor(reputation=self._reputation),
            ledger=WriteLedger(store.config.checkpoint_path / "write_ledger.jsonl"),
        )
        # one neighbor source feeds both the online pre-gate and the batch
        # consolidation pass; it reuses the store's populated A-MEM Chroma index.
        neighbors = ChromaNeighborSource(store)
        self._librarian = Librarian(
            agent=create_reconcile_agent(self._llm, self._task_description),
            program_author=create_program_author_agent(
                self._llm, self._task_description
            ),
            gate=self._gate,
            store=store,
            neighbors=neighbors,
            task_description=self._task_description,
            task_description_summary=self._task_description_summary or "",
        )
        self._consolidation_neighbors = neighbors
        self._consolidation_agent = create_consolidate_agent(
            self._llm, self._task_description
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

    # ------------------------------------------------------------------
    # CLI entry point
    # ------------------------------------------------------------------

    def run(self, programs: list[Program] | None = None) -> None:
        """CLI entry: accepts list[Program] directly."""
        if not programs:
            return
        asyncio.run(self.run_increment(programs))

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
        await self._ensure_task_summary()
        records = self._eligible_records(
            programs, posterior_programs=posterior_programs
        )
        self._ensure_write_stack()

        cards_written = 0
        for rec in records:
            try:
                written = await asyncio.wait_for(
                    self._librarian.ingest_idea(
                        base_parent_id=rec.parents[0] if rec.parents else "",
                        base_parent_code=rec.parent_code,
                        child_id=rec.id,
                        child_code=rec.code,
                        note=_record_note(rec),
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
                self._forget_records({rec.id})
                continue
            except BaseException:
                # CancelledError included: the record was marked seen before
                # ingest; without rollback the window's idea is lost.
                self._forget_records({rec.id})
                raise

        pool = programs if posterior_programs is None else posterior_programs
        await self._author_exemplars(pool)

        card_gain_events = _card_gain_events_from_programs(
            pool,
            fitness_key=self._fitness_key,
            higher_is_better=self._fitness_higher_is_better,
            metrics_context=self._metrics_context,
        )
        emit_memory_event(
            component="ideas_tracker",
            event_type="injection_posterior.compute",
            payload={
                "card_count": len(card_gain_events),
                "event_count_by_card_id": {
                    cid: len(events) for cid, events in card_gain_events.items()
                },
            },
        )
        # re-stamping (per-card store writes) and harm eviction are blocking
        # I/O; keep them off the event loop so in-flight mutations don't stall
        await asyncio.to_thread(self._restamp_and_sweep, card_gain_events)
        self._note_writes_and_maybe_consolidate(cards_written)

    def _note_writes_and_maybe_consolidate(self, written: int) -> None:
        """Accumulate cards written and schedule one consolidation pass per
        ``consolidation_every_n``. The pass runs as a background task so it never
        blocks the sweep; it is idempotent and safe to skip."""
        if self._consolidation_every_n <= 0 or written <= 0:
            return
        self._writes_since_consolidation += written
        if self._writes_since_consolidation >= self._consolidation_every_n:
            self._writes_since_consolidation = 0
            self._schedule_consolidation()

    def _schedule_consolidation(self) -> None:
        if self._store is None or self._gate is None:
            return
        if self._consolidation_task is not None and not self._consolidation_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no running loop (sync context); defer to a later increment
        self._consolidation_task = loop.create_task(self._run_consolidation())

    async def _run_consolidation(self) -> None:
        try:
            merged = await consolidate(
                store=self._store,
                gate=self._gate,
                neighbors=self._consolidation_neighbors,
                agent=self._consolidation_agent,
                eps=self._consolidation_eps,
            )
            if merged:
                logger.info(
                    "[Memory][IdeaTracker] consolidation merged {} near-dup cards",
                    merged,
                )
        except Exception as exc:
            logger.warning(
                "[Memory][IdeaTracker] consolidation pass failed ({}); skipping", exc
            )

    async def _author_exemplars(self, pool: list[Program]) -> None:
        """Author a clean ProgramCard for each top-fitness exemplar.

        ``author_program`` is cached on ``program-<id>`` so a re-selected
        exemplar never re-pays the LLM; the gate re-admits the card (its
        gain events are restamped immediately after from the full pool).
        """
        selected = _select_top_programs(
            pool,
            best_programs_percent=self._best_programs_percent,
            fitness_key=self._fitness_key,
            higher_is_better=self._fitness_higher_is_better,
            metrics_context=self._metrics_context,
        )
        for prog, fitness in selected:
            try:
                authored = await asyncio.wait_for(
                    self._librarian.author_program(
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
            self._gate.admit(
                ProgramCard(
                    id=f"program-{prog.id}",
                    program_id=prog.id,
                    task_description=self._task_description,
                    task_description_summary=self._task_description_summary or "",
                    description=authored.description,
                    fitness=fitness,
                    code=prog.code,
                    keywords=authored.keywords or [],
                )
            )

    def _restamp_and_sweep(
        self, card_gain_events: dict[str, list[ContextualGain]]
    ) -> None:
        """Attach this sweep's use-attributed gain events onto credited cards,
        then run one harm-eviction pass. Gain events are a pure function of the
        full pool, so each sweep overwrites them wholesale."""
        stamper = CardStatsStamper()
        cards = self._store.card_store.cards
        for cid in card_gain_events:
            card = cards.get(cid)
            if card is None:
                continue
            self._store.save_card_direct(
                stamper.stamp_gain_events(card, card_gain_events)
            )
        self._gate.sweep()

    def _forget_records(self, ids: set[str]) -> None:
        if not ids:
            return
        self._seen_ids -= ids
        self._all_records = [r for r in self._all_records if r.id not in ids]

    def _eligible_records(
        self,
        programs: list[Program],
        *,
        posterior_programs: list[Program] | None = None,
    ) -> list[ProgramRecord]:
        """
        Filter programs and convert to ProgramRecord.

        Skips: root programs (no parents), programs without a validated fitness
        (missing/non-positive is_valid; missing, non-finite, or sentinel
        fitness), already-seen ids.

        Parent code resolves from ``posterior_programs`` (the full pool) when
        provided: live sweeps cap ``programs`` to the newest window, and
        mutation parents are usually older archive elites outside it — without
        the full pool the verification gate, canonical dedup, and
        diff-grounding silently disable mid-run.
        """
        eligible: list[Program] = []
        for prog in programs:
            if not prog.lineage.parents:
                continue
            if _valid_fitness(prog, self._fitness_key, self._metrics_context) is None:
                continue
            if prog.id in self._seen_ids:
                continue
            eligible.append(prog)

        code_pool = programs if posterior_programs is None else posterior_programs
        parent_codes: dict[str, str] = {p.id: p.code for p in code_pool if p.code}
        records = [
            program_to_record(
                p,
                self._task_description,
                self._task_description_summary or "",
                self._fitness_key,
                parent_codes=parent_codes,
            )
            for p in eligible
        ]
        self._all_records.extend(records)
        self._seen_ids.update(p.id for p in eligible)
        return records
