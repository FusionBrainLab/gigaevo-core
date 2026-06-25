"""
IdeaTracker: PostRunHook that extracts, classifies, enriches, and stores
improvement ideas from a completed evolutionary run.

_SessionLog accumulates log entries in memory and writes all files to a
timestamped directory in a single flush() call at session end.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
import json
import math
import os
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from gigaevo.evolution.engine.hooks import IncrementalPostRunHook
from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
    MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY,
    MUTATION_OUTPUT_METADATA_KEY,
)
from gigaevo.llm.models import MultiModelRouter
from gigaevo.memory.backend_factory import MemoryBackendFactory
from gigaevo.memory.context import ContextualGain
from gigaevo.memory.core.events import emit_memory_event
from gigaevo.memory.core.protocols import (
    Deduplicator,
    Evictor,
    ReputationModel,
)
from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.ideas_tracker.analyzers import (
    Analyzer,
    ClassifyingAnalyzer,
    ClusteringAnalyzer,
)
from gigaevo.memory.ideas_tracker.idea_bank import (
    MACHINE_KEYWORD_PREFIXES,
    IdeaBank,
)
from gigaevo.memory.ideas_tracker.models import (
    Idea,
    IdeaExplanation,
    ProgramRecord,
    program_to_record,
)
from gigaevo.memory.ideas_tracker.schemas import KeywordsResponse, SummaryResponse
from gigaevo.memory.shared_memory.injection_posterior import (
    InjectionOutcome,
    compute_contextual_gains,
)
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


def _summarise_task_description(analyzer: Analyzer, task_description: str) -> str:
    """Ask the LLM for a compact summary of the task description."""
    text = str(task_description or "").strip()
    if not text:
        return "Task summary unavailable"
    try:
        parsed = analyzer.call_structured(
            "task_description_summary", SummaryResponse, text
        )
        summary = parsed.summary.strip()
        return summary or text[:240].strip()
    except Exception as exc:
        logger.warning(
            "[Memory][IdeaTracker] Task description summarization failed, using truncated text: {}",
            exc,
        )
        return text[:240].strip()


def _select_ideas_needing_enrichment(
    ideas: list[Idea], last_entry_count: dict[str, int]
) -> list[Idea]:
    return [
        idea
        for idea in ideas
        if last_entry_count.get(idea.id, -1) != len(idea.explanation.entries)
    ]


@dataclass(frozen=True)
class _EnrichmentOutcome:
    """Enriched idea plus whether every LLM step succeeded; failed ideas stay
    stale so the next sweep retries them."""

    idea: Idea
    ok: bool


async def _enrich_ideas_with_keywords_and_summaries(
    ideas: list[Idea], analyzer: Analyzer, task_summary: str
) -> list[_EnrichmentOutcome]:
    """Enrich all ideas concurrently with keywords and explanation summaries."""

    async def _enrich_one(idea: Idea) -> _EnrichmentOutcome:
        ok = True
        # On LLM failure the old keywords stay; on success the machine tokens
        # (verification gate, canonical-dedup) survive the topical refresh.
        keywords = list(idea.keywords)
        try:
            kw_parsed = await analyzer.call_structured_async(
                "keywords", KeywordsResponse, idea.description
            )
            machine = [
                kw for kw in idea.keywords if kw.startswith(MACHINE_KEYWORD_PREFIXES)
            ]
            keywords = machine + [kw for kw in kw_parsed.keywords if kw not in machine]
        except Exception as exc:
            ok = False
            logger.warning(
                "[Memory][IdeaTracker] Keyword extraction failed for idea {!r}: {}",
                idea.id,
                exc,
            )

        # On LLM failure the previously synthesized summary stays.
        summary = idea.explanation.summary
        entries = idea.explanation.entries
        if len(entries) == 1:
            summary = entries[0]
        elif len(entries) > 1:
            explanations_text = "\n".join(f"- {e}" for e in entries)
            try:
                sum_parsed = await analyzer.call_structured_async(
                    "usage_summary", SummaryResponse, explanations_text
                )
                summary = sum_parsed.summary
            except Exception as exc:
                ok = False
                logger.warning(
                    "[Memory][IdeaTracker] Summary generation failed for idea {!r}: {}",
                    idea.id,
                    exc,
                )

        enriched = idea.model_copy(
            update={
                "keywords": keywords,
                "explanation": IdeaExplanation(entries=entries, summary=summary),
                "task_description_summary": task_summary,
            }
        )
        return _EnrichmentOutcome(idea=enriched, ok=ok)

    return list(await asyncio.gather(*[_enrich_one(idea) for idea in ideas]))


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


# A cancelled sweep abandons its to_thread writer mid-ingest; the next sweep
# (or on_run_complete) must not interleave a second backend build with it.
_WRITE_PIPELINE_LOCK = threading.Lock()


def _run_write_pipeline(
    enabled: bool,
    banks_path: Path | None,
    programs_path: Path | None,
    backend: MemoryBackendFactory | None,
    checkpoint_dir: str | Path | None = None,
    best_programs_percent: float = 5.0,
    higher_is_better: bool = True,
    gain_events: dict[str, list[ContextualGain]] | None = None,
    evictor: Evictor | None = None,
    deduplicator: Deduplicator | None = None,
) -> None:
    """Optionally trigger the downstream memory write pipeline.

    ``backend`` (a Hydra-composed ``memory/backend`` factory) builds the card
    bank; it must be non-None whenever ``enabled`` is True (IdeaTracker
    enforces this at construction). ``checkpoint_dir`` pins per-run artefacts
    under the Hydra output dir.
    """
    if not enabled:
        return
    with _WRITE_PIPELINE_LOCK:
        _run_write_pipeline_locked(
            banks_path,
            programs_path,
            backend=backend,
            checkpoint_dir=checkpoint_dir,
            best_programs_percent=best_programs_percent,
            higher_is_better=higher_is_better,
            gain_events=gain_events,
            evictor=evictor,
            deduplicator=deduplicator,
        )


def _run_write_pipeline_locked(
    banks_path: Path | None,
    programs_path: Path | None,
    backend: MemoryBackendFactory | None,
    checkpoint_dir: str | Path | None,
    best_programs_percent: float,
    higher_is_better: bool,
    gain_events: dict[str, list[ContextualGain]] | None,
    evictor: Evictor | None,
    deduplicator: Deduplicator | None,
) -> None:
    if backend is None:
        raise ValueError(
            "memory write pipeline enabled but no backend factory provided; "
            "compose one via the Hydra `memory/backend` group "
            "(ideas_tracker configs override the shared `memory.backend` singleton to `local`)."
        )
    if banks_path is None:
        logger.warning(
            "[Memory][IdeaTracker] write pipeline skipped: log paths unavailable."
        )
        return
    if not banks_path.exists():
        logger.warning(
            "[Memory][IdeaTracker] write pipeline skipped: missing {}.", banks_path
        )
        return

    effective_programs_path = (
        programs_path if (programs_path and programs_path.exists()) else None
    )

    from gigaevo.memory.write_pipeline import main as _write_main

    snapshot = _write_main(
        banks_path=banks_path,
        programs_path=effective_programs_path,
        backend=backend,
        checkpoint_dir=checkpoint_dir,
        best_programs_percent=best_programs_percent,
        higher_is_better=higher_is_better,
        gain_events=gain_events,
        evictor=evictor,
        deduplicator=deduplicator,
    )
    if snapshot is not None:
        logger.info(
            "[Memory][IdeaTracker] write: processed={}, added={}, updated={}, rejected={}",
            snapshot.stats.processed,
            snapshot.stats.added,
            snapshot.stats.updated,
            snapshot.stats.rejected,
        )


# ---------------------------------------------------------------------------
# _SessionLog
# ---------------------------------------------------------------------------


class _SessionLog:
    """
    Accumulates log entries in memory during a tracker run and writes all
    files to a timestamped session directory in a single flush() call.

    Replaces the per-event read-modify-write pattern of IdeasTrackerLogger.
    Files written: log.txt, banks.json, programs.json.
    """

    def __init__(self, logs_dir: Path) -> None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir: Path = logs_dir / ts
        self._entries: list[str] = []

    # ------ file paths ------
    @property
    def banks_file(self) -> Path:
        return self.session_dir / "banks.json"

    @property
    def programs_file(self) -> Path:
        return self.session_dir / "programs.json"

    # ------ recording ------

    def record(self, action: str, **params: Any) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"[{ts}]: {action}"]
        for k, v in params.items():
            lines.append(f"  {k}: {v}")
        self._entries.append("\n".join(lines))

    # ------ flush ------

    def flush(
        self,
        bank: IdeaBank,
        *,
        records: list[ProgramRecord],
    ) -> None:
        """Write all accumulated data to the timestamped session directory."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        (self.session_dir / "log.txt").write_text(
            "\n\n".join(self._entries), encoding="utf-8"
        )

        ideas = bank.all_ideas()
        self.write_bank_snapshot(ideas, ts)

        programs_data = [
            {
                "timestamp": ts,
                "programs": [r.model_dump() for r in records],
            }
        ]
        self.programs_file.write_text(
            json.dumps(programs_data, indent=2), encoding="utf-8"
        )

    def write_bank_snapshot(self, ideas: list[Idea], timestamp: str) -> None:
        """Write banks.json as a single active-bank snapshot of typed ideas."""
        banks_data = [
            {
                "active_bank": [idea.model_dump() for idea in ideas],
                "timestamp": timestamp,
            }
        ]
        self.banks_file.write_text(json.dumps(banks_data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Hydra / YAML factory (config/ideas_tracker/*.yaml)
# ---------------------------------------------------------------------------

_CLUSTERING_ANALYZER_KEYS = frozenset(
    {
        "embeddings_model",
        "batch_size",
        "min_samples_for_dbscan",
        "dbscan_eps",
        "dbscan_min_samples",
        "max_attempts",
        "max_rounds",
        "refine_subgroup_size",
    }
)


def _build_analyzer_from_hydra_fields(
    *,
    analyzer_type: str,
    llm: MultiModelRouter,
    analyzer_fast_settings: dict[str, Any] | None,
    description_rewriting: bool,
    analyzer_max_concurrent_classifications: int = 8,
) -> ClassifyingAnalyzer | ClusteringAnalyzer:
    """Construct ClassifyingAnalyzer or ClusteringAnalyzer from flat Hydra kwargs."""
    kind = (analyzer_type or "default").strip().lower()

    if kind == "fast":
        fast = dict(analyzer_fast_settings or {})
        extra = {k: v for k, v in fast.items() if k in _CLUSTERING_ANALYZER_KEYS}
        unknown = sorted(set(fast) - _CLUSTERING_ANALYZER_KEYS)
        if unknown:
            logger.warning(
                "[Memory][IdeaTracker] ignoring unrecognized analyzer_fast_settings "
                "keys: {}",
                unknown,
            )
        return ClusteringAnalyzer(llm=llm, **extra)

    return ClassifyingAnalyzer(
        llm=llm,
        description_rewriting=description_rewriting,
        max_concurrent_classifications=analyzer_max_concurrent_classifications,
    )


# ---------------------------------------------------------------------------
# IdeaTracker
# ---------------------------------------------------------------------------


class IdeaTracker(IncrementalPostRunHook):
    """
    PostRunHook that extracts, classifies, enriches, and stores improvement
    ideas from a completed evolutionary run.

    Instantiated by Hydra. Accepts a ClassifyingAnalyzer or ClusteringAnalyzer —
    both implement the Analyzer protocol, so the pipeline is identical for both.

    Args:
        analyzer: Explicit analyser instance. When omitted, one is built from
            ``llm`` and the Hydra-style fields below.
        llm: Memory LLM router for the analyser (Hydra ``memory/llm`` group,
            threaded in by ``MemorySystem``). Required when ``analyzer`` is None.
        analyzer_type: ``"default"`` → ClassifyingAnalyzer; ``"fast"`` → ClusteringAnalyzer.
        analyzer_fast_settings: Extra kwargs for ClusteringAnalyzer when ``fast``.
        memory_write_best_programs_percent: Share of top-fitness programs
            converted into program cards by the write pipeline.
        backend: Memory backend factory (Hydra ``memory/backend`` group) used
            by the write pipeline to build the card bank. Required whenever
            ``memory_write_enabled`` is True — ``MemorySystem`` threads in the
            shared backend it built for the read provider, so the writer and
            reader share one card bank.
        checkpoint_dir: Pins per-run memory cards under the Hydra output dir.
        task_description: Human-readable description of the current task. If empty,
            loaded from the matching problems/ directory using redis_prefix.
        redis_prefix: Redis key prefix (e.g. "chains/hotpotqa/static") used to
            locate the task_description.txt file when task_description is empty.
        chunk_size: Number of ideas per LLM classification batch.
        memory_write_enabled: If True, trigger the downstream memory write pipeline.
        fitness_key: Metric key to use as fitness (default "fitness").
        metrics_context: When wired, programs whose fitness equals the
            metric's sentinel value are excluded from records and posteriors.
        logs_dir: Directory for timestamped session logs. Defaults to
            gigaevo/memory/ideas_tracker/logs/.
    """

    def __init__(
        self,
        *,
        analyzer: Analyzer | None = None,
        llm: MultiModelRouter | None = None,
        analyzer_type: str = "default",
        analyzer_fast_settings: dict[str, Any] | None = None,
        analyzer_max_concurrent_classifications: int = 8,
        description_rewriting: bool = True,
        memory_write_enabled: bool = True,
        memory_write_best_programs_percent: float = 5.0,
        backend: MemoryBackendFactory | None = None,
        checkpoint_dir: str | Path | None = None,
        task_description: str = "",
        redis_prefix: str = "",
        chunk_size: int = 5,
        fitness_key: str = "fitness",
        fitness_higher_is_better: bool = True,
        metrics_context: MetricsContext | None = None,
        logs_dir: str | Path | None = None,
        evictor: Evictor | None = None,
        deduplicator: Deduplicator | None = None,
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

        _ensure_writable_hf_cache()
        if analyzer is None:
            if llm is None:
                raise ValueError(
                    "IdeaTracker: building an analyzer requires an LLM router. "
                    "Enable the writer via `memory=writer` or `memory=full` "
                    "(MemorySystem threads in the `memory/llm` router; swap it "
                    "with `memory/llm=qwen_instruct`), or pass an explicit analyzer."
                )
            analyzer = cast(
                Analyzer,
                _build_analyzer_from_hydra_fields(
                    analyzer_type=analyzer_type,
                    llm=llm,
                    analyzer_fast_settings=analyzer_fast_settings,
                    description_rewriting=description_rewriting,
                    analyzer_max_concurrent_classifications=analyzer_max_concurrent_classifications,
                ),
            )

        self._analyzer: Analyzer = analyzer
        self._bank = IdeaBank(chunk_size=chunk_size)
        self._fitness_key = fitness_key
        self._fitness_higher_is_better = fitness_higher_is_better
        self._metrics_context = metrics_context
        self._memory_write_enabled = memory_write_enabled
        self._best_programs_percent = memory_write_best_programs_percent
        self._backend = backend
        self._checkpoint_dir = checkpoint_dir
        self._evictor = evictor
        self._deduplicator = deduplicator
        self._reputation = (
            reputation if reputation is not None else BetaBinomialReputation()
        )
        self._all_records: list[ProgramRecord] = []
        self._seen_ids: set[str] = set()
        self._classification_failures: dict[str, int] = {}
        self._max_classification_failures = 3
        self._last_entry_count: dict[str, int] = {}
        # the live hook and the post-run hook share one tracker; overlapping
        # sweeps would interleave bank mutations and _seen_ids bookkeeping
        self._run_lock = asyncio.Lock()

        if task_description:
            self._task_description = task_description
        else:
            self._task_description = _load_task_description(
                redis_prefix, Path(__file__).resolve()
            )

        resolved_logs = (
            Path(logs_dir)
            if logs_dir is not None
            else Path(__file__).resolve().parent / "logs"
        )
        resolved_logs.mkdir(parents=True, exist_ok=True)
        self._log = _SessionLog(resolved_logs)

    @cached_property
    def _task_summary(self) -> str:
        """Computed once on first access; cached for the lifetime of this instance."""
        return _summarise_task_description(self._analyzer, self._task_description)

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
        """Full pipeline: filter → analyse → enrich → log → write.

        ``programs`` feeds the expensive LLM analyzer and may be a bounded
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
        records = self._eligible_records(
            programs, posterior_programs=posterior_programs
        )

        try:
            result = await self._analyzer.analyze_async(records, self._bank)
        except BaseException:
            # CancelledError included: records were marked seen before analysis;
            # without rollback the window's ideas are permanently lost.
            self._forget_records({r.id for r in records})
            raise
        self._bank.apply(result)

        failed_ids = set(result.failed_program_ids)
        if failed_ids:
            # Without this the ids stay in _seen_ids and the LLM outage
            # permanently loses those programs' ideas. Poison programs whose
            # ideas never classify are retired after the failure cap so they
            # stop re-burning analyzer calls while inside the window.
            retry_ids: set[str] = set()
            for pid in failed_ids:
                failures = self._classification_failures.get(pid, 0) + 1
                self._classification_failures[pid] = failures
                if failures < self._max_classification_failures:
                    retry_ids.add(pid)
                else:
                    logger.warning(
                        "[Memory][IdeaTracker] program {} failed classification "
                        "{} times, retiring it from future sweeps.",
                        pid,
                        failures,
                    )
            self._forget_records(retry_ids)

        if records:
            stale_ideas = _select_ideas_needing_enrichment(
                self._bank.all_ideas(), self._last_entry_count
            )
            failed_enrichment: set[str] = set()
            if stale_ideas:
                outcomes = await _enrich_ideas_with_keywords_and_summaries(
                    stale_ideas, self._analyzer, self._task_summary
                )
                for outcome in outcomes:
                    if not outcome.ok:
                        failed_enrichment.add(outcome.idea.id)
                        continue
                    self._bank.enrich(
                        outcome.idea.id,
                        keywords=outcome.idea.keywords,
                        summary=outcome.idea.explanation.summary,
                        task_summary=self._task_summary,
                    )
            # Failed ideas are left unstamped so the next sweep retries them.
            self._last_entry_count = {
                idea.id: len(idea.explanation.entries)
                for idea in self._bank.all_ideas()
                if idea.id not in failed_enrichment
            }

        self._log.record("pipeline_complete", total_ideas=len(self._bank.all_ideas()))

        card_gain_events = _card_gain_events_from_programs(
            programs if posterior_programs is None else posterior_programs,
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

        # flush (bank serialization) and the write pipeline (backend build +
        # card ingest) are blocking I/O; keep them off the event loop so
        # in-flight mutations don't stall for the sweep
        def _flush_and_write() -> None:
            self._log.flush(self._bank, records=self._all_records)
            _run_write_pipeline(
                self._memory_write_enabled,
                self._log.banks_file,
                self._log.programs_file,
                backend=self._backend,
                checkpoint_dir=self._checkpoint_dir,
                best_programs_percent=self._best_programs_percent,
                higher_is_better=self._fitness_higher_is_better,
                gain_events=card_gain_events,
                evictor=self._evictor,
                deduplicator=self._deduplicator,
            )

        await asyncio.to_thread(_flush_and_write)

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
                self._task_summary,
                self._fitness_key,
                parent_codes=parent_codes,
            )
            for p in eligible
        ]
        self._all_records.extend(records)
        self._seen_ids.update(p.id for p in eligible)
        return records
