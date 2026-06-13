"""
IdeaTracker: PostRunHook that extracts, classifies, enriches, and stores
improvement ideas from a completed evolutionary run.

_SessionLog accumulates log entries in memory and writes all files to a
timestamped directory in a single flush() call at session end.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from functools import cached_property
import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from gigaevo.evolution.engine.hooks import IncrementalPostRunHook
from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_INJECTED_IDS_METADATA_KEY,
    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
)
from gigaevo.llm.models import MultiModelRouter
from gigaevo.memory.backend_factory import MemoryBackendFactory
from gigaevo.memory.core.protocols import (
    Deduplicator,
    Evictor,
    MemoryAdmitter,
    ReputationModel,
)
from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.efficacy import CardStatsStamper, EfficacyScorer
from gigaevo.memory.ideas_tracker.analyzers import (
    Analyzer,
    ClassifyingAnalyzer,
    ClusteringAnalyzer,
)
from gigaevo.memory.ideas_tracker.idea_bank import IdeaBank
from gigaevo.memory.ideas_tracker.models import (
    Idea,
    IdeaExplanation,
    ProgramRecord,
    program_to_record,
)
from gigaevo.memory.ideas_tracker.schemas import KeywordsResponse, SummaryResponse
from gigaevo.memory.ideas_tracker.utils.origin_analysis import (
    analyse as _analyse_origins,
)
from gigaevo.memory.shared_memory.injection_posterior import InjectionOutcome
from gigaevo.memory.shared_memory.models import CardStatsBlock
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


async def _enrich_ideas_with_keywords_and_summaries(
    ideas: list[Idea], analyzer: Analyzer, task_summary: str
) -> list[Idea]:
    """Enrich all ideas concurrently with keywords and explanation summaries."""

    async def _enrich_one(idea: Idea) -> Idea:
        keywords: list[str] = []
        try:
            kw_parsed = await analyzer.call_structured_async(
                "keywords", KeywordsResponse, idea.description
            )
            keywords = kw_parsed.keywords
        except Exception as exc:
            logger.warning(
                "[Memory][IdeaTracker] Keyword extraction failed for idea {!r}: {}",
                idea.id,
                exc,
            )

        summary = ""
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
                logger.warning(
                    "[Memory][IdeaTracker] Summary generation failed for idea {!r}: {}",
                    idea.id,
                    exc,
                )

        return idea.model_copy(
            update={
                "keywords": keywords,
                "explanation": IdeaExplanation(entries=entries, summary=summary),
                "task_description_summary": task_summary,
            }
        )

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


def _card_posterior_from_programs(
    programs: list[Program],
    *,
    fitness_key: str,
    higher_is_better: bool,
    reputation: ReputationModel | None = None,
    metrics_context: MetricsContext | None = None,
) -> dict[str, CardStatsBlock]:
    """Injection-efficacy posterior per injected card id, from live programs.

    Extracts each program's id, parents, valid fitness, and the
    ``memory_selected_idea_ids`` stamped when that program was mutated — the cards
    its children's prompts contained — then delegates to ``reputation``'s
    ``compute_injection_posteriors``, which credits each card with its children's
    outcomes under the configured thresholds (defaults when no reputation is
    wired). The result is keyed by the injected cards' own ids (idea and
    ``program-<uuid>`` alike) so every auction candidate draws a real downside
    posterior.
    """
    rows = [
        InjectionOutcome(
            id=prog.id,
            parents=prog.lineage.parents,
            fitness=_valid_fitness(prog, fitness_key, metrics_context),
            selected_ids=prog.get_metadata(MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY)
            or [],
            injected_ids=prog.get_metadata(MUTATION_MEMORY_INJECTED_IDS_METADATA_KEY),
            invalid=_evaluated_invalid(prog, fitness_key, metrics_context),
        )
        for prog in programs
    ]
    rep = reputation if reputation is not None else BetaBinomialReputation()
    return rep.compute_injection_posteriors(rows, higher_is_better=higher_is_better)


def _run_write_pipeline(
    enabled: bool,
    banks_path: Path | None,
    best_ideas_path: Path | None,
    programs_path: Path | None,
    backend: MemoryBackendFactory | None,
    checkpoint_dir: str | Path | None = None,
    best_programs_percent: float = 5.0,
    higher_is_better: bool = True,
    card_posterior: dict[str, CardStatsBlock] | None = None,
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
    if backend is None:
        raise ValueError(
            "memory write pipeline enabled but no backend factory provided; "
            "compose one via the Hydra `memory/backend` group "
            "(ideas_tracker configs override the shared `memory.backend` singleton to `local`)."
        )
    if banks_path is None or best_ideas_path is None:
        logger.warning(
            "[Memory][IdeaTracker] write pipeline skipped: log paths unavailable."
        )
        return
    if not banks_path.exists():
        logger.warning(
            "[Memory][IdeaTracker] write pipeline skipped: missing {}.", banks_path
        )
        return

    try:
        with best_ideas_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        has_snapshot = isinstance(payload, list) and any(
            isinstance(i, dict) and "best_ideas" in i for i in payload
        )
    except Exception:
        has_snapshot = False

    if not has_snapshot:
        logger.warning(
            "[Memory][IdeaTracker] write pipeline skipped: no best_ideas snapshot."
        )
        return

    effective_programs_path = (
        programs_path if (programs_path and programs_path.exists()) else None
    )

    from gigaevo.memory.write_pipeline import main as _write_main

    snapshot = _write_main(
        banks_path=banks_path,
        best_ideas_path=best_ideas_path,
        programs_path=effective_programs_path,
        backend=backend,
        checkpoint_dir=checkpoint_dir,
        best_programs_percent=best_programs_percent,
        higher_is_better=higher_is_better,
        card_posterior=card_posterior,
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
    Files written: log.txt, banks.json, programs.json, best_ideas.json.
    """

    def __init__(
        self,
        logs_dir: Path,
        admitter: MemoryAdmitter | None = None,
        higher_is_better: bool = True,
        scorer: EfficacyScorer | None = None,
    ) -> None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir: Path = logs_dir / ts
        self._entries: list[str] = []
        self._admitter = admitter
        self._higher_is_better = higher_is_better
        self._scorer = scorer

    # ------ file paths ------
    @property
    def banks_file(self) -> Path:
        return self.session_dir / "banks.json"

    @property
    def programs_file(self) -> Path:
        return self.session_dir / "programs.json"

    @property
    def best_ideas_file(self) -> Path:
        return self.session_dir / "best_ideas.json"

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

        self.write_evolution_statistics(ideas, ts)

    def write_bank_snapshot(self, ideas: list[Idea], timestamp: str) -> None:
        """Write banks.json as a single active-bank snapshot of typed ideas."""
        banks_data = [
            {
                "active_bank": [idea.model_dump() for idea in ideas],
                "timestamp": timestamp,
            }
        ]
        self.banks_file.write_text(json.dumps(banks_data, indent=2), encoding="utf-8")

    def write_evolution_statistics(self, ideas: list[Idea], timestamp: str) -> None:
        """Run origin analysis and rewrite the bank snapshot with per-idea
        ``evolution_statistics`` stamped onto copies of the matching ideas."""
        if not self.banks_file.exists() or not self.programs_file.exists():
            return
        try:
            _result = _analyse_origins(
                banks_path=str(self.banks_file),
                programs_path=str(self.programs_file),
                admitter=self._admitter,
                higher_is_better=self._higher_is_better,
                scorer=self._scorer,
            )
        except RuntimeError as exc:
            if "No valid programs" in str(exc):
                return
            raise
        except Exception as exc:
            logger.warning(
                "[Memory][IdeaTracker] Could not compute evolutionary statistics: {}",
                exc,
            )
            return

        if not _result.summary:
            return

        stats_by_idea = CardStatsStamper().idea_statistics(_result.summary)

        enriched = [
            idea.model_copy(update={"evolution_statistics": stats_by_idea[idea.id]})
            if idea.id in stats_by_idea
            else idea
            for idea in ideas
        ]
        self.write_bank_snapshot(enriched, timestamp)

        best_ideas = [stats_row.as_json_row() for stats_row in _result.best_ideas]
        self.best_ideas_file.write_text(
            json.dumps([{"timestamp": timestamp, "best_ideas": best_ideas}], indent=2),
            encoding="utf-8",
        )


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
        fast.pop("recompute_center", None)
        extra = {k: v for k, v in fast.items() if k in _CLUSTERING_ANALYZER_KEYS}
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
        llm: Memory LLM router for the analyser (Hydra ``llms`` group, composed
            via ``/llms@ideas_tracker.llm``). Required when ``analyzer`` is None.
        analyzer_type: ``"default"`` → ClassifyingAnalyzer; ``"fast"`` → ClusteringAnalyzer.
        analyzer_fast_settings: Extra kwargs for ClusteringAnalyzer when ``fast``.
        memory_write_best_programs_percent: Share of top-fitness programs
            converted into program cards by the write pipeline.
        backend: Memory backend factory (Hydra ``memory/backend`` group) used
            by the write pipeline to build the card bank. Required whenever
            ``memory_write_enabled`` is True — ideas_tracker configs share the
            top-level ``memory.backend`` singleton via ``${ref:memory.backend}``.
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
        admitter: MemoryAdmitter | None = None,
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
                "backend factory; wire the shared `memory.backend` node (`${ref:memory.backend}`) "
                "(ideas_tracker configs do this by default) or pass "
                "memory_write_enabled=False."
            )

        _ensure_writable_hf_cache()
        if analyzer is None:
            if llm is None:
                raise ValueError(
                    "IdeaTracker: building an analyzer requires an LLM router; "
                    "compose `/llms@ideas_tracker.llm: gemini_flash_openrouter` "
                    "(ideas_tracker configs do this by default) or pass an "
                    "explicit analyzer."
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
        self._last_entry_count: dict[str, int] = {}

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
        self._log = _SessionLog(
            resolved_logs,
            admitter=admitter,
            higher_is_better=fitness_higher_is_better,
            scorer=self._reputation.scorer(),
        )

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
        time budget). ``posterior_programs`` feeds the cheap, pure injection
        posterior, which needs the full program set so child→parent lineage
        resolves; a capped window would sever lineage and collapse the
        intro-event population. Defaults to ``programs`` when not supplied.
        """
        records = self._eligible_records(programs)

        result = await self._analyzer.analyze_async(records, self._bank)
        self._bank.apply(result)

        if records:
            stale_ideas = _select_ideas_needing_enrichment(
                self._bank.all_ideas(), self._last_entry_count
            )
            if stale_ideas:
                enriched = await _enrich_ideas_with_keywords_and_summaries(
                    stale_ideas, self._analyzer, self._task_summary
                )
                for idea in enriched:
                    self._bank.enrich(
                        idea.id,
                        keywords=idea.keywords,
                        summary=idea.explanation.summary,
                        task_summary=self._task_summary,
                    )
            self._last_entry_count = {
                idea.id: len(idea.explanation.entries)
                for idea in self._bank.all_ideas()
            }

        self._log.record("pipeline_complete", total_ideas=len(self._bank.all_ideas()))
        self._log.flush(self._bank, records=self._all_records)

        card_posterior = _card_posterior_from_programs(
            programs if posterior_programs is None else posterior_programs,
            fitness_key=self._fitness_key,
            higher_is_better=self._fitness_higher_is_better,
            reputation=self._reputation,
            metrics_context=self._metrics_context,
        )

        _run_write_pipeline(
            self._memory_write_enabled,
            self._log.banks_file,
            self._log.best_ideas_file,
            self._log.programs_file,
            backend=self._backend,
            checkpoint_dir=self._checkpoint_dir,
            best_programs_percent=self._best_programs_percent,
            higher_is_better=self._fitness_higher_is_better,
            card_posterior=card_posterior,
            evictor=self._evictor,
            deduplicator=self._deduplicator,
        )

    def _eligible_records(self, programs: list[Program]) -> list[ProgramRecord]:
        """
        Filter programs and convert to ProgramRecord.

        Skips: root programs (no parents), programs without a validated fitness
        (missing/non-positive is_valid; missing, non-finite, or sentinel
        fitness), already-seen ids.
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

        parent_codes: dict[str, str] = {p.id: p.code for p in programs if p.code}
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
