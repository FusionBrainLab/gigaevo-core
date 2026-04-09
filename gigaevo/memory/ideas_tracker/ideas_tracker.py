"""
IdeaTracker: PostRunHook that extracts, classifies, enriches, and stores
improvement ideas from a completed evolutionary run.

_SessionLog accumulates log entries in memory and writes all files to a
timestamped directory in a single flush() call at session end.
"""

from __future__ import annotations

import ast
import asyncio
from datetime import datetime
from functools import cached_property
import importlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from loguru import logger
import pandas as pd

from gigaevo.evolution.engine.hooks import PostRunHook
from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
)
from gigaevo.memory.ideas_tracker.analyzers import (
    Analyzer,
    ClassifyingAnalyzer,
    ClusteringAnalyzer,
)
from gigaevo.memory.ideas_tracker.idea_bank import IdeaBank, build_usage_payload
from gigaevo.memory.ideas_tracker.models import (
    Idea,
    IdeaExplanation,
    ProgramRecord,
    UsagePayload,
    program_to_record,
)
from gigaevo.memory.ideas_tracker.utils.origin_analysis import compute_origin_analysis
from gigaevo.memory.utils import to_float
from gigaevo.programs.metrics.context import VALIDITY_KEY
from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS, Program

if TYPE_CHECKING:
    from gigaevo.database.program_storage import ProgramStorage

load_dotenv()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


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
            "[Memory] Failed to read task description for prefix {!r}: {}", prefix, exc
        )
    return "No description available"


def _summarise_task_description(analyzer: Analyzer, task_description: str) -> str:
    """Ask the LLM for a compact summary of the task description."""
    text = str(task_description or "").strip()
    if not text:
        return "Task summary unavailable"
    try:
        raw = analyzer.call("task_description_summary", text)
        parsed = json.loads(raw)
        summary = str(parsed.get("summary", "")).strip()
        return summary or text[:240].strip()
    except Exception as exc:
        logger.warning(
            "[Memory] Task description summarization failed, using truncated text: {}",
            exc,
        )
        return text[:240].strip()


def _build_usage_updates(
    programs: list[Program],
    task_summary: str,
    fitness_key: str,
) -> dict[str, UsagePayload]:
    """Build per-memory-card usage payloads from program fitness deltas."""

    def _as_string_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(i).strip() for i in value if str(i).strip()]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text[0] in "[{(":
                try:
                    return [str(i).strip() for i in json.loads(text) if str(i).strip()]
                except Exception:
                    try:
                        return [
                            str(i).strip()
                            for i in ast.literal_eval(text)
                            if str(i).strip()
                        ]
                    except Exception:
                        pass
            return [text]
        return []

    fitness_by_id: dict[str, float] = {}
    for prog in programs:
        is_valid = to_float(prog.metrics.get(VALIDITY_KEY))
        if is_valid is None or is_valid <= 0:
            continue
        f = to_float(prog.metrics.get(fitness_key))
        if f is not None:
            fitness_by_id[prog.id] = f

    usage_by_card: dict[str, dict[str, list[float]]] = {}
    for prog in programs:
        is_valid = to_float(prog.metrics.get(VALIDITY_KEY))
        if is_valid is None or is_valid <= 0:
            continue
        selected = _as_string_list(
            prog.metadata.get(MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY)
        )
        if not selected:
            continue
        child_fitness = to_float(prog.metrics.get(fitness_key))
        if child_fitness is None:
            continue
        parent_fitnesses = [
            fitness_by_id[pid] for pid in prog.lineage.parents if pid in fitness_by_id
        ]
        if not parent_fitnesses:
            continue
        delta = child_fitness - max(parent_fitnesses)
        for card_id in list(dict.fromkeys(selected)):
            usage_by_card.setdefault(card_id, {}).setdefault(task_summary, []).append(
                delta
            )

    return {
        card_id: build_usage_payload(task_deltas)
        for card_id, task_deltas in usage_by_card.items()
    }


async def _enrich_ideas(
    ideas: list[Idea], analyzer: Analyzer, task_summary: str
) -> list[Idea]:
    """Enrich all ideas concurrently with keywords and explanation summaries."""

    async def _enrich_one(idea: Idea) -> Idea:
        keywords: list[str] = []
        try:
            kw_raw = await analyzer.call_async("keywords", idea.description)
            keywords = json.loads(kw_raw).get("keywords", [])
        except Exception as exc:
            logger.warning(
                "[Memory] Keyword extraction failed for idea {!r}: {}", idea.id, exc
            )

        summary = ""
        entries = idea.explanation.entries
        if len(entries) == 1:
            summary = entries[0]
        elif len(entries) > 1:
            explanations_text = "\n".join(f"- {e}" for e in entries)
            try:
                sum_raw = await analyzer.call_async("usage_summary", explanations_text)
                summary = json.loads(sum_raw).get("summary", "")
            except Exception as exc:
                logger.warning(
                    "[Memory] Summary generation failed for idea {!r}: {}", idea.id, exc
                )

        return idea.model_copy(
            update={
                "keywords": keywords,
                "explanation": IdeaExplanation(entries=entries, summary=summary),
                "task_description_summary": task_summary,
            }
        )

    return list(await asyncio.gather(*[_enrich_one(idea) for idea in ideas]))


def _run_write_pipeline(
    enabled: bool,
    banks_path: Path | None,
    best_ideas_path: Path | None,
    programs_path: Path | None,
    usage_updates_path: Path | None,
    memory_usage_tracking_enabled: bool,
) -> None:
    """Optionally trigger the downstream memory write pipeline."""
    if not enabled:
        return
    if banks_path is None or best_ideas_path is None:
        logger.warning("Memory write pipeline skipped: log paths unavailable.")
        return
    if not banks_path.exists():
        logger.warning(f"Memory write pipeline skipped: missing {banks_path}.")
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
        logger.warning("Memory write pipeline skipped: no best_ideas snapshot.")
        return

    env_overrides = {
        "MEMORY_BANKS_PATH": str(banks_path),
        "MEMORY_BEST_IDEAS_PATH": str(best_ideas_path),
    }
    if programs_path and programs_path.exists():
        env_overrides["MEMORY_PROGRAMS_PATH"] = str(programs_path)
    if (
        memory_usage_tracking_enabled
        and usage_updates_path
        and usage_updates_path.exists()
    ):
        env_overrides["MEMORY_USAGE_UPDATES_PATH"] = str(usage_updates_path)

    previous = {k: os.environ.get(k) for k in env_overrides}
    try:
        os.environ.update(env_overrides)
        mod = importlib.import_module("gigaevo.memory.write_pipeline")
        mod = importlib.reload(mod)
        snapshot = mod.main()
        if isinstance(snapshot, dict):
            stats = snapshot.get("stats", {})
            if isinstance(stats, dict):
                logger.info(
                    f"Memory write: processed={stats.get('processed', 0)}, "
                    f"added={stats.get('added', 0)}, updated={stats.get('updated', 0)}, "
                    f"rejected={stats.get('rejected', 0)}"
                )
    finally:
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# _SessionLog
# ---------------------------------------------------------------------------


class _SessionLog:
    """
    Accumulates log entries in memory during a tracker run and writes all
    files to a timestamped session directory in a single flush() call.

    Replaces the per-event read-modify-write pattern of IdeasTrackerLogger.
    Files written: log.txt, banks.json, programs.json, best_ideas.json,
    memory_usage_updates.json.
    """

    def __init__(self, logs_dir: Path) -> None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir: Path = logs_dir / ts
        self._entries: list[str] = []
        self._usage_updates: dict[str, UsagePayload] = {}

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

    @property
    def usage_updates_file(self) -> Path:
        return self.session_dir / "memory_usage_updates.json"

    # ------ recording ------

    def record(self, action: str, **params: Any) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"[{ts}]: {action}"]
        for k, v in params.items():
            lines.append(f"  {k}: {v}")
        self._entries.append("\n".join(lines))

    def record_usage_updates(self, updates: dict[str, UsagePayload]) -> None:
        self._usage_updates = updates

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

        banks_data = [
            {
                "active_bank": [i.model_dump() for i in bank.all_ideas()],
                "timestamp": ts,
            }
        ]
        self.banks_file.write_text(json.dumps(banks_data, indent=2), encoding="utf-8")

        programs_data = [
            {
                "timestamp": ts,
                "programs": [r.model_dump() for r in records],
            }
        ]
        self.programs_file.write_text(
            json.dumps(programs_data, indent=2), encoding="utf-8"
        )

        self.usage_updates_file.write_text(
            json.dumps(
                [
                    {
                        "timestamp": ts,
                        "usage_updates": {
                            card_id: payload.model_dump()
                            for card_id, payload in self._usage_updates.items()
                        },
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )

        self._compute_and_write_statistics()

    def _compute_and_write_statistics(self) -> None:
        """Run origin analysis and inject per-idea statistics into banks.json."""
        if not self.banks_file.exists() or not self.programs_file.exists():
            return
        try:
            df_summary, df_best_ideas = compute_origin_analysis(
                banks_path=str(self.banks_file),
                programs_path=str(self.programs_file),
            )
        except RuntimeError as exc:
            if "No valid programs" in str(exc):
                return
            raise
        except Exception as exc:
            logger.warning(f"Could not compute evolutionary statistics: {exc}")
            return

        if df_summary.empty:
            return

        # Inject per-idea stats into banks.json
        stats_by_idea: dict[str, dict] = {}
        for _, row in df_summary.iterrows():
            idea_id = str(row["idea_id"])
            quartile = str(row["quartile"])
            metrics = {
                k: (v if pd.notna(v) else None)
                for k, v in row.drop(["idea_id", "quartile", "description"]).items()
            }
            stats_by_idea.setdefault(idea_id, {})[quartile] = metrics

        banks_data = json.loads(self.banks_file.read_text(encoding="utf-8"))
        for snapshot in banks_data:
            if not isinstance(snapshot, dict):
                continue
            for idea in snapshot.get("active_bank", []):
                if isinstance(idea, dict) and idea.get("id") in stats_by_idea:
                    idea["evolution_statistics"] = stats_by_idea[idea["id"]]
        self.banks_file.write_text(json.dumps(banks_data, indent=2), encoding="utf-8")

        # Write best_ideas.json
        best_ideas = [
            {k: (v if pd.notna(v) else None) for k, v in row.items()}
            for _, row in df_best_ideas.iterrows()
        ]
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.best_ideas_file.write_text(
            json.dumps([{"timestamp": ts, "best_ideas": best_ideas}], indent=2),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# IdeaTracker
# ---------------------------------------------------------------------------


class IdeaTracker(PostRunHook):
    """
    PostRunHook that extracts, classifies, enriches, and stores improvement
    ideas from a completed evolutionary run.

    Instantiated by Hydra. Accepts a ClassifyingAnalyzer or ClusteringAnalyzer —
    both implement the Analyzer protocol, so the pipeline is identical for both.

    Args:
        analyzer: The idea analyser to use. If None, defaults to ClassifyingAnalyzer
            with its own default model — useful for the CLI entry point.
        task_description: Human-readable description of the current task. If empty,
            loaded from the matching problems/ directory using redis_prefix.
        redis_prefix: Redis key prefix (e.g. "chains/hotpotqa/static") used to
            locate the task_description.txt file when task_description is empty.
        chunk_size: Number of ideas per LLM classification batch.
        memory_write_enabled: If True, trigger the downstream memory write pipeline.
        memory_usage_tracking_enabled: If True, compute fitness deltas for memory cards.
        fitness_key: Metric key to use as fitness (default "fitness").
        logs_dir: Directory for timestamped session logs. Defaults to
            gigaevo/memory/ideas_tracker/logs/.
    """

    def __init__(
        self,
        *,
        analyzer: ClassifyingAnalyzer | ClusteringAnalyzer | None = None,
        task_description: str = "",
        redis_prefix: str = "",
        chunk_size: int = 5,
        memory_write_enabled: bool = True,
        memory_usage_tracking_enabled: bool = True,
        fitness_key: str = "fitness",
        logs_dir: str | Path | None = None,
    ) -> None:
        if analyzer is None:
            analyzer = ClassifyingAnalyzer()

        self._analyzer: ClassifyingAnalyzer | ClusteringAnalyzer = analyzer
        self._bank = IdeaBank(chunk_size=chunk_size)
        self._fitness_key = fitness_key
        self._memory_write_enabled = memory_write_enabled
        self._memory_usage_tracking_enabled = memory_usage_tracking_enabled
        self._all_records: list[ProgramRecord] = []
        self._seen_ids: set[str] = set()

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
            logger.warning("IdeaTracker: no programs in storage, skipping.")
            return
        await self._run(programs)

    # ------------------------------------------------------------------
    # CLI entry point
    # ------------------------------------------------------------------

    def run(self, programs: list[Program] | None = None) -> None:
        """CLI entry: accepts list[Program] directly."""
        if not programs:
            return
        asyncio.run(self._run(programs))

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    async def _run(self, programs: list[Program]) -> None:
        """Full pipeline: filter → analyse → enrich → log → write."""
        if self._memory_usage_tracking_enabled:
            usage_updates = _build_usage_updates(
                programs, self._task_summary, self._fitness_key
            )
            self._log.record_usage_updates(usage_updates)
        else:
            usage_updates = {}

        records = self._eligible_records(programs)

        result = await self._analyzer.analyze_async(records, self._bank)
        self._bank.apply(result)

        if self._memory_usage_tracking_enabled and usage_updates:
            self._bank.apply_usage_updates(usage_updates)

        enriched = await _enrich_ideas(
            self._bank.all_ideas(), self._analyzer, self._task_summary
        )
        for idea in enriched:
            self._bank.enrich(
                idea.id,
                keywords=idea.keywords,
                summary=idea.explanation.summary,
                task_summary=self._task_summary,
            )

        self._log.record("pipeline_complete", total_ideas=len(self._bank.all_ideas()))
        self._log.flush(self._bank, records=self._all_records)

        _run_write_pipeline(
            self._memory_write_enabled,
            self._log.banks_file,
            self._log.best_ideas_file,
            self._log.programs_file,
            self._log.usage_updates_file,
            self._memory_usage_tracking_enabled,
        )

    def _eligible_records(self, programs: list[Program]) -> list[ProgramRecord]:
        """
        Filter programs and convert to ProgramRecord.

        Skips: root programs (no parents), invalid programs (is_valid != 1.0), already-seen ids.
        """
        eligible: list[Program] = []
        for prog in programs:
            if not prog.lineage.parents:
                continue
            is_valid = to_float(prog.metrics.get(VALIDITY_KEY))
            if is_valid is None or is_valid <= 0:
                continue
            if prog.id in self._seen_ids:
                continue
            eligible.append(prog)

        records = [
            program_to_record(
                p, self._task_description, self._task_summary, self._fitness_key
            )
            for p in eligible
        ]
        self._all_records.extend(records)
        self._seen_ids.update(p.id for p in eligible)
        return records
