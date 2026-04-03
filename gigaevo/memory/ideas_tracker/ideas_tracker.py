"""Ideas Tracker: extract and rank improvement ideas from evolutionary program runs.

``IdeaTracker`` is a ``PostRunHook`` — the evolution engine calls
``on_run_complete(storage)`` after the generation loop finishes.  It works
directly with ``Program`` objects (no DataFrame conversion).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
import tqdm

from gigaevo.evolution.engine.hooks import PostRunHook
from gigaevo.memory.ideas_tracker.components.analyzer import IdeaAnalyzer
from gigaevo.memory.ideas_tracker.components.analyzer_f import IdeaAnalyzerFast
from gigaevo.memory.ideas_tracker.components.data_components import (
    IncomingIdeas,
    ProgramRecord,
)
from gigaevo.memory.ideas_tracker.components.fabrics.analyzer_fabric import (
    create_analyzer,
)
from gigaevo.memory.ideas_tracker.components.fabrics.postprocessing_fabric import (
    create_postprocessing,
)
from gigaevo.memory.ideas_tracker.components.memory_pipeline import (
    apply_memory_usage_updates_to_idea_banks,
    run_memory_write_pipeline,
)
from gigaevo.memory.ideas_tracker.components.records_manager import RecordManager
from gigaevo.memory.ideas_tracker.components.statistics import (
    compute_evolutionary_statistics,
)
from gigaevo.memory.ideas_tracker.components.summary import summarize_task_description
from gigaevo.memory.ideas_tracker.utils.helpers import (
    build_memory_usage_updates_from_programs,
    sort_ideas,
    to_float,
)
from gigaevo.memory.ideas_tracker.utils.it_logger import IdeasTrackerLogger
from gigaevo.memory.ideas_tracker.utils.records_converter import programs_to_records
from gigaevo.memory.ideas_tracker.utils.task_description_loader import (
    load_task_description,
)
from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS, Program

if TYPE_CHECKING:
    from gigaevo.database.program_storage import ProgramStorage


class IdeaTracker(PostRunHook):
    """PostRunHook that analyses programs and classifies improvement ideas.

    Instantiated via Hydra (``ideas_tracker=default`` or ``fast``).
    Accepts ``list[Program]`` directly — no DataFrame conversion.
    """

    def __init__(
        self,
        *,
        task_description: str = "",
        analyzer_type: str = "default",
        analyzer_model: str = "google/gemini-3-flash-preview",
        analyzer_base_url: str = "https://openrouter.ai/api/v1",
        analyzer_reasoning: dict[str, Any] | None = None,
        analyzer_fast_settings: dict[str, Any] | None = None,
        list_max_ideas: int = 20,
        postprocessing_type: str = "default",
        description_rewriting: bool = True,
        record_conversion_type: str = "default",
        memory_write_enabled: bool = True,
        memory_write_best_programs_percent: float = 5.0,
        memory_usage_tracking_enabled: bool = True,
        fitness_key: str = "fitness",
        checkpoint_dir: str | None = None,
        namespace: str | None = None,
        redis_prefix: str = "",
        logs_dir: str | Path | None = None,
    ) -> None:
        # Config
        self.memory_write_enabled = memory_write_enabled
        self.memory_write_best_programs_percent = memory_write_best_programs_percent
        self.memory_usage_tracking_enabled = memory_usage_tracking_enabled
        self._fitness_key = fitness_key
        self.analyzer_pipeline_type = analyzer_type
        self._record_conversion_type = record_conversion_type

        # Logger
        self.logger = IdeasTrackerLogger(
            Path(__file__).resolve(),
            logs_dir=logs_dir,
        )

        # Idea manager
        self.ideas_manager = RecordManager(list_max_ideas=list_max_ideas)
        self.ideas_manager.logger = self.logger

        # Analyzer
        analyzer_config: dict[str, Any] = {
            "type": analyzer_type,
            "model": analyzer_model,
            "base_url": analyzer_base_url,
        }
        if analyzer_reasoning is not None:
            analyzer_config["reasoning"] = analyzer_reasoning
        if analyzer_fast_settings is not None:
            analyzer_config["analyzer_fast_settings"] = analyzer_fast_settings
        self.analyzer = create_analyzer(analyzer_config)
        if hasattr(self.analyzer, "description_rewriting"):
            self.analyzer.description_rewriting = description_rewriting
        self.analyzer.logger = self.logger

        # Postprocessing
        self.postprocessing = create_postprocessing({"type": postprocessing_type})

        # Task description
        if task_description:
            self.task_description = task_description
        else:
            self.task_description = load_task_description(
                redis_prefix, Path(__file__).resolve()
            )

        # Internal state
        self.programs_card: list[ProgramRecord] = []
        self.programs_ids: set[str] = set()
        self.memory_usage_updates_by_card: dict[str, dict[str, Any]] = {}

        # Compute and cache task summary
        self._task_description_summary_cache: str = ""
        self._get_task_description_summary()

        # Init log
        self.logger.log_init(
            component="IdeaTracker",
            model_name=self.analyzer.model,
            list_max_ideas=list_max_ideas,
            memory_write_enabled=memory_write_enabled,
            memory_write_best_programs_percent=memory_write_best_programs_percent,
            memory_usage_tracking_enabled=memory_usage_tracking_enabled,
        )

    # ------------------------------------------------------------------
    # PostRunHook interface
    # ------------------------------------------------------------------

    async def on_run_complete(self, storage: ProgramStorage) -> None:
        """Called by EvolutionEngine after the generation loop finishes."""
        programs = await storage.get_all(exclude=EXCLUDE_STAGE_RESULTS)
        if not programs:
            logger.warning("IdeaTracker: no programs in storage, skipping.")
            return
        self._run_on_programs(programs)

    # ------------------------------------------------------------------
    # CLI entry point
    # ------------------------------------------------------------------

    def run(self, programs: list[Program] | None = None) -> None:
        """CLI entry: accepts ``list[Program]`` directly."""
        if not programs:
            return
        self._run_on_programs(programs)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _run_on_programs(self, programs: list[Program]) -> None:
        """Core pipeline: filter, analyze, enrich, log, write."""
        if self.memory_usage_tracking_enabled:
            self.memory_usage_updates_by_card = (
                build_memory_usage_updates_from_programs(
                    programs,
                    self._get_task_description_summary(),
                    fitness_key=self._fitness_key,
                )
            )
            self.logger.log_memory_usage_updates(self.memory_usage_updates_by_card)
        else:
            self.memory_usage_updates_by_card = {}

        new_programs_processed = self._get_new_programs(programs)

        if self.analyzer_pipeline_type == "default":
            self._default_analyzer_pipeline(new_programs_processed)
        else:
            self._fast_analyzer_pipeline(new_programs_processed)

        if self.memory_usage_tracking_enabled and self.memory_usage_updates_by_card:
            apply_memory_usage_updates_to_idea_banks(
                self.ideas_manager.record_bank, self.memory_usage_updates_by_card
            )

        self._enrich_ideas()
        self.logger.dump_final_state(self.ideas_manager)
        self.logger.log_programs(self._programs_to_dicts())
        compute_evolutionary_statistics(self.logger)
        run_memory_write_pipeline(
            self.memory_write_enabled,
            self.memory_usage_tracking_enabled,
            self.logger,
        )

    # ------------------------------------------------------------------
    # Program filtering
    # ------------------------------------------------------------------

    def _get_new_programs(self, programs: list[Program]) -> list[ProgramRecord]:
        """Filter roots, zero/negative fitness, duplicates; convert to records."""
        new_programs: list[Program] = []
        for prog in programs:
            if not prog.lineage.parents:
                continue
            fitness = to_float(prog.metrics.get(self._fitness_key))
            if fitness is None or fitness <= 0:
                continue
            if prog.id in self.programs_ids:
                continue
            new_programs.append(prog)

        task_summary = self._get_task_description_summary()
        records, ids = programs_to_records(
            new_programs,
            self.task_description,
            task_summary,
            fitness_key=self._fitness_key,
        )
        self.programs_card.extend(records)
        self.programs_ids.update(ids)
        return records

    # ------------------------------------------------------------------
    # Analyzer pipelines
    # ------------------------------------------------------------------

    def _default_analyzer_pipeline(self, new_programs: list[ProgramRecord]) -> None:
        if not isinstance(self.analyzer, IdeaAnalyzer):
            raise TypeError(
                "default analyzer pipeline requires IdeaAnalyzer (not IdeaAnalyzerFast)"
            )
        pbar = tqdm.tqdm(total=len(new_programs), leave=False)
        for prog in new_programs:
            self._process_program(prog)
            pbar.update(1)
        pbar.close()

    def _fast_analyzer_pipeline(self, new_programs: list[ProgramRecord]) -> None:
        if not isinstance(self.analyzer, IdeaAnalyzerFast):
            raise TypeError("fast analyzer pipeline requires IdeaAnalyzerFast")
        is_async_conversion = self._record_conversion_type == "fast"
        processed_programs = asyncio.run(
            self.analyzer.run(
                new_programs, record_extended_multi_async=is_async_conversion
            )
        )
        for prog in processed_programs:
            self.ideas_manager.record_bank.import_idea_extended(prog, is_forced=True)

    def _process_program(self, program: ProgramRecord) -> None:
        active_ideas = self.ideas_manager.ideas_groups_texts()
        inactive_ideas = self.ideas_manager.ideas_groups_texts(use_inactive=True)
        program_ideas = IncomingIdeas(program.improvements)
        classified_ideas = self.analyzer.process_ideas(
            program_ideas,
            active_ideas,
            inactive_ideas,
        )
        sorted_ideas = sort_ideas(classified_ideas.ideas)
        for n_idea in sorted_ideas["new"]:
            self.ideas_manager.add_new_idea(
                description=n_idea["description"],
                linked_program=program.id,
                generation=program.generation,
                category=program.category,
                strategy=program.strategy,
                task_description=program.task_description,
                change_motivation=n_idea["change_motivation"],
            )
        for idea_r in sorted_ideas["rewrite"]:
            self.ideas_manager.modify_idea(
                idea_r["target_idea_id"],
                [program.id],
                program.generation,
                idea_r["description"],
                idea_r["change_motivation"],
            )
        for idea_u in sorted_ideas["update"]:
            self.ideas_manager.modify_idea(
                idea_u["target_idea_id"],
                [program.id],
                program.generation,
                None,
                idea_u["change_motivation"],
            )

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------

    def _enrich_ideas(self) -> None:
        all_uuids = list(self.ideas_manager.record_bank.uuids)
        all_ideas = [
            self.ideas_manager.record_bank.get_idea(uuid) for uuid in all_uuids
        ]
        task_summary = self._get_task_description_summary()
        result = self.postprocessing(all_ideas, self.analyzer, task_summary)
        if asyncio.iscoroutine(result):
            enriched_ideas = asyncio.run(result)
        else:
            enriched_ideas = result
        for idea in enriched_ideas:
            self.ideas_manager.enrich_idea_metadata(
                idea.id,
                idea.keywords,
                idea.explanation["summary"],
                task_summary,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_task_description_summary(self) -> str:
        cached = self._task_description_summary_cache
        if cached:
            return cached
        summary = summarize_task_description(self.analyzer, self.task_description)
        self._task_description_summary_cache = summary
        return summary

    def _programs_to_dicts(self) -> list[dict[str, Any]]:
        return [prog.to_dict() for prog in self.programs_card]
