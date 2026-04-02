"""Ideas Tracker: extract and rank improvement ideas from evolutionary program runs.

``IdeaTracker`` loads program data, classifies ideas with an LLM-backed analyzer,
maintains active and inactive idea banks, enriches records with postprocessing,
and optionally integrates with the memory write pipeline.
"""

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger
import pandas as pd
import tqdm

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
from gigaevo.memory.ideas_tracker.utils.cfg_loader import load_config
from gigaevo.memory.ideas_tracker.utils.dataframe_loader import load_dataframe
from gigaevo.memory.ideas_tracker.utils.helpers import (
    build_memory_usage_updates,
    sort_ideas,
)
from gigaevo.memory.ideas_tracker.utils.it_logger import IdeasTrackerLogger
from gigaevo.memory.ideas_tracker.utils.records_converter import (
    convert_programs_to_records,
)
from gigaevo.memory.ideas_tracker.utils.task_description_loader import (
    load_task_description,
)


class IdeaTracker:
    """Track and analyze ideas produced during evolutionary program search.

    Owns idea bank, runs the analyzer and postprocessing
    pipelines, and records program state for logging and optional memory export.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        logs_dir: str | Path | None = None,
    ) -> None:
        """Load YAML configuration and wire analyzer, postprocessing, and logging.

        Args:
            config_path: Path to the ideas_tracker YAML config, or None to use
                the default ``config/memory.yaml`` (ideas_tracker section) relative
                to the project layout resolved from this file.
            logs_dir: Optional directory for session logs; timestamped subfolders
                are created inside it. When None, uses ``.../ideas_tracker/logs``.
        """
        # -------Config loading-------
        self.config = load_config(config_path, Path(__file__).resolve())
        self.redis_config = self.config.redis_config
        mem = self.config.memory_config
        self.memory_write_pipeline_enabled = mem.memory_write_pipeline_enabled
        self.memory_write_best_programs_percent = mem.best_programs_percent
        self.memory_usage_tracking_enabled = mem.memory_usage_tracking_enabled

        # -------Logger initialization-------
        self.logger = IdeasTrackerLogger(
            Path(__file__).resolve(),
            logs_dir=logs_dir,
        )

        # -------Idea manager initialization-------
        self.ideas_manager = RecordManager(
            list_max_ideas=self.config.pipeline_config.list_max_ideas
        )
        self.ideas_manager.logger = self.logger
        # -------Analyzer initialization-------
        self.analyzer = create_analyzer(self.config.pipeline_config.analyzer_settings)
        self.analyzer_pipeline_type = self.config.pipeline_config.analyzer_pipeline_type
        if hasattr(self.analyzer, "description_rewriting"):
            self.analyzer.description_rewriting = (
                self.config.pipeline_config.description_rewriting
            )
        self.analyzer.logger = self.logger
        # -------Postprocessing pipeline initialization-------
        self.postprocessing = create_postprocessing(
            self.config.pipeline_config.postprocessing
        )
        # -------Task description loading-------
        self.task_description: str = load_task_description(
            self.redis_config.redis_prefix, Path(__file__).resolve()
        )
        # -------Internal variables initialization-------
        self.programs_card: list[ProgramRecord] = []
        self.programs_ids: set[str] = set()
        self.memory_usage_updates_by_card: dict[str, dict[str, Any]] = {}
        # -------Task description summary calculation-------
        self._get_task_description_summary()
        # -------Logger initialization-------
        self.logger.log_init(
            component="IdeaTracker",
            model_name=self.analyzer.model,
            redis_host=self.redis_config.redis_host,
            redis_port=self.redis_config.redis_port,
            redis_db=self.redis_config.redis_db,
            redis_prefix=self.redis_config.redis_prefix,
            label=self.redis_config.label,
            list_max_ideas=self.config.pipeline_config.list_max_ideas,
            memory_write_pipeline_enabled=self.memory_write_pipeline_enabled,
            memory_write_best_programs_percent=self.memory_write_best_programs_percent,
            memory_usage_tracking_enabled=self.memory_usage_tracking_enabled,
        )

    def _get_task_description_summary(self) -> str:
        """Return a short summary of the task description, computing and caching once.

        Returns:
            Cached summary string from ``_summarize_task_description``, or the
            existing non-empty cache if already set on the instance.
        """
        cached = getattr(self, "_task_description_summary_cache", None)
        if isinstance(cached, str) and cached:
            return cached
        summary = summarize_task_description(self.analyzer, self.task_description)
        self._task_description_summary_cache = summary
        return summary

    def _programs_to_dicts(self) -> list[dict[str, Any]]:
        """Serialize ``programs_card`` entries for logging (JSON-friendly dicts).

        Returns:
            A list of dicts, one per ``ProgramRecord`` via ``to_dict()``.
        """
        return [prog.to_dict() for prog in self.programs_card]

    def wrap_data(self, programs: pd.DataFrame) -> list[ProgramRecord]:
        """Convert program rows to ``ProgramRecord`` instances and append internal state.

        Args:
            programs: DataFrame rows suitable for ``convert_programs_to_records``
                (e.g. ``program_id``, ``metric_fitness``, ``generation``,
                ``parent_ids``, ``metadata_mutation_output``).

        Returns:
            Newly built records; they are also appended to ``programs_card`` and
            their ids merged into ``programs_ids``.
        """
        task_description_summary = self._get_task_description_summary()
        programs_processed, programs_ids = convert_programs_to_records(
            programs, self.task_description, task_description_summary
        )
        self.programs_card.extend(programs_processed)
        self.programs_ids.update(programs_ids)
        return programs_processed

    def get_new_programs(self, df: pd.DataFrame) -> list[ProgramRecord]:
        """Return programs that are new, non-root, and have positive fitness.

        Keeps rows with ``metric_fitness > 0``, ``is_root`` false, and
        ``program_id`` not already present in ``programs_ids``, then wraps them
        via ``wrap_data``.

        Args:
            df: Full program table from the configured data source.

        Returns:
            ``ProgramRecord`` list for programs not yet tracked, after conversion.
        """
        search_condition = (df["metric_fitness"] > 0) & (~df["is_root"])
        valid_programs = df[search_condition]
        all_programs = set(valid_programs["program_id"])
        new_programs = all_programs.difference(self.programs_ids)
        mask = df["program_id"].isin(new_programs)
        selected_programs = df[mask].copy()
        converted_new_programs = self.wrap_data(selected_programs)
        return converted_new_programs

    def process_program(self, program: ProgramRecord) -> None:
        """Classify ideas from one program and update active/inactive idea banks.

        Runs the analyzer on ``program.improvements``, then applies new, rewrite,
        and update actions through ``ideas_manager``.

        Args:
            program: Source program record (improvements and lineage metadata).
        """
        active_ideas = self.ideas_manager.ideas_groups_texts()
        inactive_ideas = self.ideas_manager.ideas_groups_texts(use_inactive=True)
        program_ideas = IncomingIdeas(program.improvements)
        classified_ideas = self.analyzer.process_ideas(
            program_ideas,
            active_ideas,
            inactive_ideas,  # type: ignore[arg-type]
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

    def enrich_ideas(self) -> None:
        """Enrich every idea in the record bank with postprocessing LLM outputs.

        Runs the configured postprocessing pipeline (e.g. keywords and explanation
        summary) and persists results via ``ideas_manager.enrich_idea_metadata``.
        """
        all_uuids = list(self.ideas_manager.record_bank.uuids)
        all_ideas = [
            self.ideas_manager.record_bank.get_idea(uuid) for uuid in all_uuids
        ]
        task_description_summary = self._get_task_description_summary()
        enriched_ideas = self.postprocessing(
            all_ideas, self.analyzer, task_description_summary
        )
        for idea in enriched_ideas:
            self.ideas_manager.enrich_idea_metadata(
                idea.id,
                idea.keywords,
                idea.explanation["summary"],
                task_description_summary,
            )

    def default_analyzer_pipeline(self, new_programs: list[ProgramRecord]) -> None:
        """Process each program with ``process_program`` (sequential, with tqdm).

        Args:
            new_programs: Batch of programs to analyze one by one.
        """
        pbar = tqdm.tqdm(total=len(new_programs), leave=False)
        for prog in new_programs:
            self.process_program(prog)
            pbar.update(1)
        pbar.close()

    def fast_analyzer_pipeline(self, new_programs: list[ProgramRecord]) -> None:
        """Run the batched fast analyzer and import extended idea records.

        Uses ``asyncio.run`` on ``IdeaAnalyzerFast.run``; imports each result into
        ``ideas_manager.record_bank`` with ``is_forced=True``.

        Args:
            new_programs: Programs to pass to the fast analyzer pipeline.

        Raises:
            TypeError: If the configured analyzer is not ``IdeaAnalyzerFast``.
        """
        if not isinstance(self.analyzer, IdeaAnalyzerFast):
            raise TypeError(
                "fast_analyzer_pipeline requires analyzer config type 'fast' (IdeaAnalyzerFast)"
            )
        is_async_conversion = (
            self.config.pipeline_config.analyzer_settings.get(
                "record_conversion", {}
            ).get("type", "default")
            == "fast"
        )
        processed_programs = asyncio.run(
            self.analyzer.run(
                new_programs, record_extended_multi_async=is_async_conversion
            )
        )
        for prog in processed_programs:
            self.ideas_manager.record_bank.import_idea_extended(prog, is_forced=True)

    def run(self, path_to_database: str | Path | None = None) -> None:
        """End-to-end run: load data, analyze new programs, enrich ideas, log, memory I/O.

        Optionally builds memory-usage updates, runs default or fast analyzer
        pipeline, applies memory updates to idea banks, dumps final state,
        logs programs, computes evolutionary statistics, and runs the memory
        write pipeline when enabled.

        Args:
            path_to_database: Optional path to a local dataframe source; if None,
                ``load_dataframe`` uses Redis settings from config.
        """
        df = load_dataframe(self.redis_config, path_to_database)
        if df.empty:
            logger.warning(
                "Ideas tracker skipped: no programs found in the selected source."
            )
            return

        if self.memory_usage_tracking_enabled:
            self.memory_usage_updates_by_card = build_memory_usage_updates(
                df, self._get_task_description_summary()
            )
            self.logger.log_memory_usage_updates(self.memory_usage_updates_by_card)
        else:
            self.memory_usage_updates_by_card = {}

        new_programs_processed = self.get_new_programs(df)

        if self.analyzer_pipeline_type == "default":
            self.default_analyzer_pipeline(new_programs_processed)
        else:
            self.fast_analyzer_pipeline(new_programs_processed)

        if self.memory_usage_tracking_enabled and self.memory_usage_updates_by_card:
            apply_memory_usage_updates_to_idea_banks(
                self.ideas_manager.record_bank, self.memory_usage_updates_by_card
            )

        self.enrich_ideas()

        self.logger.dump_final_state(self.ideas_manager)
        self.logger.log_programs(self._programs_to_dicts())

        # --- Evolutionary statistics (origin analysis) ---
        compute_evolutionary_statistics(self.logger)

        # --- Optional memory DB write for best ideas ---
        run_memory_write_pipeline(
            self.memory_write_pipeline_enabled,
            self.memory_usage_tracking_enabled,
            self.logger,
        )


if __name__ == "__main__":
    p = Path(__file__).resolve().parent / "test.csv"
    itd = IdeaTracker()
    itd.run(p)
