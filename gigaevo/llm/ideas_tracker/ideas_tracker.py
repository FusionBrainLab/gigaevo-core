import ast
import asyncio
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Optional

import pandas as pd
import tqdm

sys.path.append("../gigaevo-core-internal")
from gigaevo.llm.ideas_tracker.components.analyzer_f import IdeaAnalyzerFast
from gigaevo.llm.ideas_tracker.components.data_components import (
    IncomingIdeas,
    ProgramRecord,
)
from gigaevo.llm.ideas_tracker.components.fabrics.analyzer_fabric import create_analyzer
from gigaevo.llm.ideas_tracker.components.fabrics.fabric_redis import (
    create_redis_config,
)
from gigaevo.llm.ideas_tracker.components.records_manager import RecordManager
from gigaevo.llm.ideas_tracker.components.statistics import (
    compute_evolutionary_statistics,
)
from gigaevo.llm.ideas_tracker.components.summary import _summarize_task_description
from gigaevo.llm.ideas_tracker.utils.cfg_loader import _load_config
from gigaevo.llm.ideas_tracker.utils.it_logger import IdeasTrackerLogger
from gigaevo.llm.ideas_tracker.utils.records_converter import (
    convert_programs_to_records,
)
from gigaevo.llm.ideas_tracker.utils.task_description_loader import (
    _load_task_description,
)
from gigaevo.utils.redis import fetch_evolution_dataframe


class IdeaTracker:
    """
    Main class for tracking and analyzing ideas from evolutionary program runs.

    Manages idea banks (active and inactive), processes programs to extract ideas,
    and maintains rankings of ideas based on their impact on program fitness.
    """

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        *,
        logs_dir: str | Path | None = None,
    ) -> None:
        """
        Initialize IdeaTracker with configuration from YAML file.

        Sets up idea management, LLM analyzer, Redis connection, logging, and
        statistics tracking. Initializes both active and inactive idea banks.

        Args:
            config_path: Optional path to configuration YAML file. If None, uses
                default config/memory.yaml from project root (ideas_tracker section).
        """
        self.config = _load_config(config_path, Path(__file__).resolve())
        self.logger = IdeasTrackerLogger(Path(__file__).resolve(), logs_dir=logs_dir)

        list_max_ideas = int(self.config.get("list_max_ideas", 5))
        self.ideas_manager = RecordManager(list_max_ideas=list_max_ideas)

        analyzer_cfg = self.config.get("analyzer", {})
        analyzer_cfg["analyzer_fast_settings"] = self.config.get(
            "analyzer_fast_settings", {}
        )
        self.analyzer = create_analyzer(analyzer_cfg)
        self.analyzer_pipeline_type = analyzer_cfg.get("type", "default")

        self.programs_card: list[ProgramRecord] = []
        self.programs_ids: set[str] = set()
        self.memory_usage_updates_by_card: dict[str, dict[str, Any]] = {}

        self.gen_delta: int = int(self.config.get("gen_delta", 100000))

        self.redis_config = create_redis_config(self.config.get("redis", {}))

        self.task_description: str = _load_task_description(
            self.redis_config.redis_prefix, Path(__file__).resolve()
        )

        self.description_rewriting = self.config.get("description_rewriting", False)
        if hasattr(self.analyzer, "description_rewriting"):
            self.analyzer.description_rewriting = self.description_rewriting

        statistics_config = self.config.get("statistics", {}) or {}
        self.top_k_delta_fitness = statistics_config.get("top_k_delta_fitness", 5)
        self.top_k_fitness = statistics_config.get("top_k_fitness", 5)
        self.statistics_enabled = statistics_config.get("enabled", True)
        self.statistics_mode = statistics_config.get("mode", "top_k")

        memory_write_cfg = self.config.get("memory_write_pipeline", False)
        if isinstance(memory_write_cfg, dict):
            self.memory_write_pipeline_enabled = self._to_bool(
                memory_write_cfg.get("enabled", False),
                default=False,
            )
            best_programs_percent = self._to_float(
                memory_write_cfg.get("best_programs_percent")
            )
        else:
            self.memory_write_pipeline_enabled = self._to_bool(
                memory_write_cfg,
                default=False,
            )
            best_programs_percent = None
        self.memory_write_best_programs_percent = max(
            0.0,
            best_programs_percent if best_programs_percent is not None else 5.0,
        )

        usage_tracking_cfg = self.config.get("usage_tracking", {"enabled": True})
        if isinstance(usage_tracking_cfg, dict):
            self.memory_usage_tracking_enabled = self._to_bool(
                usage_tracking_cfg.get("enabled", True),
                default=True,
            )
        else:
            self.memory_usage_tracking_enabled = self._to_bool(
                usage_tracking_cfg,
                default=True,
            )

        self.ideas_manager.logger = self.logger
        self.analyzer.logger = self.logger  # type: ignore[assignment]

        self._get_task_description_summary()

        self.logger.log_init(
            component="IdeaTracker",
            model_name=self.analyzer.model,
            gen_delta=self.gen_delta,
            redis_host=self.redis_config.redis_host,
            redis_port=self.redis_config.redis_port,
            redis_db=self.redis_config.redis_db,
            redis_prefix=self.redis_config.redis_prefix,
            label=self.redis_config.label,
            statistics_enabled=self.statistics_enabled,
            statistics_mode=self.statistics_mode,
            top_k_delta_fitness=self.top_k_delta_fitness,
            top_k_fitness=self.top_k_fitness,
            list_max_ideas=list_max_ideas,
            memory_write_pipeline_enabled=self.memory_write_pipeline_enabled,
            memory_write_best_programs_percent=self.memory_write_best_programs_percent,
            memory_usage_tracking_enabled=self.memory_usage_tracking_enabled,
        )

    @staticmethod
    def _to_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return default

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(parsed) or math.isinf(parsed):
            return None
        return parsed

    @staticmethod
    def _as_string_list(value: Any) -> list[str]:
        if isinstance(value, list):
            parsed = value
        elif isinstance(value, tuple):
            parsed = list(value)
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            parsed: Any = text
            if text[0] in "[{(":
                try:
                    parsed = json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    try:
                        parsed = ast.literal_eval(text)
                    except Exception:
                        parsed = text
            if not isinstance(parsed, list):
                return [str(parsed).strip()] if str(parsed).strip() else []
        else:
            return []

        out: list[str] = []
        for item in parsed:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out

    @staticmethod
    def _parse_json_like(value: Any) -> Any:
        """Parse JSON-ish strings from CSV back to Python objects when possible."""
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            return value
        if text[0] not in "[{":
            return value
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(text)
            except Exception:
                return value

    @classmethod
    def _coerce_bool_series(cls, series: pd.Series) -> pd.Series:
        """Coerce CSV-backed truthy/falsy values into a bool series."""
        if series.dtype == bool:
            return series
        normalized = series.astype(str).str.strip().str.lower()
        return normalized.map(
            {"true": True, "false": False, "1": True, "0": False}
        ).fillna(False)

    @classmethod
    def normalize_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize Redis/CSV data into the shape expected by IdeaTracker."""
        result = df.copy()

        if "is_root" in result.columns:
            result["is_root"] = cls._coerce_bool_series(result["is_root"])

        for col in ("parent_ids", "children_ids", "metadata_mutation_output"):
            if col in result.columns:
                result[col] = result[col].apply(cls._parse_json_like)

        return result

    @staticmethod
    def _median_or_none(values: list[float]) -> float | None:
        if not values:
            return None
        return float(statistics.median(values))

    @staticmethod
    def _extract_usage_task_deltas(usage: Any) -> dict[str, list[float]]:
        if not isinstance(usage, dict):
            return {}
        used = usage.get("used")
        if not isinstance(used, dict):
            return {}
        entries = used.get("entries")
        if not isinstance(entries, list):
            return {}

        task_to_deltas: dict[str, list[float]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            task_summary = str(entry.get("task_description_summary") or "").strip()
            if not task_summary:
                continue
            raw_deltas = entry.get("fitness_delta_per_use")
            if raw_deltas is None:
                raw_deltas = entry.get("fitness_deltas")
            if not isinstance(raw_deltas, list):
                continue

            parsed_deltas: list[float] = []
            for raw_delta in raw_deltas:
                delta = IdeaTracker._to_float(raw_delta)
                if delta is not None:
                    parsed_deltas.append(delta)
            if not parsed_deltas:
                continue
            task_to_deltas.setdefault(task_summary, []).extend(parsed_deltas)
        return task_to_deltas

    @staticmethod
    def _build_usage_payload_from_task_deltas(
        task_to_deltas: dict[str, list[float]],
    ) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        total_deltas: list[float] = []

        for task_summary in sorted(task_to_deltas):
            deltas = [
                parsed
                for raw in task_to_deltas.get(task_summary, [])
                if (parsed := IdeaTracker._to_float(raw)) is not None
            ]
            if not deltas:
                continue
            entries.append(
                {
                    "task_description_summary": task_summary,
                    "used_count": len(deltas),
                    "fitness_delta_per_use": deltas,
                    "median_delta_fitness": IdeaTracker._median_or_none(deltas),
                }
            )
            total_deltas.extend(deltas)

        return {
            "used": {
                "entries": entries,
                "total": {
                    "total_used": len(total_deltas),
                    "median_delta_fitness": IdeaTracker._median_or_none(total_deltas),
                },
            }
        }

    @staticmethod
    def _merge_usage_payloads(
        existing_usage: Any, incoming_usage: Any
    ) -> dict[str, Any]:
        existing_task_deltas = IdeaTracker._extract_usage_task_deltas(existing_usage)
        incoming_task_deltas = IdeaTracker._extract_usage_task_deltas(incoming_usage)
        if not existing_task_deltas and not incoming_task_deltas:
            if isinstance(existing_usage, dict):
                return dict(existing_usage)
            if isinstance(incoming_usage, dict):
                return dict(incoming_usage)
            return {}

        merged_task_deltas: dict[str, list[float]] = {
            task: list(deltas) for task, deltas in existing_task_deltas.items()
        }
        for task_summary, deltas in incoming_task_deltas.items():
            merged_task_deltas.setdefault(task_summary, []).extend(deltas)

        merged_usage: dict[str, Any] = (
            dict(existing_usage) if isinstance(existing_usage, dict) else {}
        )
        if isinstance(incoming_usage, dict):
            for key, value in incoming_usage.items():
                if key != "used":
                    merged_usage[key] = value
        merged_usage["used"] = IdeaTracker._build_usage_payload_from_task_deltas(
            merged_task_deltas
        )["used"]
        return merged_usage

    def _get_task_description_summary(self) -> str:
        cached = getattr(self, "_task_description_summary_cache", None)
        if isinstance(cached, str) and cached:
            return cached
        summary = _summarize_task_description(self.analyzer, self.task_description)
        self._task_description_summary_cache = summary
        return summary

    def _build_memory_usage_updates(
        self, programs_df: pd.DataFrame
    ) -> dict[str, dict[str, Any]]:
        required_columns = {
            "program_id",
            "metric_fitness",
            "parent_ids",
            "metadata_memory_selected_idea_ids",
        }
        if not required_columns.issubset(programs_df.columns):
            return {}

        fitness_by_program_id: dict[str, float] = {}
        for _, row in programs_df.iterrows():
            program_id = str(row.get("program_id") or "").strip()
            if not program_id:
                continue
            fitness = self._to_float(row.get("metric_fitness"))
            if fitness is not None:
                fitness_by_program_id[program_id] = fitness

        task_summary = self._get_task_description_summary()
        if not task_summary:
            task_summary = "Task summary unavailable"

        usage_by_card: dict[str, dict[str, list[float]]] = {}
        for _, row in programs_df.iterrows():
            selected_ids = self._as_string_list(
                row.get("metadata_memory_selected_idea_ids")
            )
            if not selected_ids:
                continue

            child_fitness = self._to_float(row.get("metric_fitness"))
            if child_fitness is None:
                continue

            parent_ids = self._as_string_list(row.get("parent_ids"))
            parent_fitnesses = [
                fitness_by_program_id[parent_id]
                for parent_id in parent_ids
                if parent_id in fitness_by_program_id
            ]
            if not parent_fitnesses:
                continue

            delta_fitness = child_fitness - max(parent_fitnesses)
            unique_selected_ids = list(dict.fromkeys(selected_ids))
            for card_id in unique_selected_ids:
                per_task = usage_by_card.setdefault(card_id, {})
                per_task.setdefault(task_summary, []).append(delta_fitness)

        return {
            card_id: self._build_usage_payload_from_task_deltas(task_deltas)
            for card_id, task_deltas in usage_by_card.items()
        }

    def _apply_memory_usage_updates_to_idea_banks(self) -> None:
        if not self.memory_usage_updates_by_card:
            return

        for bank in (
            self.ideas_manager.record_bank,
            self.ideas_manager.inactive_record_bank,
        ):
            for idea in bank.all_ideas_cards():
                usage_update = self.memory_usage_updates_by_card.get(str(idea.id or ""))
                if not usage_update:
                    continue
                merged_usage = self._merge_usage_payloads(
                    getattr(idea, "usage", {}),
                    usage_update,
                )
                idea.update_metadata(usage=merged_usage)

    def wrap_data(self, programs: pd.DataFrame) -> list[ProgramRecord]:
        """
        Wrap program data in ProgramRecord dataclass instances and store them.

        Args:
            programs: DataFrame containing program data with columns: program_id,
                metric_fitness, generation, parent_ids, metadata_mutation_output.

        Returns:
            List of ProgramRecord instances created from the DataFrame.
        """
        task_description_summary = self._get_task_description_summary()
        programs_processed, programs_ids = convert_programs_to_records(
            programs, self.task_description, task_description_summary
        )
        self.programs_card.extend(programs_processed)
        self.programs_ids.update(programs_ids)
        return programs_processed

    async def load_database(self) -> pd.DataFrame:
        """
        Load fresh copy of Redis database as Pandas DataFrame.

        Returns:
            DataFrame containing evolution data from Redis.
        """
        dataset = await fetch_evolution_dataframe(self.redis_config)
        return dataset

    def get_new_programs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Find programs that haven't been processed and are not discarded.

        Filters programs that:
        - Have fitness > 0
        - Are not root programs (is_root == False)
        - Have not been processed before (not in self.programs_ids)

        Args:
            df: DataFrame containing all programs from the database.

        Returns:
            DataFrame containing only new, valid programs to process.
        """
        search_condition = (df["metric_fitness"] > 0) & (df["is_root"] == False)
        valid_programs = df[search_condition]
        all_programs = set(valid_programs["program_id"])
        new_programs = all_programs.difference(self.programs_ids)
        mask = df["program_id"].isin(new_programs)
        return df[mask]

    def process_program(self, program: ProgramRecord) -> None:
        """
        Process individual program by classifying ideas from it.

        Extracts ideas from program improvements, classifies them as new or existing,
        and updates the appropriate idea banks (active or inactive).

        Args:
            program: ProgramRecord containing program data and improvements.
        """
        active_ideas = self.ideas_manager.ideas_groups_texts()
        inactive_ideas = self.ideas_manager.ideas_groups_texts(use_inactive=True)
        program_ideas = IncomingIdeas(program.improvements)
        classified_ideas = self.analyzer.process_ideas(
            program_ideas, active_ideas, inactive_ideas
        )

        for n_idea in [
            idea for idea in classified_ideas.ideas if not idea["classified"]
        ]:
            idea_description = n_idea["description"]
            change_motivation = n_idea["change_motivation"]
            self.ideas_manager.add_new_idea(
                idea_description,
                program.id,
                program.generation,
                program.category,
                program.strategy,
                program.task_description,
                change_motivation,
            )

        for idea_r in [
            idea
            for idea in classified_ideas.ideas
            if idea["classified"] and idea["rewrite"]
        ]:
            self.ideas_manager.modify_idea(
                idea_r["target_idea_id"],
                [program.id],
                program.generation,
                idea_r["description"],
                idea_r["change_motivation"],
            )

        for idea_u in [
            idea
            for idea in classified_ideas.ideas
            if idea["classified"] and not idea["rewrite"]
        ]:
            self.ideas_manager.modify_idea(
                idea_u["target_idea_id"],
                [program.id],
                program.generation,
                None,
                idea_u["change_motivation"],
            )

    def refresh_main_bank(self, current_generation: int) -> None:
        """
        Move inactive ideas from main bank to inactive ideas bank.

        Ideas are considered inactive if their last_generation is more than
        gen_delta generations away from current_generation.

        Args:
            current_generation: Current generation number to compare against.
        """
        self.ideas_manager.move_inactive(current_generation, self.gen_delta)

    def _programs_to_dicts(self) -> list[dict[str, Any]]:
        """
        Convert stored ProgramRecord instances into plain dictionaries
        suitable for JSON serialization in logs.
        """
        return [
            {
                "id": prog.id,
                "fitness": prog.fitness,
                "generation": prog.generation,
                "parents": prog.parents,
                "insights": prog.insights,
                "improvements": prog.improvements,
                "strategy": prog.strategy,
                "task_description": prog.task_description,
                "task_description_summary": prog.task_description_summary,
                "code": prog.code,
            }
            for prog in self.programs_card
        ]

    def enrich_ideas(self) -> None:
        """
        Enrich every idea in both banks with LLM-generated metadata.

        For each idea:
        1. Sends its description to the "keywords" prompt and stores the result.
        2. Sends its explanations list to the "usage_summary" prompt and stores the summary.
        3. Sends its task description to "task_description_summary" and stores a short summary.
        """
        all_uuids = list(self.ideas_manager.record_bank.uuids) + list(
            self.ideas_manager.inactive_record_bank.uuids
        )
        task_summary_cache: dict[str, str] = {}
        pbar = tqdm.tqdm(total=len(all_uuids), desc="Enriching ideas", leave=False)
        for idea_id in all_uuids:
            if idea_id in self.ideas_manager.record_bank.uuids:
                idea = self.ideas_manager.record_bank.get_idea(idea_id)
            else:
                idea = self.ideas_manager.inactive_record_bank.get_idea(idea_id)

            # --- Keywords ---
            keywords: list[str] = []
            try:
                kw_response = self.analyzer.call_llm("keywords", idea.description)
                kw_parsed = json.loads(kw_response)
                keywords = kw_parsed.get("keywords", [])
            except Exception:
                pass

            # --- Explanation summary ---
            summary = ""
            explanations = getattr(idea, "explanation", {}).get("explanations", [])
            valid_explanations = [e for e in explanations if isinstance(e, str)]
            if len(valid_explanations) == 1:
                summary = valid_explanations[0]
            elif len(valid_explanations) > 1:
                explanations_text = "\n".join(f"- {e}" for e in valid_explanations)
                try:
                    sum_response = self.analyzer.call_llm(
                        "usage_summary", explanations_text
                    )
                    sum_parsed = json.loads(sum_response)
                    summary = sum_parsed.get("summary", "")
                except Exception:
                    pass

            # --- Task description summary ---
            task_description = str(getattr(idea, "task_description", "") or "").strip()
            task_description_summary = _summarize_task_description(
                self.analyzer,
                task_description,
                cache=task_summary_cache,
            )

            self.ideas_manager.enrich_idea_metadata(
                idea_id,
                keywords=keywords,
                summary=summary,
                task_description_summary=task_description_summary,
            )
            pbar.update(1)
        pbar.close()

    def _has_best_ideas_snapshot(self, best_ideas_path: Path) -> bool:
        """
        Check that best_ideas.json contains at least one snapshot with 'best_ideas'.
        """
        try:
            with best_ideas_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False

        if isinstance(payload, dict):
            return "best_ideas" in payload
        if isinstance(payload, list):
            return any(
                isinstance(item, dict) and "best_ideas" in item for item in payload
            )
        return False

    def run_memory_write_pipeline(self) -> None:
        """
        Optionally run memory_write_example.py using current run's banks/best ideas logs.
        """
        if not self.memory_write_pipeline_enabled:
            return

        banks_path = self.logger.banks_file
        best_ideas_path = self.logger.best_ideas_file

        if banks_path is None or best_ideas_path is None:
            print("Memory write pipeline skipped: logger output paths are unavailable.")
            return
        if not banks_path.exists():
            print(f"Memory write pipeline skipped: missing banks file at {banks_path}.")
            return
        if not best_ideas_path.exists() or not self._has_best_ideas_snapshot(
            best_ideas_path
        ):
            print(
                "Memory write pipeline skipped: best_ideas snapshot was not generated for this run."
            )
            return

        # memory_write_example resolves relative paths against evo_memory_agent_api/,
        # so we must pass absolute paths here.
        env_overrides = {
            "MEMORY_BANKS_PATH": str(banks_path.resolve()),
            "MEMORY_BEST_IDEAS_PATH": str(best_ideas_path.resolve()),
        }
        programs_path = self.logger.programs_file
        if programs_path is not None and programs_path.exists():
            env_overrides["MEMORY_PROGRAMS_PATH"] = str(programs_path.resolve())
        usage_updates_path = self.logger.memory_usage_updates_file
        if (
            self.memory_usage_tracking_enabled
            and usage_updates_path is not None
            and usage_updates_path.exists()
        ):
            env_overrides["MEMORY_USAGE_UPDATES_PATH"] = str(usage_updates_path.resolve())
        previous_env = {key: os.environ.get(key) for key in env_overrides}

        try:
            os.environ.update(env_overrides)
            memory_write_module = importlib.import_module(
                "evo_memory_agent_api.memory_write_example"
            )
            memory_write_module = importlib.reload(memory_write_module)
            snapshot = memory_write_module.main()
            if isinstance(snapshot, dict):
                stats = snapshot.get("stats", {})
                stats_by_card_type = snapshot.get("stats_by_card_type", {})
                ideas_stats = (
                    stats_by_card_type.get("ideas", {})
                    if isinstance(stats_by_card_type, dict)
                    else {}
                )
                programs_stats = (
                    stats_by_card_type.get("programs", {})
                    if isinstance(stats_by_card_type, dict)
                    else {}
                )
                if isinstance(stats, dict):
                    print(
                        "Memory write pipeline stats: "
                        f"processed={stats.get('processed', 0)}, "
                        f"ideas_processed={ideas_stats.get('processed', 0)}, "
                        f"programs_processed={programs_stats.get('processed', 0)}, "
                        f"added={stats.get('added', 0)}, "
                        f"ideas_added={ideas_stats.get('added', 0)}, "
                        f"programs_added={programs_stats.get('added', 0)}, "
                        f"updated={stats.get('updated', 0)}, "
                        f"ideas_updated={ideas_stats.get('updated', 0)}, "
                        f"programs_updated={programs_stats.get('updated', 0)}, "
                        f"rejected={stats.get('rejected', 0)}"
                    )
        finally:
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def default_analyzer_pipeline(self, new_programs: list[ProgramRecord]) -> None:
        pbar = tqdm.tqdm(total=len(new_programs), leave=False)
        for prog in new_programs:
            self.process_program(prog)
            pbar.update(1)
        pbar.close()

    def fast_analyzer_pipeline(self, new_programs: list[ProgramRecord]) -> None:
        if not isinstance(self.analyzer, IdeaAnalyzerFast):
            raise TypeError(
                "fast_analyzer_pipeline requires analyzer config type 'fast' (IdeaAnalyzerFast)"
            )
        processed_programs = asyncio.run(self.analyzer.run(new_programs))
        for prog in processed_programs:
            self.ideas_manager.record_bank.import_idea_extended(prog, is_forced=True)

    def run(self, path_to_database: Optional[str | Path] = None) -> None:
        """
        Main execution method: load database, process new programs, and update banks.

        Loads programs from Redis, processes new programs to extract and classify ideas,
        moves inactive ideas to inactive bank, and dumps final state to logger.
        """
        if path_to_database is not None:
            if isinstance(path_to_database, str):
                path_to_database = Path(path_to_database)
            if path_to_database.is_file() and path_to_database.suffix == ".csv":
                df = pd.read_csv(path_to_database)
            else:
                raise ValueError(f"Invalid database file: {path_to_database}")
        else:
            df = asyncio.run(self.load_database())

        df = self.normalize_dataframe(df)
        if df.empty:
            print("Ideas tracker skipped: no programs found in the selected source.")
            return

        required_columns = {
            "program_id",
            "metric_fitness",
            "generation",
            "is_root",
            "parent_ids",
            "metadata_mutation_output",
        }
        missing_columns = sorted(required_columns.difference(df.columns))
        if missing_columns:
            raise ValueError(
                "Ideas tracker input is missing required columns: "
                + ", ".join(missing_columns)
            )

        if self.memory_usage_tracking_enabled:
            self.memory_usage_updates_by_card = self._build_memory_usage_updates(df)
            self.logger.log_memory_usage_updates(self.memory_usage_updates_by_card)
        else:
            self.memory_usage_updates_by_card = {}

        last_gen = df["generation"].max()
        new_programs = self.get_new_programs(df)
        new_programs_processed = self.wrap_data(new_programs)
        if self.analyzer_pipeline_type == "default":
            self.default_analyzer_pipeline(new_programs_processed)
        else:
            self.fast_analyzer_pipeline(new_programs_processed)
        self.refresh_main_bank(last_gen)
        if self.memory_usage_tracking_enabled:
            self._apply_memory_usage_updates_to_idea_banks()

        self.enrich_ideas()

        self.logger.dump_final_state(self.ideas_manager)
        self.logger.log_programs(self._programs_to_dicts())

        # --- Evolutionary statistics (origin analysis) ---
        compute_evolutionary_statistics(self.logger)

        # --- Optional memory DB write for best ideas ---
        self.run_memory_write_pipeline()


if __name__ == "__main__":
    p = Path(__file__).resolve().parent / "test.csv"
    itd = IdeaTracker()
    itd.run(p)
