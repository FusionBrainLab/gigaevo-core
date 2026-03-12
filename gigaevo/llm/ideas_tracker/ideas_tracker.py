import asyncio
import ast
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
import yaml

sys.path.append("../gigaevo-core-internal")
from gigaevo.llm.ideas_tracker.components.analyzer import IdeaAnalyzer
from gigaevo.llm.ideas_tracker.components.data_components import (
    IncomingIdeas,
    ProgramRecord,
)
from gigaevo.llm.ideas_tracker.components.records_manager import RecordManager
from gigaevo.llm.ideas_tracker.utils.ideas_stats import (
    top_delta_ideas,
    top_fitness_ideas,
)
from gigaevo.llm.ideas_tracker.utils.impact_metrics import avg_score, delta_impact
from gigaevo.llm.ideas_tracker.utils.it_logger import IdeasTrackerLogger
from gigaevo.llm.ideas_tracker.utils.selected_ideas_6 import compute_origin_analysis
from tools.utils import RedisRunConfig, fetch_evolution_dataframe


class IdeaTracker:
    """
    Main class for tracking and analyzing ideas from evolutionary program runs.

    Manages idea banks (active and inactive), processes programs to extract ideas,
    and maintains rankings of ideas based on their impact on program fitness.
    """

    def __init__(self, config_path: Optional[str | Path] = None) -> None:
        """
        Initialize IdeaTracker with configuration from YAML file.

        Sets up idea management, LLM analyzer, Redis connection, logging, and
        statistics tracking. Initializes both active and inactive idea banks.

        Args:
            config_path: Optional path to configuration YAML file. If None, uses
                default config/memory.yaml from project root (ideas_tracker section).
        """
        self.config = self._load_config(config_path)
        self.logger = IdeasTrackerLogger(Path(__file__).resolve())

        list_max_ideas = int(self.config.get("list_max_ideas", 5))
        self.ideas_manager = RecordManager(list_max_ideas=list_max_ideas)

        model_name = self.config.get("model") or "deepseek/deepseek-v3.2"
        base_url = self.config.get("base_url")
        reasoning_cfg = self.config.get("reasoning", {}) or {}
        self.analyzer = IdeaAnalyzer(
            model_name,
            reasoning=reasoning_cfg,
            base_url=base_url,
        )
        self.programs_card: list[ProgramRecord] = []
        self.programs_ids: set[str] = set()
        self.memory_usage_updates_by_card: dict[str, dict[str, Any]] = {}

        self.gen_delta: int = int(self.config.get("gen_delta", 100000))

        redis_cfg = self.config.get("redis", {}) or {}
        self.redis_config = RedisRunConfig(
            redis_host=redis_cfg.get("redis_host", "localhost"),
            redis_port=int(redis_cfg.get("redis_port", 6379)),
            redis_db=int(redis_cfg.get("redis_db", 0)),
            redis_prefix=redis_cfg.get("redis_prefix", ""),
            label=redis_cfg.get("label", ""),
        )

        self.task_description: str = self._load_task_description()

        self.description_rewriting = self.config.get("description_rewriting", False)
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
        else:
            self.memory_write_pipeline_enabled = self._to_bool(
                memory_write_cfg,
                default=False,
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

        self.logger.log_init(
            component="IdeaTracker",
            model_name=model_name,
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
            memory_usage_tracking_enabled=self.memory_usage_tracking_enabled,
        )

    def _load_config(self, config_path: Optional[str | Path]) -> dict[str, Any]:
        """
        Load IdeaTracker configuration from YAML file.

        Resolves project root (3 levels up from this file) to find default config
        at config/memory.yaml when no path is provided.
        If the loaded file contains an ``ideas_tracker`` section, that section is used.
        Otherwise, the full file payload is treated as legacy IdeaTracker config.

        Args:
            config_path: Path to configuration file, or None to use default location.

        Returns:
            Dictionary containing configuration values with keys: gen_delta, model, redis.
            Returns default configuration if file is missing.
        """
        default_config: dict[str, Any] = {
            "gen_delta": 100000,
            "list_max_ideas": 5,
            "model": "deepseek/deepseek-v3.2",
            "base_url": "https://openrouter.ai/api/v1",
            "redis": {
                "redis_host": "localhost",
                "redis_port": 6379,
                "redis_db": 0,
                "redis_prefix": "heilbron",
                "label": "",
            },
            "memory_write_pipeline": {"enabled": False},
            "usage_tracking": {"enabled": True},
        }

        if config_path is None:
            project_root = Path(__file__).resolve().parents[3]
            path_obj = project_root / "config" / "memory.yaml"
        else:
            path_obj = Path(config_path)

        if not path_obj.is_file():
            return default_config

        with path_obj.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}

        if not isinstance(payload, dict):
            return default_config

        # Unified config format stores tracker settings under ideas_tracker.
        ideas_tracker_cfg = payload.get("ideas_tracker")
        if isinstance(ideas_tracker_cfg, dict):
            return ideas_tracker_cfg

        # Backward-compatible fallback: treat the whole payload as tracker config.
        return payload

    def _load_task_description(self) -> str:
        """
        Load human-readable task description for the current experiment.

        Searches problems/ directory tree for a leaf directory matching redis_prefix
        and loads task_description.txt from it. Returns placeholder if not found.

        Returns:
            Task description text from matching directory, or "No description available".
        """
        prefix_value = getattr(self.redis_config, "redis_prefix", "") or ""
        if not prefix_value:
            return "No description available"
        prefix_value = prefix_value.replace("/", "_")

        project_root = Path(__file__).resolve().parents[3]
        problems_root = project_root / "problems"

        try:
            # Walk the problems tree and collect all leaf directories.
            leaf_dirs: list[Path] = []
            for root, dirs, _files in os.walk(problems_root):
                if "initial_programs" in dirs:
                    leaf_dirs.append(Path(root))

            for leaf in leaf_dirs:
                split_index = leaf.parts.index("problems") + 1
                true_name = "_".join(leaf.parts[split_index:])
                if true_name == prefix_value:
                    candidate_file = leaf / "task_description.txt"
                    if candidate_file.is_file():
                        return candidate_file.read_text(encoding="utf-8").strip()
        except Exception:
            return "No description available"

        return "No description available"

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
    def _merge_usage_payloads(existing_usage: Any, incoming_usage: Any) -> dict[str, Any]:
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

    def _summarize_task_description(
        self, task_description: str, cache: dict[str, str] | None = None
    ) -> str:
        task_text = str(task_description or "").strip()
        if not task_text:
            return ""
        if cache is not None and task_text in cache:
            return cache[task_text]

        summary = ""
        try:
            task_sum_response = self.analyzer.call_llm("task_description_summary", task_text)
            task_sum_parsed = json.loads(task_sum_response)
            summary = str(task_sum_parsed.get("summary", "")).strip()
        except Exception:
            summary = ""
        if not summary:
            summary = task_text[:240].strip()

        if cache is not None:
            cache[task_text] = summary
        return summary

    def _build_memory_usage_updates(self, programs_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
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

        task_summary = self._summarize_task_description(self.task_description)
        if not task_summary:
            task_summary = "Task summary unavailable"

        usage_by_card: dict[str, dict[str, list[float]]] = {}
        for _, row in programs_df.iterrows():
            selected_ids = self._as_string_list(row.get("metadata_memory_selected_idea_ids"))
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
        programs_processed = []

        for _, program in programs.iterrows():
            mutation_metadata = program["metadata_mutation_output"]
            if isinstance(mutation_metadata, str):
                mutation_metadata = json.loads(mutation_metadata)
            parent_ids = program["parent_ids"]
            if isinstance(parent_ids, str):
                try:
                    parent_ids = json.loads(parent_ids)
                except (json.JSONDecodeError, TypeError):
                    parent_ids = []
            new_program = ProgramRecord(
                id=program["program_id"],
                fitness=program["metric_fitness"],
                generation=program["generation"],
                parents=parent_ids,
                insights=mutation_metadata["insights_used"],
                improvements=mutation_metadata["changes"],
                category="",
                strategy=mutation_metadata["archetype"],
                task_description=self.task_description,
            )
            programs_processed.append(new_program)
            self.programs_ids.add(program["program_id"])
        self.programs_card.extend(programs_processed)
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

    def get_rankings(self) -> list[dict[str, Any]]:
        """
        Get ideas ranking from main bank with fitness statistics.

        Calculates average fitness for programs with and without each idea,
        and computes impact as the difference between these averages.

        Returns:
            List of dictionaries containing idea rankings with keys:
            - id: Idea UUID
            - description: Idea description
            - programs: List of program IDs using this idea
            - fitness: List of fitness values for programs with this idea
            - an_fitness: List of fitness values for programs without this idea
            - avg_score_with: Average fitness with this idea
            - avg_score_without: Average fitness without this idea
            - impact: Difference between avg_score_with and avg_score_without
        """
        programs_rk_stats = self.ideas_manager.get_rankings()
        for program in self.programs_card:
            for index, idea in enumerate(programs_rk_stats):
                if program.id in idea["programs"]:
                    programs_rk_stats[index]["fitness"].append(program.fitness)
                else:
                    programs_rk_stats[index]["an_fitness"].append(program.fitness)
        for index, idea in enumerate(programs_rk_stats):
            programs_rk_stats[index]["avg_score_with"] = avg_score(idea["fitness"])
            programs_rk_stats[index]["avg_score_without"] = avg_score(
                idea["an_fitness"]
            )
            programs_rk_stats[index]["impact"] = delta_impact(
                idea["fitness"], idea["an_fitness"]
            )

        self.logger.log_rankings(programs_rk_stats)
        return programs_rk_stats

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
            }
            for prog in self.programs_card
        ]

    def top_ideas(self) -> dict[str, Any]:
        """
        Calculate top ideas based on configured statistics mode.

        Returns:
            Dictionary containing top ideas according to statistics_mode
            (top_k, top_fitness, or delta_fitness). Empty dict if statistics disabled.
        """
        if not self.statistics_enabled:
            return {}
        active_ideas_card = [
            idea for idea in self.ideas_manager.record_bank.all_ideas_cards()
        ]
        inactive_ideas_card = [
            idea for idea in self.ideas_manager.inactive_record_bank.all_ideas_cards()
        ]
        all_ideas_card = active_ideas_card + inactive_ideas_card
        if self.statistics_mode == "top_k":
            statistics = {
                "top_fitness_ideas": top_fitness_ideas(
                    self.programs_card, all_ideas_card, self.top_k_fitness
                ),
                "top_delta_ideas": top_delta_ideas(
                    self.programs_card, all_ideas_card, self.top_k_delta_fitness
                ),
            }
        elif self.statistics_mode == "top_fitness":
            statistics = {
                "top_fitness_ideas": top_fitness_ideas(
                    self.programs_card, all_ideas_card, self.top_k_fitness
                )
            }
        elif self.statistics_mode == "delta_fitness":
            statistics = {
                "top_delta_ideas": top_delta_ideas(
                    self.programs_card, all_ideas_card, self.top_k_delta_fitness
                )
            }
        else:
            raise ValueError(f"Invalid statistics mode: {self.statistics_mode}")
        return statistics

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
            task_description_summary = self._summarize_task_description(
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

    def compute_evolutionary_statistics(self) -> None:
        """
        Run origin-based evolutionary statistics on saved banks/programs JSONs
        and inject per-idea metrics into banks.json under 'evolution_statistics'.

        Requires that dump_final_state and log_programs have already been called
        so that banks.json and programs.json exist in the session directory.
        """
        banks_path = self.logger.banks_file
        programs_path = self.logger.programs_file

        if banks_path is None or programs_path is None:
            return
        if not banks_path.exists() or not programs_path.exists():
            return

        try:
            df_summary, df_best_ideas = compute_origin_analysis(
                banks_path=str(banks_path),
                programs_path=str(programs_path),
            )
        except RuntimeError as exc:
            if str(exc) == "No valid programs with numeric generation and fitness found.":
                return
            raise

        if df_summary.empty:
            return

        best_ideas_records = df_best_ideas.to_dict(orient="records")
        sanitized: list[dict[str, Any]] = []
        for rec in best_ideas_records:
            sanitized.append({k: (v if pd.notna(v) else None) for k, v in rec.items()})
        self.logger.log_best_ideas({"best_ideas": sanitized})

        stats_by_idea: dict[str, dict[str, Any]] = {}
        for _, row in df_summary.iterrows():
            idea_id = row["idea_id"]
            quartile = row["quartile"]
            metrics = row.drop(["idea_id", "quartile", "description"]).to_dict()
            metrics = {k: (v if pd.notna(v) else None) for k, v in metrics.items()}
            if idea_id not in stats_by_idea:
                stats_by_idea[idea_id] = {}
            stats_by_idea[idea_id][quartile] = metrics

        with open(banks_path, "r", encoding="utf-8") as f:
            banks_data = json.load(f)

        for snapshot in banks_data:
            if not isinstance(snapshot, dict):
                continue
            for bank_key in ("active_bank", "inactive_bank"):
                bank = snapshot.get(bank_key, [])
                if not isinstance(bank, list):
                    continue
                for idea in bank:
                    if not isinstance(idea, dict):
                        continue
                    idea_id = idea.get("id", "")
                    if idea_id in stats_by_idea:
                        idea["evolution_statistics"] = stats_by_idea[idea_id]

        with open(banks_path, "w", encoding="utf-8") as f:
            json.dump(banks_data, f, indent=4)

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

        env_overrides = {
            "MEMORY_BANKS_PATH": str(banks_path),
            "MEMORY_BEST_IDEAS_PATH": str(best_ideas_path),
        }
        usage_updates_path = self.logger.memory_usage_updates_file
        if (
            self.memory_usage_tracking_enabled
            and usage_updates_path is not None
            and usage_updates_path.exists()
        ):
            env_overrides["MEMORY_USAGE_UPDATES_PATH"] = str(usage_updates_path)
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
                if isinstance(stats, dict):
                    print(
                        "Memory write pipeline stats: "
                        f"processed={stats.get('processed', 0)}, "
                        f"added={stats.get('added', 0)}, "
                        f"updated={stats.get('updated', 0)}, "
                        f"rejected={stats.get('rejected', 0)}"
                    )
        finally:
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

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

        if self.memory_usage_tracking_enabled:
            self.memory_usage_updates_by_card = self._build_memory_usage_updates(df)
            self.logger.log_memory_usage_updates(self.memory_usage_updates_by_card)
        else:
            self.memory_usage_updates_by_card = {}

        last_gen = df["generation"].max()
        new_programs = self.get_new_programs(df)
        new_programs_processed = self.wrap_data(new_programs)
        pbar = tqdm.tqdm(total=len(new_programs), leave=False)
        for prog in new_programs_processed:
            self.process_program(prog)
            pbar.update(1)
        pbar.close()
        self.refresh_main_bank(last_gen)
        if self.memory_usage_tracking_enabled:
            self._apply_memory_usage_updates_to_idea_banks()

        self.enrich_ideas()

        self.logger.dump_final_state(self.ideas_manager)
        self.logger.log_programs(self._programs_to_dicts())

        self.get_rankings()
        # self.logger.log_best_ideas(self.top_ideas())  # deprecated: handled by compute_evolutionary_statistics

        # --- Evolutionary statistics (origin analysis) ---
        self.compute_evolutionary_statistics()

        # --- Optional memory DB write for best ideas ---
        self.run_memory_write_pipeline()


if __name__ == "__main__":
    p = Path(__file__).resolve().parent / "heilbron_gemini_flash.csv"
    itd = IdeaTracker()
    itd.run()
