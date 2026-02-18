import asyncio
import json
import os
from pathlib import Path
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
from gigaevo.llm.ideas_tracker.utils.selected_ideas_6 import compute_origin_analysis
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
                default config/ideas_tracker.yaml from project root.
        """
        self.config = self._load_config(config_path)
        self.logger = IdeasTrackerLogger(Path(__file__).resolve())

        list_max_ideas = int(self.config.get("list_max_ideas", 5))
        self.ideas_manager = RecordManager(list_max_ideas=list_max_ideas)

        model_name = self.config.get("model") or "deepseek/deepseek-v3.2"
        reasoning_cfg = self.config.get("reasoning", {}) or {}
        self.analyzer = IdeaAnalyzer(model_name, reasoning=reasoning_cfg)
        self.programs_card: list[ProgramRecord] = []
        self.programs_ids: set[str] = set()

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
        )

    def _load_config(self, config_path: Optional[str | Path]) -> dict[str, Any]:
        """
        Load IdeaTracker configuration from YAML file.

        Resolves project root (3 levels up from this file) to find default config
        at config/ideas_tracker.yaml when no path is provided.

        Args:
            config_path: Path to configuration file, or None to use default location.

        Returns:
            Dictionary containing configuration values with keys: gen_delta, model, redis.
            Returns default configuration if file is missing.
        """
        if config_path is None:
            # Resolve project root from this file's location:
            # .../gigaevo-core-internal/gigaevo/llm/ideas_tracker/ideas_tracker.py
            project_root = Path(__file__).resolve().parents[3]
            path_obj = project_root / "config" / "ideas_tracker.yaml"
        else:
            path_obj = Path(config_path)

        if not path_obj.is_file():
            # Fallback to defaults if config file is missing
            return {
                "gen_delta": 100000,
                "list_max_ideas": 5,
                "model": "deepseek/deepseek-v3.2",
                "redis": {
                    "redis_host": "localhost",
                    "redis_port": 6379,
                    "redis_db": 0,
                    "redis_prefix": "heilbron",
                    "label": "",
                },
            }

        with path_obj.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data

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

        # Resolve project root from this file's location:
        # .../gigaevo-core-internal/gigaevo/llm/ideas_tracker/ideas_tracker.py
        project_root = Path(__file__).resolve().parents[3]
        problems_root = project_root / "problems"

        try:
            # Walk the problems tree and collect all leaf directories.
            leaf_dirs: list[Path] = []
            for root, dirs, _files in os.walk(problems_root):
                if "initial_programs" in dirs:
                    leaf_dirs.append(Path(root))

            # Find the first leaf directory whose *name* matches redis_prefix.
            for leaf in leaf_dirs:
                split_index = leaf.parts.index("problems") + 1
                true_name = "_".join(leaf.parts[split_index:])
                if true_name == prefix_value:
                    candidate_file = leaf / "task_description.txt"
                    if candidate_file.is_file():
                        return candidate_file.read_text(encoding="utf-8").strip()
        except Exception:
            # Best-effort: if anything goes wrong, fall back to default text.
            return "No description available"

        return "No description available"

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
            new_program = ProgramRecord(
                id=program["program_id"],
                fitness=program["metric_fitness"],
                generation=program["generation"],
                parents=program["parent_ids"],
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

        # Add all truly new ideas into the main (active) ideas bank.
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

        df_summary, df_best_ideas = compute_origin_analysis(
            banks_path=str(banks_path),
            programs_path=str(programs_path),
        )

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

        last_gen = df["generation"].max()
        new_programs = self.get_new_programs(df)
        new_programs_processed = self.wrap_data(new_programs)
        pbar = tqdm.tqdm(total=len(new_programs), leave=False)
        for prog in new_programs_processed:
            self.process_program(prog)
            pbar.update(1)
        pbar.close()
        self.refresh_main_bank(last_gen)

        self.logger.dump_final_state(self.ideas_manager)
        self.logger.log_programs(self._programs_to_dicts())

        self.get_rankings()
        # self.logger.log_best_ideas(self.top_ideas())  # deprecated: handled by compute_evolutionary_statistics

        # --- Evolutionary statistics (origin analysis) ---
        self.compute_evolutionary_statistics()


if __name__ == "__main__":
    itd = IdeaTracker()
    itd.run()
