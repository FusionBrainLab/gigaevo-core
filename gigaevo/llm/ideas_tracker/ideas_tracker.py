import asyncio
from pathlib import Path
import sys
from typing import Any, Optional

import pandas as pd
import tqdm
import yaml

sys.path.append("../gigaevo-core-internal")
from gigaevo.llm.ideas_tracker.components.analyzer import IdeaAnalyzer
from gigaevo.llm.ideas_tracker.components.data_components import ProgramRecord
from gigaevo.llm.ideas_tracker.components.records_manager import RecordManager
from gigaevo.llm.ideas_tracker.utils.it_logger import IdeasTrackerLogger
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

        Args:
            config_path: Optional path to configuration YAML file. If None, uses
                default config/ideas_tracker.yaml from project root.
        """
        # Load configuration from YAML
        self.config = self._load_config(config_path)

        # Initialize logger (logs directory next to this file)
        self.logger = IdeasTrackerLogger(Path(__file__).resolve())

        self.ideas_manager = RecordManager()
        # Initialize analyzer with model from config (fallback to default)
        model_name = self.config.get("model") or "deepseek/deepseek-v3.2"
        self.analyzer = IdeaAnalyzer(model_name)
        self.programs_card: list[ProgramRecord] = []
        self.programs_ids: set[str] = set()

        # Generation delta for moving inactive ideas
        self.gen_delta: int = int(self.config.get("gen_delta", 100000))

        # Redis connection configuration
        redis_cfg = self.config.get("redis", {}) or {}
        self.redis_config = RedisRunConfig(
            redis_host=redis_cfg.get("redis_host", "localhost"),
            redis_port=int(redis_cfg.get("redis_port", 6379)),
            redis_db=int(redis_cfg.get("redis_db", 0)),
            redis_prefix=redis_cfg.get("redis_prefix", ""),
            label=redis_cfg.get("label", ""),
        )

        # Attach logger to components
        self.ideas_manager.logger = self.logger
        self.analyzer.logger = self.logger  # type: ignore[assignment]

        # Log init parameters
        self.logger.log_init(
            component="IdeaTracker",
            model_name=model_name,
            gen_delta=self.gen_delta,
            redis_host=self.redis_config.redis_host,
            redis_port=self.redis_config.redis_port,
            redis_db=self.redis_config.redis_db,
            redis_prefix=self.redis_config.redis_prefix,
            label=self.redis_config.label,
        )

    def _load_config(self, config_path: Optional[str | Path]) -> dict[str, Any]:
        """
        Load IdeaTracker configuration from YAML file located in the root config folder.

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
            new_program = ProgramRecord(
                id=program["program_id"],
                fitness=program["metric_fitness"],
                generation=program["generation"],
                parents=program["parent_ids"],
                insights=mutation_metadata["insights_used"],
                improvements=mutation_metadata["changes"],
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
        program_ideas = [pg_idea["description"] for pg_idea in program.improvements]
        new_ideas, existing_ideas_short_ids = self.analyzer.process_ideas(
            program_ideas, active_ideas, inactive_ideas
        )

        # Add all truly new ideas into the main (active) ideas bank.
        for n_idea in new_ideas:
            self.ideas_manager.add_new_idea(n_idea, program.id, program.generation)

        # For ideas that are already known, resolve their full UUID and update them.
        for short_id in existing_ideas_short_ids:
            # Try to find the idea first in the active bank, then in the inactive bank.
            full_id = self.ideas_manager.get_full_id(short_id, active_ideas)
            if not full_id:
                full_id = self.ideas_manager.get_full_id(short_id, inactive_ideas)
            if not full_id:
                # If we cannot resolve the id from either bank snapshot, skip updating.
                continue
            self.ideas_manager.modify_idea(full_id, [program.id], program.generation)

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
            programs_rk_stats[index]["avg_score_with"] = sum(idea["fitness"]) / len(
                idea["fitness"]
            )
            programs_rk_stats[index]["avg_score_without"] = sum(
                idea["an_fitness"]
            ) / len(idea["an_fitness"])
            programs_rk_stats[index]["impact"] = sum(idea["fitness"]) / len(
                idea["fitness"]
            ) - sum(idea["an_fitness"]) / len(idea["an_fitness"])

        # Persist rankings snapshot via logger
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

    def run(self) -> None:
        """
        Main execution method: load database, process new programs, and update banks.

        Loads programs from Redis, processes new programs to extract and classify ideas,
        moves inactive ideas to inactive bank, and dumps final state to logger.
        """
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

        # After processing and refreshing banks, dump final state of banks
        self.logger.dump_final_state(self.ideas_manager)

        # Dump programs data as JSON snapshot similar to banks
        self.logger.log_programs(self._programs_to_dicts())

        self.get_rankings()


if __name__ == "__main__":
    itd = IdeaTracker()
    itd.run()
