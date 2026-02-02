import asyncio
from pathlib import Path
import sys

import pandas as pd
import tqdm
import yaml

sys.path.append("../gigaevo-core-internal")
from gigaevo.llm.ideas_tracker.components.analyzer import IdeaAnalyzer
from gigaevo.llm.ideas_tracker.components.data_components import ProgramRecord
from gigaevo.llm.ideas_tracker.components.records_manager import RecordManager
from tools.utils import RedisRunConfig, fetch_evolution_dataframe


class IdeaTracker:
    def __init__(self, config_path=None):
        # Load configuration from YAML
        self.config = self._load_config(config_path)

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

    def _load_config(self, config_path) -> dict:
        """
        Load IdeaTracker configuration from YAML file located in the root config folder.
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
        Wraps program data in dataclass and store it
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

    async def load_database(self):
        """
        Load fresh copy of Redis database as Pandas DataFrame.
        """
        dataset = await fetch_evolution_dataframe(self.redis_config)
        return dataset

    def get_new_programs(self, df: pd.DataFrame):
        """
        Find program that have't been processed and not discarded
        """
        search_condition = (df["metric_fitness"] > 0) & (df["is_root"] == False)
        valid_programs = df[search_condition]
        all_programs = set(valid_programs["program_id"])
        new_programs = all_programs.difference(self.programs_ids)
        mask = df["program_id"].isin(new_programs)
        return df[mask]

    def process_program(self, program: ProgramRecord):
        """
        Process individual programm by classifying ideas from it.
        Parsed ideas loaded into corresponding banks
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

    def refresh_main_bank(self, current_generation: int):
        """
        Move inactive ideas from main bank to inactive ideas bank
        """
        self.ideas_manager.move_inactive(current_generation, self.gen_delta)

    def get_rankings(self):
        """
        Receive ideas ranking from main bank
        """
        return self.ideas_manager.get_rankings()

    def run(self):
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


if __name__ == "__main__":
    import json

    p = Path(__file__).resolve().parent / "rankings.json"
    itd = IdeaTracker()
    itd.run()
    rankings = itd.get_rankings()
    rankings["scores"] = list(rankings["scores"])
    with open(p, "w") as f:
        json.dump(rankings, f, indent=4)
