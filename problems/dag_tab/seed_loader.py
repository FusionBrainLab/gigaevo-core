from __future__ import annotations

from pathlib import Path

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.programs.program import Program
from problems.tabular._common.tabular_data import load_dataset

from .graph import FeatureGraph


class DagTabSeedLoader:
    """Create a raw-feature baseline for the selected tabular dataset."""

    def __init__(
        self, dataset: str = "california", problem_dir: str | Path | None = None
    ):
        self.dataset = dataset
        self.problem_dir = Path(problem_dir) if problem_dir is not None else None

    async def load(self, storage: ProgramStorage) -> list[Program]:
        dataset = load_dataset(self.dataset)
        graph = FeatureGraph(
            dataset=self.dataset,
            raw_columns=[f"x{index}" for index in range(dataset.X_train.shape[1])],
            nodes=[],
        )
        program = Program(code=graph.to_json(), iteration=0)
        program.metadata = {
            "source": "initial_program",
            "strategy_name": "raw_baseline",
            "dataset": self.dataset,
        }
        await storage.add(program)
        return [program]
