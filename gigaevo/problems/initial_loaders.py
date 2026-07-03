from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Protocol

from tqdm import tqdm

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.programs.program import Program


class InitialProgramLoader(Protocol):
    async def load(self, storage: ProgramStorage) -> list[Program]: ...


class DirectoryProgramLoader:
    def __init__(self, problem_dir: str | Path, pattern: str = "*.py"):
        self.problem_dir = Path(problem_dir)
        self.pattern = pattern

    async def load(self, storage: ProgramStorage) -> list[Program]:
        initial_dir = self.problem_dir / "initial_programs"
        if not initial_dir.exists():
            return []
        python_files = list(initial_dir.glob(self.pattern))
        programs: list[Program] = []
        for program_file in tqdm(python_files, desc="Loading initial programs"):
            try:
                program_code = program_file.read_text()
                program = Program(code=program_code, iteration=len(programs))
                program.metadata = {
                    "source": "initial_program",
                    "strategy_name": program_file.stem,
                    "file_path": str(program_file),
                }
                await storage.add(program)
                programs.append(program)
            except Exception:
                continue
        return programs


class TopProgramsLoader:
    """Seed a run with the top-N programs of another run's storage."""

    def __init__(
        self,
        *,
        source: ProgramStorage,
        metric_key: str,
        higher_is_better: bool,
        top_n: int = 50,
    ):
        self.source = source
        self.metric_key = metric_key
        self.higher_is_better = higher_is_better
        self.top_n = top_n

    async def load(self, storage: ProgramStorage) -> list[Program]:
        async with self.source as source:
            all_programs = await source.get_all()
            if not all_programs:
                return []
            sentinel = -float("inf") if self.higher_is_better else float("inf")
            programs_with_metric = [
                p for p in all_programs if p.metrics and self.metric_key in p.metrics
            ]
            programs_with_metric.sort(
                key=lambda p: p.metrics.get(self.metric_key, sentinel),
                reverse=self.higher_is_better,
            )
            selected = programs_with_metric[: self.top_n]

            added: list[Program] = []
            all_ids = set(selected.id for selected in selected)
            for rank, program in enumerate(
                tqdm(selected, desc="Loading selected programs")
            ):
                copy = Program(code=program.code, id=program.id, iteration=rank)
                copy.metadata = {
                    "source": "top_programs",
                    "source_prefix": source.key_prefix,
                    "selection_rank": rank + 1,
                    "original_id": program.id,
                }
                copy.metadata = {**copy.metadata, **program.metadata}
                copy.metrics = deepcopy(program.metrics)
                copy.stage_results = deepcopy(program.stage_results)
                for child in program.lineage.children:
                    if child in all_ids:
                        copy.lineage.children.append(child)
                for parent in program.lineage.parents:
                    if parent in all_ids:
                        copy.lineage.parents.append(parent)
                await storage.add(copy)
                added.append(copy)
            return added
