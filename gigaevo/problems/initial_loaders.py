from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Protocol

from tqdm import tqdm

from gigaevo.database.redis_program_storage import (
    RedisProgramStorage,
    RedisProgramStorageConfig,
)
from gigaevo.programs.program import Program


class InitialProgramLoader(Protocol):
    async def load(self, storage: RedisProgramStorage) -> list[Program]: ...


class DirectoryProgramLoader:
    def __init__(self, problem_dir: str | Path):
        self.problem_dir = Path(problem_dir)

    async def load(self, storage: RedisProgramStorage) -> list[Program]:
        initial_dir = self.problem_dir / "initial_programs"
        if not initial_dir.exists():
            return []
        python_files = list(initial_dir.glob("*.py"))
        programs: list[Program] = []
        for program_file in tqdm(python_files, desc="Loading initial programs"):
            try:
                program_code = program_file.read_text()
                program = Program(code=program_code)
                program.metadata = {
                    "source": "initial_program",
                    "strategy_name": program_file.stem,
                    "file_path": str(program_file),
                    "iteration": 0,
                }
                await storage.add(program)
                programs.append(program)
            except Exception:
                continue
        return programs


class RedisTopProgramsLoader:
    def __init__(
        self,
        *,
        source_host: str,
        source_port: int,
        source_db: int,
        key_prefix: str,
        metric_key: str,
        higher_is_better: bool,
        top_n: int = 50,
        max_connections: int = 50,
        connection_pool_timeout: float = 30.0,
        health_check_interval: int = 60,
    ):
        self.source_host = source_host
        self.source_port = source_port
        self.source_db = source_db
        self.key_prefix = key_prefix
        self.metric_key = metric_key
        self.higher_is_better = higher_is_better
        self.top_n = top_n
        self.max_connections = max_connections
        self.connection_pool_timeout = connection_pool_timeout
        self.health_check_interval = health_check_interval

    async def load(self, storage: RedisProgramStorage) -> list[Program]:
        source = RedisProgramStorage(
            RedisProgramStorageConfig(
                redis_url=f"redis://{self.source_host}:{self.source_port}/{self.source_db}",
                key_prefix=self.key_prefix,
                max_connections=self.max_connections,
                connection_pool_timeout=self.connection_pool_timeout,
                health_check_interval=self.health_check_interval,
                read_only=True,
            )
        )
        try:
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
                copy = Program(code=program.code, id=program.id)
                copy.metadata = {
                    "source": "redis_selection",
                    "source_db": self.source_db,
                    "selection_rank": rank + 1,
                    "original_id": program.id,
                    "iteration": 0,
                }
                copy.metadata = {**copy.metadata, **program.metadata}
                copy.metrics = deepcopy(program.metrics)
                copy.stage_results = deepcopy(program.stage_results)
                for child in program.lineage.children:
                    if child.id in all_ids:
                        copy.lineage.children.append(child)
                for parent in program.lineage.parents:
                    if parent.id in all_ids:
                        copy.lineage.parents.append(parent)
                await storage.add(copy)
                added.append(copy)
            return added
        finally:
            try:
                await source.close()
            except Exception:
                pass
