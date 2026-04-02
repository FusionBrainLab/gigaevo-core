from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger
import pandas as pd

from gigaevo.database.redis_program_storage import (
    RedisProgramStorage,
    RedisProgramStorageConfig,
)
from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS


@dataclass
class RedisRunConfig:
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_prefix: str = ""
    label: str = ""

    def url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    def display_label(self) -> str:
        return self.label or f"{self.redis_prefix}@{self.redis_db}"


async def fetch_evolution_dataframe(
    config: RedisRunConfig, add_stage_results: bool = False
) -> pd.DataFrame:
    storage = RedisProgramStorage(
        RedisProgramStorageConfig(
            redis_url=config.url(),  # type: ignore[arg-type]  # pydantic validates str -> AnyUrl
            key_prefix=config.redis_prefix,
            max_connections=50,
            connection_pool_timeout=30.0,
            health_check_interval=60,
            read_only=True,
        )
    )

    try:
        exclude = None if add_stage_results else EXCLUDE_STAGE_RESULTS
        programs = await storage.get_all(exclude=exclude)
    finally:
        await storage.close()

    if not programs:
        logger.warning(
            f"No programs found for prefix='{config.redis_prefix}' at {config.url()}"
        )
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for program in programs:
        row: dict[str, Any] = {
            "program_id": program.id,
            "name": program.name or "unnamed",
            "code": program.code,
            "created_at": program.created_at,
            "atomic_counter": program.atomic_counter,
            "state": program.state.value,
            "is_complete": program.is_complete,
            "generation": program.generation,
            "is_root": program.is_root,
            "parent_ids": (program.lineage.parents),
            "children_ids": (program.lineage.children),
        }
        if add_stage_results:
            row["stage_results"] = program.stage_results
        metrics = program.metrics
        for mname, mval in metrics.items():
            row[f"metric_{mname}"] = mval

        lineage = program.lineage
        row["lineage_num_parents"] = len(lineage.parents)
        row["lineage_num_children"] = len(lineage.children)
        row["lineage_mutation"] = lineage.mutation
        row["lineage_generation"] = lineage.generation

        metadata = program.metadata
        for k, v in metadata.items():
            row[f"metadata_{k}"] = v

        rows.append(row)

    df = pd.DataFrame(rows)
    for col in ["created_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df
