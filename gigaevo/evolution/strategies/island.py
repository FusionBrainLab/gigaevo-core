from __future__ import annotations

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.evolution.storage.archive_storage import RedisArchiveStorage
from gigaevo.evolution.strategies.elite_selectors import EliteSelector
from gigaevo.evolution.strategies.metadata_manager import MetadataManager
from gigaevo.evolution.strategies.migrant_selectors import MigrantSelector
from gigaevo.evolution.strategies.models import BehaviorSpace
from gigaevo.evolution.strategies.removers import ArchiveRemover
from gigaevo.evolution.strategies.selectors import ArchiveSelector
from gigaevo.programs.program import Program


class IslandConfig(BaseModel):
    """Configuration for an individual MAP-Elites island."""

    island_id: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Unique identifier for the island",
    )
    max_size: int | None = Field(
        default=None,
        ge=1,
        description="Max programs in the archive; excess entries are removed.",
    )
    behavior_space: BehaviorSpace
    archive_selector: ArchiveSelector = Field(
        description="Comparator used by archive to decide if newcomer is better"
    )
    archive_remover: ArchiveRemover | None = Field(
        description="Policy to remove programs when archive exceeds max_size"
    )
    elite_selector: EliteSelector = Field(
        description="Selector for choosing elites to mutate"
    )
    migrant_selector: MigrantSelector = Field(
        description="Selector for choosing migrants"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field
    @property
    def redis_prefix(self) -> str:
        return f"island_{self.island_id}"

    @field_validator("archive_remover")
    @classmethod
    def _validate_archive_remover(cls, v, info):
        if info.data.get("max_size") is not None and v is None:
            raise ValueError("`max_size` is set but `archive_remover` is None")
        return v


class MapElitesIsland:
    """Single MAP-Elites island."""

    def __init__(self, config: IslandConfig, program_storage: RedisProgramStorage):
        self.config = config
        self.program_storage = program_storage
        self.archive_storage = RedisArchiveStorage(
            program_storage=program_storage, key_prefix=config.redis_prefix
        )
        self.metadata_manager = MetadataManager(program_storage)
        logger.info("Island {} init (max_size={})", config.island_id, config.max_size)

    # -------------------------- Public API --------------------------

    async def add(self, program: Program) -> bool:
        """Insert `program` into its behavior cell if it improves the elite."""
        missing = set(self.config.behavior_space.behavior_keys) - program.metrics.keys()
        if missing:
            raise KeyError(f"Program missing required behavior keys: {missing}")

        cell = self.config.behavior_space.get_cell(program.metrics)
        improved = await self.archive_storage.add_elite(
            cell, program, self.config.archive_selector
        )
        if not improved:
            return False

        await self.metadata_manager.set_current_island(program, self.config.island_id)
        await self._enforce_size_limit()
        logger.debug("Island {}: added {}", self.config.island_id, program.id)
        return True

    async def select_elites(self, total: int) -> list[Program]:
        """Return up to `total` elite programs for mutation."""
        elites = await self.get_elites()
        if not elites or total <= 0:
            return []
        if len(elites) <= total:
            logger.debug(
                "Island {}: {} elites ≤ requested {}",
                self.config.island_id,
                len(elites),
                total,
            )
            return elites
        selected = self.config.elite_selector(elites, total)
        logger.debug(
            "Island {}: selected {} / {}",
            self.config.island_id,
            len(selected),
            len(elites),
        )
        return selected

    async def select_migrants(self, count: int) -> list[Program]:
        """Select programs to emigrate to other islands."""
        elites = await self.get_elites()
        return [] if not elites else self.config.migrant_selector(elites, count)

    async def get_elite_ids(self) -> list[str]:
        """IDs of all elites in this island (source of truth is the archive)."""
        return await self.archive_storage.get_all_elites()  # list[str]

    async def get_elites(self) -> list[Program]:
        """Materialized elite programs (filters out missing)."""
        ids = await self.get_elite_ids()
        if not ids:
            return []
        programs = await self.program_storage.mget(ids)
        return [p for p in programs if p is not None]

    async def __len__(self) -> int:
        """Number of elites in this island."""
        return len(await self.get_elite_ids())

    async def _enforce_size_limit(self) -> None:
        """If `max_size` is set, remove excess programs using the configured remover."""
        if self.config.max_size is None or self.config.archive_remover is None:
            return

        current = await self.__len__()
        if current <= self.config.max_size:
            return

        logger.warning(
            "Island {}: enforcing size limit {} → {}",
            self.config.island_id,
            current,
            self.config.max_size,
        )

        to_remove: list[Program] = self.config.archive_remover(
            await self.get_elites(), self.config.max_size
        )
        removed = 0
        for prog in to_remove:
            await self.archive_storage.remove_elite_by_id(prog.id)
            await self.metadata_manager.clear_current_island(prog)
            removed += 1

        final_count = await self.__len__()
        logger.info(
            "Island {}: removed {} elites. Population: {} → {} (target {})",
            self.config.island_id,
            removed,
            current,
            final_count,
            self.config.max_size,
        )
