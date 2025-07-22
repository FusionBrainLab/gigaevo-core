import random
from typing import Any, List, Optional, Tuple, Dict
from pydantic import BaseModel, Field, computed_field, field_validator, ConfigDict
from loguru import logger
import asyncio

from src.evolution.storage.archive_storage import RedisArchiveStorage
from src.evolution.strategies.metadata_manager import MetadataManager
from src.database.redis_program_storage import RedisProgramStorage
from src.programs.program import Program
from .models import BehaviorSpace, QualityDiversityMetrics, SelectionMode, DEFAULT_MIGRATION_RATE
from .selectors import ArchiveSelector
from .removers import ArchiveRemover
from .elite_selectors import EliteSelector
from .migrant_selectors import MigrantSelector, TopFitnessMigrantSelector
from .qd_metrics import compute_qd_metrics_for_island

class IslandConfig(BaseModel):
    """Configuration for individual evolution islands."""
    island_id: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="Unique identifier for the island",
    )
    max_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of programs in the archive. If the archive is full, excess entries will be removed."
    )
    behavior_space: BehaviorSpace = Field(description="Behavior space configuration")
    archive_selector: ArchiveSelector = Field(description="Selector for choosing elite programs")
    archive_remover: Optional[ArchiveRemover] = Field(description="Remover for removing programs from the archive")
    elite_selector: EliteSelector = Field(description="Selector for choosing elite programs")
    migrant_selector: MigrantSelector = Field(description="Selector for choosing migrants")
    migration_rate: float = Field(
        default=DEFAULT_MIGRATION_RATE,
        ge=0.0,
        le=1.0,
        description="Rate of inter-island migration (0-1)",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field
    @property
    def redis_prefix(self) -> str:
        return f"island_{self.island_id}"

    @field_validator("archive_remover")
    def validate_archive_remover(cls, v, info):
        if info.data.get("max_size") is not None and v is None:
            raise ValueError("`max_size` is set, but `archive_remover` is not set")
        return v

class MapElitesIsland:
    """Single MAP-Elites island implementation."""
    def __init__(self, config: IslandConfig, program_storage: RedisProgramStorage):
        self.config = config
        self.program_storage = program_storage
        self.archive_storage = RedisArchiveStorage(
            program_storage=program_storage,
            key_prefix=config.redis_prefix
        )
        self.metadata_manager = MetadataManager(program_storage)
        self.last_qd_metrics: Optional[QualityDiversityMetrics] = None
        logger.info(f"Initialized MAP-Elites island {config.island_id} with max_size={config.max_size}")

    async def add(self, program: Program) -> bool:
        if not isinstance(program.metrics, dict):
            raise ValueError(f"Program metrics must be a dictionary, got {type(program.metrics)}")
        missing_keys = set(self.config.behavior_space.behavior_keys) - program.metrics.keys()
        if missing_keys:
            raise KeyError(f"Program missing required behavior keys: {missing_keys}")
        
        cell = self.config.behavior_space.get_cell(program.metrics)
        success = await self.archive_storage.add_elite(cell, program, self.config.archive_selector)
        if not success:
            return False
        
        await self.metadata_manager.set_current_island(program, self.config.island_id)
        await self.enforce_size_limit()
        
        logger.debug(f"Island {self.config.island_id}: Added program {program.id}")
        return True

    async def enforce_size_limit(self) -> None:
        """Enforce the maximum archive size by removing excess programs."""
        if self.config.max_size is None or self.config.archive_remover is None:
            return
        
        elites = await self.archive_storage.get_all_elites()
        current_count = len(elites)
        if current_count <= self.config.max_size:
            return
        
        logger.warning(f"Island {self.config.island_id}: Enforcing size limit - {current_count} elites, target {self.config.max_size}")
        
        to_remove = self.config.archive_remover(elites, self.config.max_size)
        removal_count = 0
        
        for elite in to_remove:
            success = await self.archive_storage.remove_elite_by_id(elite.id)
            if success:
                await self.metadata_manager.clear_current_island(elite)
                removal_count += 1
        
        final_elites = await self.archive_storage.get_all_elites()
        final_count = len(final_elites)
        
        if final_count > self.config.max_size:
            logger.error(f"CRITICAL: Island {self.config.island_id} size limit enforcement FAILED! "
                        f"Still has {final_count} elites after trying to remove {removal_count} programs. "
                        f"Target was {self.config.max_size}. This indicates a bug in the archive remover.")
        else:
            logger.info(f"Island {self.config.island_id}: Successfully removed {removal_count} elites. "
                       f"Population: {current_count} → {final_count} (target: {self.config.max_size})")

    async def select_elites(self, total: int) -> List[Program]:
        all_elites = await self.archive_storage.get_all_elites()
        if not all_elites:
            return []
        if len(all_elites) <= total:
            logger.debug(f"Island {self.config.island_id}: Only {len(all_elites)} elites available, requested {total}")
            return all_elites
        
        try:
            selected = self.config.elite_selector(all_elites, total)
            logger.debug(f"Island {self.config.island_id}: Selected {len(selected)} elites from {len(all_elites)}")
            return selected
        except Exception as e:
            logger.warning(f"Elite selection failed for island {self.config.island_id}: {e}")
            return random.sample(all_elites, min(total, len(all_elites)))

    async def get_all_elites(self) -> List[Program]:
        return await self.archive_storage.get_all_elites()

    async def get_elite_count(self) -> int:
        elites = await self.get_all_elites()
        return len(elites)

    async def get_archive_as_dict(self) -> Dict[Tuple[int, ...], Program]:
        archive = {}
        elites = await self.get_all_elites()
        for elite in elites:
            try:
                cell = self.config.behavior_space.get_cell(elite.metrics)
                archive[cell] = elite
            except Exception as e:
                logger.warning(f"Could not map elite {elite.id} to cell: {e}")
        return archive

    async def select_migrants(self, count: int) -> List[Program]:
        elites = await self.get_all_elites()
        if not elites:
            return []
        try:
            return self.config.migrant_selector(elites, count)
        except Exception as e:
            logger.warning(f"Migrant selection failed for island {self.config.island_id}: {e}")
            return random.sample(elites, min(count, len(elites)))

    async def compute_qd_metrics(self) -> QualityDiversityMetrics:
        elites = await self.get_all_elites()
        metrics = compute_qd_metrics_for_island(elites, self.config)
        self.last_qd_metrics = metrics
        return metrics