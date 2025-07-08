import random
from typing import Any, List, Optional, Tuple, Dict
import numpy as np
from pydantic import BaseModel, Field, computed_field, field_validator, ConfigDict
from loguru import logger
import asyncio

from src.evolution.storage.archive_storage import RedisArchiveStorage
from src.database.program_storage import RedisProgramStorage
from src.programs.program import Program
from .models import BehaviorSpace, QualityDiversityMetrics, SelectionMode, DEFAULT_MIGRATION_RATE
from .selectors import ArchiveSelector
from .removers import ArchiveRemover
from .elite_selectors import EliteSelector
from .migrant_selectors import MigrantSelector, TopFitnessMigrantSelector


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
    """Single MAP-Elites island implementation with robust size enforcement."""

    def __init__(self, config: IslandConfig, program_storage: RedisProgramStorage):
        self.config = config
        self.program_storage = program_storage
        self.archive_storage = RedisArchiveStorage(
            program_storage=program_storage,
            key_prefix=config.redis_prefix
        )
        self.last_qd_metrics: Optional[QualityDiversityMetrics] = None
        # Lock for all operations that modify the archive
        self._archive_lock = asyncio.Lock()
        # Cache for elite count to reduce expensive get_all_elites() calls
        self._cached_elite_count: Optional[int] = None
        self._cache_invalidated = True

        logger.info(f"Initialized MAP-Elites island {config.island_id} with max_size={config.max_size}")

    def _cell_key(self, cell: Tuple[int, ...]) -> str:
        return f"cell:{','.join(map(str, cell))}"

    async def _get_elite_count_cached(self) -> int:
        """Get elite count with caching to reduce expensive operations."""
        if self._cache_invalidated or self._cached_elite_count is None:
            elites = await self.get_all_elites()
            self._cached_elite_count = len(elites)
            self._cache_invalidated = False
        return self._cached_elite_count

    def _invalidate_cache(self):
        """Invalidate the cache when the archive is modified."""
        self._cache_invalidated = True

    async def add(self, program: Program) -> bool:
        """Add a program with robust size enforcement and minimal race conditions."""
        try:
            if not isinstance(program.metrics, dict):
                raise ValueError(f"Program metrics must be a dictionary, got {type(program.metrics)}")

            missing_keys = set(self.config.behavior_space.behavior_keys) - program.metrics.keys()
            if missing_keys:
                raise KeyError(f"Program missing required behavior keys: {missing_keys}")

            # Use the lock for all archive operations to prevent race conditions
            async with self._archive_lock:
                cell = self.config.behavior_space.get_cell(program.metrics)
                cell_key = self._cell_key(cell)
                
                # Check if there's already an elite in this cell
                existing_elite = await self.archive_storage.get_elite(cell_key)
                
                # If there's an existing elite, check if the new program is better
                if existing_elite is not None:
                    if not self.config.archive_selector(program, existing_elite):
                        logger.debug(f"Island {self.config.island_id}: Program {program.id} not better than existing elite")
                        return False
                    # This will replace the existing elite, so no size change
                    replacement = True
                else:
                    # This will add a new elite, check capacity
                    replacement = False
                    if self.config.max_size is not None:
                        current_count = await self._get_elite_count_cached()
                        if current_count >= self.config.max_size:
                            # We're at capacity and adding to an empty cell
                            # Need to remove one program first
                            await self._enforce_size_limit_internal(self.config.max_size - 1)
                            logger.debug(f"Island {self.config.island_id}: Made room for new program")

                # Add the program
                updated = await self._update_cell_elite(cell, program)
                if not updated:
                    logger.warning(f"Island {self.config.island_id}: Failed to update cell elite for program {program.id}")
                    return False

                # Update metadata
                program.metadata.setdefault("home_island", self.config.island_id)
                program.metadata["current_island"] = self.config.island_id
                try:
                    await self.program_storage.update(program)
                except Exception as e:
                    logger.warning(f"Failed to persist metadata for program {program.id}: {e}")

                # Invalidate cache since we modified the archive
                if not replacement:
                    self._invalidate_cache()
                # Ensure max_size constraint even after replacements
                if self.config.max_size is not None:
                    await self._enforce_size_limit_internal(self.config.max_size)

                logger.debug(f"Island {self.config.island_id}: Added program {program.id} to cell {cell}")
                return True

        except Exception as e:
            logger.error(f"Failed to add program {program.id} to island {self.config.island_id}: {e}")
            raise

    async def _update_cell_elite(self, cell: Tuple[int, ...], program: Program) -> bool:
        """Update a cell elite (must be called within _archive_lock)."""
        cell_key = self._cell_key(cell)

        def _is_better(new: Program, curr: Program) -> bool:
            return self.config.archive_selector(new, curr)

        updated = await self.archive_storage.set_elite_if_better(cell_key, program, _is_better)
        if updated:
            logger.debug(f"Island {self.config.island_id}: Updated elite in cell {cell} with program {program.id}")
        return updated

    async def _enforce_size_limit_internal(self, target_size: int) -> None:
        """Internal size enforcement that assumes we're already in the lock."""
        if self.config.max_size is None or self.config.archive_remover is None:
            return

        elites = await self.get_all_elites()
        current_count = len(elites)
        
        if current_count <= target_size:
            return

        logger.warning(f"Island {self.config.island_id}: Enforcing size limit - {current_count} elites, target {target_size}")
        
        to_remove = self.config.archive_remover(elites, target_size)
        removal_count = 0
        
        for elite in to_remove:
            try:
                cell = self.config.behavior_space.get_cell(elite.metrics)
                cell_key = self._cell_key(cell)
                await self.archive_storage.remove_elite(cell_key, elite)
                removal_count += 1
                logger.debug(f"Removed elite {elite.id} from island {self.config.island_id}")
            except Exception as exc:
                logger.warning(f"Failed to remove elite {elite.id}: {exc}")

        # Invalidate cache after removals
        self._invalidate_cache()
        
        logger.info(f"Island {self.config.island_id}: Removed {removal_count} elites to enforce size limit")

    async def enforce_size_limit(self) -> None:
        """Public method to enforce size limit (with lock)."""
        if self.config.max_size is None:
            return
            
        async with self._archive_lock:
            await self._enforce_size_limit_internal(self.config.max_size)

    async def select_elites(self, total: int) -> List[Program]:
        """Select elites for evolution."""
        all_elites = await self.get_all_elites()
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
        """Get all elite programs."""
        return await self.archive_storage.iter_all_elites()

    async def get_elite_count(self) -> int:
        """Get the number of elite programs (cached for efficiency)."""
        return await self._get_elite_count_cached()

    async def get_archive_as_dict(self) -> Dict[Tuple[int, ...], Program]:
        """Get archive as a dictionary mapping cells to programs."""
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
        """Select programs for migration."""
        elites = await self.get_all_elites()
        if not elites:
            return []

        try:
            return self.config.migrant_selector(elites, count)
        except Exception as e:
            logger.warning(f"Migrant selection failed for island {self.config.island_id}: {e}")
            return random.sample(elites, min(count, len(elites)))

    async def compute_qd_metrics(self) -> QualityDiversityMetrics:
        """Compute Quality-Diversity metrics for this island."""
        elites = await self.get_all_elites()
        
        if not elites:
            return QualityDiversityMetrics(
                qd_score=0.0,
                coverage=0.0,
                maximum_fitness=float('-inf'),
                average_fitness=0.0,
                filled_cells=0,
                total_cells=self.config.behavior_space.total_cells
            )

        # Calculate metrics
        fitness_values = [elite.metrics.get('fitness', 0.0) for elite in elites]
        max_fitness = max(fitness_values) if fitness_values else 0.0
        avg_fitness = sum(fitness_values) / len(fitness_values) if fitness_values else 0.0
        
        # QD score is sum of all fitness values (total quality)
        qd_score = sum(fitness_values)
        
        # Coverage is fraction of cells filled
        filled_cells = len(elites)
        total_cells = self.config.behavior_space.total_cells
        coverage = filled_cells / total_cells if total_cells > 0 else 0.0
        
        metrics = QualityDiversityMetrics(
            qd_score=qd_score,
            coverage=coverage,
            maximum_fitness=max_fitness,
            average_fitness=avg_fitness,
            filled_cells=filled_cells,
            total_cells=total_cells
        )
        
        self.last_qd_metrics = metrics
        return metrics