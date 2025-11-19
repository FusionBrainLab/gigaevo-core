from __future__ import annotations

from typing import Union

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.evolution.storage.archive_storage import RedisArchiveStorage
from gigaevo.evolution.strategies.elite_selectors import EliteSelector
from gigaevo.evolution.strategies.migrant_selectors import MigrantSelector
from gigaevo.evolution.strategies.models import BehaviorSpace, DynamicBehaviorSpace
from gigaevo.evolution.strategies.removers import ArchiveRemover
from gigaevo.evolution.strategies.selectors import ArchiveSelector
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState

# Metadata keys for island tracking
METADATA_KEY_HOME_ISLAND = "home_island"
METADATA_KEY_CURRENT_ISLAND = "current_island"


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
    behavior_space: Union[DynamicBehaviorSpace, BehaviorSpace]
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
        self.state_manager = ProgramStateManager(program_storage)
        logger.info("Island {} init (max_size={})", config.island_id, config.max_size)

    # -------------------------- Public API --------------------------

    async def add(self, program: Program) -> bool:
        """Insert `program` into its behavior cell if it improves the elite."""
        missing = set(self.config.behavior_space.behavior_keys) - program.metrics.keys()
        if missing:
            logger.debug(
                "Island {}: program {} missing behavior keys: {}",
                self.config.island_id,
                program.id,
                missing,
            )
            raise KeyError(f"Program missing required behavior keys: {missing}")

        # Map program to behavior cell
        # Check for expansion first (only if dynamic)
        if isinstance(self.config.behavior_space, DynamicBehaviorSpace):
            if self.config.behavior_space.check_and_expand(program.metrics):
                logger.info(
                    "Island {}: behavior space expanded by program {}, triggering re-indexing",
                    self.config.island_id,
                    program.id,
                )
                await self.reindex_archive()

        cell = self.config.behavior_space.get_cell(program.metrics)
        behavior_values = {
            k: program.metrics[k] for k in self.config.behavior_space.behavior_keys
        }
        logger.debug(
            "Island {}: program {} -> cell {} (behavior: {})",
            self.config.island_id,
            program.id,
            cell,
            behavior_values,
        )

        # Try to add to archive
        current_elite = await self.archive_storage.get_elite(cell)
        improved = await self.archive_storage.add_elite(
            cell, program, self.config.archive_selector
        )

        if not improved:
            if current_elite:
                logger.debug(
                    "Island {}: program {} rejected (cell {} occupied by {})",
                    self.config.island_id,
                    program.id,
                    cell,
                    current_elite.id,
                )
            else:
                logger.debug(
                    "Island {}: program {} rejected (failed archive criteria)",
                    self.config.island_id,
                    program.id,
                )
            return False

        if current_elite:
            logger.debug(
                "Island {}: program {} replaced {} in cell {}",
                self.config.island_id,
                program.id,
                current_elite.id,
                cell,
            )
        else:
            logger.debug(
                "Island {}: program {} filled empty cell {}",
                self.config.island_id,
                program.id,
                cell,
            )

        program.metadata.setdefault(METADATA_KEY_HOME_ISLAND, self.config.island_id)
        program.metadata[METADATA_KEY_CURRENT_ISLAND] = self.config.island_id
        await self.state_manager.update_program(program)

        # If behavior space is dynamic, optimize bounds aggressively on success
        if isinstance(self.config.behavior_space, DynamicBehaviorSpace):
            await self.optimize_space()

        await self._enforce_size_limit()
        return True

    async def select_elites(self, total: int) -> list[Program]:
        """Return up to `total` elite programs for mutation."""
        elites = await self.get_elites()
        archive_size = len(elites)

        logger.debug(
            "Island {}: selecting elites (requested={}, archive_size={})",
            self.config.island_id,
            total,
            archive_size,
        )

        if not elites or total <= 0:
            logger.debug("Island {}: no elites to select", self.config.island_id)
            return []

        if len(elites) <= total:
            logger.debug(
                "Island {}: returning all {} elites (≤ requested {})",
                self.config.island_id,
                len(elites),
                total,
            )
            return elites

        # Use configured selector
        selector_type = type(self.config.elite_selector).__name__
        selected = self.config.elite_selector(elites, total)

        logger.debug(
            "Island {}: selected {} / {} elites using {}",
            self.config.island_id,
            len(selected),
            len(elites),
            selector_type,
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
            logger.debug(
                "Island {}: size check OK ({}/{})",
                self.config.island_id,
                current,
                self.config.max_size,
            )
            return

        excess = current - self.config.max_size
        logger.warning(
            "Island {}: size limit exceeded! {} programs over limit ({}/{})",
            self.config.island_id,
            excess,
            current,
            self.config.max_size,
        )

        remover_type = type(self.config.archive_remover).__name__
        logger.debug(
            "Island {}: using {} to remove {} programs",
            self.config.island_id,
            remover_type,
            excess,
        )

        to_remove: list[Program] = self.config.archive_remover(
            await self.get_elites(), self.config.max_size
        )

        logger.debug(
            "Island {}: remover selected {} programs for removal: {}",
            self.config.island_id,
            len(to_remove),
            [p.id for p in to_remove[:5]] + (["..."] if len(to_remove) > 5 else []),
        )

        removed = 0
        for prog in to_remove:
            await self.archive_storage.remove_elite_by_id(prog.id)
            if prog.metadata.get(METADATA_KEY_CURRENT_ISLAND):
                prog.metadata[METADATA_KEY_CURRENT_ISLAND] = None
                await self.state_manager.update_program(prog)
            await self.state_manager.set_program_state(prog, ProgramState.DISCARDED)
            removed += 1

        final_count = await self.__len__()
        logger.info(
            "Island {}: size enforcement complete. {} → {} (target: {}, removed: {})",
            self.config.island_id,
            current,
            final_count,
            self.config.max_size,
            removed,
        )

        # Opportunity to shrink/optimize space if we removed items
        if removed > 0:
            await self.optimize_space()

    async def reindex_archive(self) -> None:
        """Re-calculate cell coordinates for all elites based on current behavior space."""
        # 1. Get all current elites
        elites = await self.get_elites()
        if not elites:
            return

        # 2. Clear the current mapping
        await self.archive_storage.clear_all_elites()

        # 3. Batch re-insert
        # We can't simply use bulk_add because we need to handle collisions
        # (two elites might now map to the same cell).
        # We sort by fitness (or whatever archive_selector prioritizes) to ensure best ones win.
        # However, archive_selector is a comparator, not a key.
        # For simplicity, we just re-add them one by one. The archive logic handles replacements.

        readded = 0

        # Pre-calculate placements
        placements = []
        for p in elites:
            try:
                cell = self.config.behavior_space.get_cell(p.metrics)
                placements.append((cell, p))
            except Exception as e:
                logger.warning(
                    "Island {}: failed to map program {} during re-index: {}",
                    self.config.island_id,
                    p.id,
                    e,
                )

        # Use bulk add (which handles is_better logic internally per item)
        readded = await self.archive_storage.bulk_add_elites(
            placements, self.config.archive_selector
        )

        logger.info(
            "Island {}: re-indexed {} elites ({} preserved)",
            self.config.island_id,
            len(elites),
            readded,
        )

    async def optimize_space(self) -> None:
        """Analyze current population and optimize behavior space bounds (shrink/tighten)."""
        if not isinstance(self.config.behavior_space, DynamicBehaviorSpace):
            return

        elites = await self.get_elites()
        if not elites:
            return

        metrics_batch = [p.metrics for p in elites]

        # Calculate tight bounds with buffer
        new_bounds = self.config.behavior_space.calculate_optimized_bounds(
            metrics_batch
        )

        old_description = self.config.behavior_space.describe()
        if self.config.behavior_space.update_bounds(new_bounds):
            new_description = self.config.behavior_space.describe()

            changes = []
            for key in self.config.behavior_space.behavior_keys:
                old_min = old_description[key]["min"]
                old_max = old_description[key]["max"]
                new_min = new_description[key]["min"]
                new_max = new_description[key]["max"]

                if old_min != new_min or old_max != new_max:
                    changes.append(
                        f"{key}: [{old_min:.3f}, {old_max:.3f}] -> [{new_min:.3f}, {new_max:.3f}]"
                    )

            logger.info(
                "Island {}: optimized behavior space bounds:\n  {}",
                self.config.island_id,
                "\n  ".join(changes),
            )
            await self.reindex_archive()
