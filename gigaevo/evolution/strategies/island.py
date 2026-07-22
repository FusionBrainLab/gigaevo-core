from __future__ import annotations

import asyncio
import math

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.evolution.storage.archive_storage import ArchiveStorageFactory
from gigaevo.evolution.strategies.elite_selectors import EliteSelector
from gigaevo.evolution.strategies.migrant_selectors import MigrantSelector
from gigaevo.evolution.strategies.models import BehaviorSpace, DynamicBehaviorSpace
from gigaevo.evolution.strategies.removers import ArchiveRemover
from gigaevo.evolution.strategies.selectors import ArchiveSelector
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.utils.json import dumps as _dumps
from gigaevo.utils.json import loads as _loads

# Metadata keys for island tracking
METADATA_KEY_HOME_ISLAND = "home_island"
METADATA_KEY_CURRENT_ISLAND = "current_island"
_DYNAMIC_SPACE_STATE_PREFIX = "strategy:dynamic_behavior_space"
_DYNAMIC_SPACE_STATE_VERSION = 1


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
    behavior_space: DynamicBehaviorSpace | BehaviorSpace
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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def archive_prefix(self) -> str:
        return f"island_{self.island_id}"

    @field_validator("archive_remover")
    @classmethod
    def _validate_archive_remover(cls, v, info):
        if info.data.get("max_size") is not None and v is None:
            raise ValueError("`max_size` is set but `archive_remover` is None")
        return v


class MapElitesIsland:
    """Single MAP-Elites island."""

    def __init__(
        self,
        config: IslandConfig,
        program_storage: ProgramStorage,
        archive_storage_factory: ArchiveStorageFactory,
    ):
        self.config = config
        self.program_storage = program_storage
        self.archive_storage = archive_storage_factory(config.archive_prefix)
        self.state_manager = ProgramStateManager(program_storage)
        # Bounds and archive cell keys form one logical state.  Serialize their
        # mutation and Memory V2 snapshots so readers cannot observe a rebin in
        # progress.
        self._archive_state_lock = asyncio.Lock()
        logger.info(
            "[Island][{}] init (max_size={})", config.island_id, config.max_size
        )

    # -------------------------- Public API --------------------------

    async def add(self, program: Program) -> bool:
        """Insert `program` into its behavior cell if it improves the elite."""
        async with self._archive_state_lock:
            return await self._add_locked(program)

    async def _add_locked(self, program: Program) -> bool:
        """Implement :meth:`add` while the archive-state lock is held."""
        missing = set(self.config.behavior_space.behavior_keys) - program.metrics.keys()
        if missing:
            logger.debug(
                "[Island][{}] program {} missing behavior keys: {}",
                self.config.island_id,
                program.id,
                missing,
            )
            raise KeyError(f"Program missing required behavior keys: {missing}")

        # Map program to behavior cell
        # Check for expansion first (only if dynamic)
        if isinstance(self.config.behavior_space, DynamicBehaviorSpace):
            previous_bounds = self._live_bounds()
            if self.config.behavior_space.check_and_expand(program.metrics):
                logger.info(
                    "[Island][{}] behavior space expanded by program {}, triggering re-indexing",
                    self.config.island_id,
                    program.id,
                )
                await self._commit_rebin_locked(previous_bounds)

        cell = self.config.behavior_space.get_cell(program.metrics)
        behavior_values = {
            k: program.metrics[k] for k in self.config.behavior_space.behavior_keys
        }
        logger.debug(
            "[Island][{}] program {} -> cell {} (behavior: {})",
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
                    "[Island][{}] program {} rejected (cell {} occupied by {})",
                    self.config.island_id,
                    program.id,
                    cell,
                    current_elite.id,
                )
            else:
                logger.debug(
                    "[Island][{}] program {} rejected (failed archive criteria)",
                    self.config.island_id,
                    program.id,
                )
            return False

        if current_elite:
            logger.info(
                "[Island][{}] FRONTIER IMPROVED | {} replaced {} in cell {}"
                " | old_metrics={} new_metrics={}",
                self.config.island_id,
                program.short_id,
                current_elite.short_id,
                cell,
                current_elite.metrics,
                program.metrics,
            )
        else:
            logger.info(
                "[Island][{}] FRONTIER NEW CELL | {} filled cell {} (metrics={})",
                self.config.island_id,
                program.short_id,
                cell,
                program.metrics,
            )

        program.metadata.setdefault(METADATA_KEY_HOME_ISLAND, self.config.island_id)
        program.metadata[METADATA_KEY_CURRENT_ISLAND] = self.config.island_id
        await self.state_manager.update_program(program)

        # If behavior space is dynamic, optimize bounds aggressively on success
        if isinstance(self.config.behavior_space, DynamicBehaviorSpace):
            await self._optimize_space_locked()

        await self._enforce_size_limit_locked()
        return True

    async def can_accept(self, program: Program) -> bool:
        """Check archive compatibility without mutating dynamic bounds."""

        required = set(self.config.behavior_space.behavior_keys)
        if not required.issubset(program.metrics):
            return False

        async with self._archive_state_lock:
            candidate_space = self.config.behavior_space.model_copy(deep=True)
            expanded = False
            if isinstance(candidate_space, DynamicBehaviorSpace):
                expanded = candidate_space.check_and_expand(program.metrics)
            cell = candidate_space.get_cell(program.metrics)

            if not expanded:
                incumbent = await self.archive_storage.get_elite(cell)
                return incumbent is None or self.config.archive_selector(
                    program, incumbent
                )

            # Expansion changes every cell boundary. Simulate the post-reindex
            # occupant on the copied grid rather than querying old physical keys.
            current: Program | None = None
            for elite in await self.get_elites():
                try:
                    elite_cell = candidate_space.get_cell(elite.metrics)
                except Exception:
                    continue
                if elite_cell != cell:
                    continue
                if current is None or self.config.archive_selector(elite, current):
                    current = elite
            return current is None or self.config.archive_selector(program, current)

    async def select_elites(self, total: int) -> list[Program]:
        """Return up to `total` elite programs for mutation."""
        async with self._archive_state_lock:
            elites = await self.get_elites()
        archive_size = len(elites)

        logger.debug(
            "[Island][{}] selecting elites (requested={}, archive_size={})",
            self.config.island_id,
            total,
            archive_size,
        )

        if not elites or total <= 0:
            logger.debug("[Island][{}] no elites to select", self.config.island_id)
            return []

        if len(elites) <= total:
            logger.debug(
                "[Island][{}] returning all {} elites (≤ requested {})",
                self.config.island_id,
                len(elites),
                total,
            )
            return elites

        # Use configured selector
        selector_type = type(self.config.elite_selector).__name__
        selected = self.config.elite_selector(elites, total)

        logger.debug(
            "[Island][{}] selected {} / {} elites using {}",
            self.config.island_id,
            len(selected),
            len(elites),
            selector_type,
        )
        return selected

    async def select_migrants(self, count: int) -> list[Program]:
        """Select programs to emigrate to other islands."""
        async with self._archive_state_lock:
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

    async def remove_elite_by_id(self, program_id: str) -> bool:
        """Remove one elite without racing a dynamic archive replacement."""

        async with self._archive_state_lock:
            return await self.archive_storage.remove_elite_by_id(program_id)

    async def snapshot_archive_state(self) -> tuple[list[Program], BehaviorSpace]:
        """Return elites and their behavior space from one coherent rebin epoch."""

        async with self._archive_state_lock:
            elites = await self.get_elites()
            behavior_space = self.config.behavior_space.model_copy(deep=True)
            return elites, behavior_space

    async def __len__(self) -> int:
        """Number of elites in this island."""
        return await self.archive_storage.size()

    async def _enforce_size_limit(self) -> None:
        async with self._archive_state_lock:
            await self._enforce_size_limit_locked()

    async def _enforce_size_limit_locked(self) -> None:
        """If `max_size` is set, remove excess programs using the configured remover."""
        if self.config.max_size is None or self.config.archive_remover is None:
            return

        current = await self.__len__()
        if current <= self.config.max_size:
            logger.debug(
                "[Island][{}] size check OK ({}/{})",
                self.config.island_id,
                current,
                self.config.max_size,
            )
            return

        excess = current - self.config.max_size
        logger.warning(
            "[Island][{}] size limit exceeded! {} programs over limit ({}/{})",
            self.config.island_id,
            excess,
            current,
            self.config.max_size,
        )

        remover_type = type(self.config.archive_remover).__name__
        logger.debug(
            "[Island][{}] using {} to remove {} programs",
            self.config.island_id,
            remover_type,
            excess,
        )

        to_remove: list[Program] = self.config.archive_remover(
            await self.get_elites(), self.config.max_size
        )

        logger.debug(
            "[Island][{}] remover selected {} programs for removal: {}",
            self.config.island_id,
            len(to_remove),
            [p.id for p in to_remove[:5]] + (["..."] if len(to_remove) > 5 else []),
        )

        # Batch-remove from archive (constant number of storage round-trips)
        await self.archive_storage.bulk_remove_elites_by_id([p.id for p in to_remove])

        # Per-program state work — run in parallel (each targets a different program/lock)
        async def _discard_one(prog: Program) -> None:
            if prog.metadata.get(METADATA_KEY_CURRENT_ISLAND):
                prog.metadata[METADATA_KEY_CURRENT_ISLAND] = None
                await self.state_manager.update_program(prog)
            await self.state_manager.set_program_state(prog, ProgramState.DISCARDED)

        await asyncio.gather(
            *[_discard_one(p) for p in to_remove], return_exceptions=True
        )
        removed = len(to_remove)

        final_count = await self.__len__()
        logger.info(
            "[Island][{}] size enforcement complete. {} → {} (target: {}, removed: {})",
            self.config.island_id,
            current,
            final_count,
            self.config.max_size,
            removed,
        )

        # Opportunity to shrink/optimize space if we removed items
        if removed > 0:
            await self._optimize_space_locked()

    async def reindex_archive(self) -> None:
        """Re-calculate cell coordinates for all elites based on current behavior space."""
        async with self._archive_state_lock:
            await self._reindex_archive_locked()
            await self._save_dynamic_space_state_locked()

    async def _reindex_archive_locked(
        self, elites: list[Program] | None = None
    ) -> None:
        """Atomically re-index while the archive-state lock is held."""

        if elites is None:
            elites = await self.get_elites()
        placements = []
        for p in elites:
            try:
                cell = self.config.behavior_space.get_cell(p.metrics)
                placements.append((cell, p))
            except Exception as e:
                raise ValueError(
                    f"failed to map program {p.id!r} while re-indexing island "
                    f"{self.config.island_id!r}"
                ) from e

        # Collision resolution happens before one complete map is published, so
        # readers see either the pre-rebin or post-rebin archive, never a partial
        # clear-and-repopulate state.
        readded = await self.archive_storage.replace_all_elites(
            placements,
            self.config.archive_selector,
        )

        logger.info(
            "[Island][{}] re-indexed {} elites ({} preserved)",
            self.config.island_id,
            len(elites),
            readded,
        )

    async def optimize_space(self) -> None:
        """Analyze current population and optimize behavior space bounds (shrink/tighten)."""
        async with self._archive_state_lock:
            await self._optimize_space_locked()

    async def _optimize_space_locked(self) -> None:
        """Optimize dynamic bounds while the archive-state lock is held."""

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
        previous_bounds = self._live_bounds()
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
                "[Island][{}] optimized behavior space bounds:\n  {}",
                self.config.island_id,
                "\n  ".join(changes),
            )
            await self._commit_rebin_locked(previous_bounds)

    def _live_bounds(self) -> dict[str, tuple[float, float]]:
        return {
            key: (binning.min_val, binning.max_val)
            for key, binning in self.config.behavior_space.bins.items()
        }

    def _apply_live_bounds(self, bounds: dict[str, tuple[float, float]]) -> None:
        space = self.config.behavior_space
        if not isinstance(space, DynamicBehaviorSpace):
            return
        space.update_bounds(
            {key: (lower, upper) for key, (lower, upper) in bounds.items()}
        )

    async def _commit_rebin_locked(
        self, previous_bounds: dict[str, tuple[float, float]]
    ) -> None:
        """Commit bounds, archive keys, and resume metadata as one logical change."""

        elites = await self.get_elites()
        try:
            await self._reindex_archive_locked(elites)
            await self._save_dynamic_space_state_locked()
        except BaseException:
            # Replacement is atomic for the supported stores, so an ordinary
            # failed publish leaves the prior map in place.
            self._apply_live_bounds(previous_bounds)
            raise

    @property
    def _dynamic_space_state_key(self) -> str:
        return f"{_DYNAMIC_SPACE_STATE_PREFIX}:{self.config.island_id}"

    def _dynamic_space_state_payload(self) -> dict[str, object]:
        return {
            "version": _DYNAMIC_SPACE_STATE_VERSION,
            "live_bounds": {
                key: [lower, upper]
                for key, (lower, upper) in self._live_bounds().items()
            },
        }

    def _parse_dynamic_space_state(self, raw: str) -> dict[str, tuple[float, float]]:
        try:
            payload = _loads(raw)
        except Exception as exc:
            raise ValueError(
                f"invalid dynamic behavior-space state for island "
                f"{self.config.island_id!r}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError("dynamic behavior-space state must be an object")
        if payload.get("version") != _DYNAMIC_SPACE_STATE_VERSION:
            raise ValueError(
                "unsupported dynamic behavior-space state version: "
                f"{payload.get('version')!r}"
            )
        raw_bounds = payload.get("live_bounds")
        if not isinstance(raw_bounds, dict) or set(raw_bounds) != set(
            self.config.behavior_space.behavior_keys
        ):
            raise ValueError("persisted dynamic behavior-space bounds are incomplete")

        result: dict[str, tuple[float, float]] = {}
        space = self.config.behavior_space
        if not isinstance(space, DynamicBehaviorSpace):
            return result
        for key in space.behavior_keys:
            row = raw_bounds[key]
            if not isinstance(row, list) or len(row) != 2:
                raise ValueError(f"invalid persisted bounds for behavior axis {key!r}")
            lower, upper = float(row[0]), float(row[1])
            hard_lower, hard_upper = space._initial_bounds[key]
            if (
                not math.isfinite(lower)
                or not math.isfinite(upper)
                or lower > upper
                or lower < hard_lower
                or upper > hard_upper
            ):
                raise ValueError(f"invalid persisted bounds for behavior axis {key!r}")
            result[key] = (lower, upper)
        return result

    async def _save_dynamic_space_state_locked(self) -> None:
        if not isinstance(self.config.behavior_space, DynamicBehaviorSpace):
            return
        await self.program_storage.save_run_state(
            self._dynamic_space_state_key,
            _dumps(self._dynamic_space_state_payload()),
        )

    async def restore_dynamic_space_state(self) -> None:
        """Restore live dynamic bounds and repair persisted archive cell keys."""

        space = self.config.behavior_space
        if not isinstance(space, DynamicBehaviorSpace):
            return
        async with self._archive_state_lock:
            elites = await self.get_elites()
            raw = await self.program_storage.load_run_state_str(
                self._dynamic_space_state_key
            )
            if raw is not None:
                self._apply_live_bounds(self._parse_dynamic_space_state(raw))
            elif elites:
                # Compatibility for archives created before dynamic state was
                # persisted: reconstruct deterministic tight bounds from elites.
                self._apply_live_bounds(
                    space.calculate_optimized_bounds(
                        [elite.metrics for elite in elites]
                    )
                )
            await self._reindex_archive_locked(elites)
            await self._save_dynamic_space_state_locked()
