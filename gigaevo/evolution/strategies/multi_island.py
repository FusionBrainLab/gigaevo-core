from __future__ import annotations

import asyncio
import random

from loguru import logger

from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.evolution.strategies.base import EvolutionStrategy, StrategyMetrics
from gigaevo.evolution.strategies.island import (
    METADATA_KEY_CURRENT_ISLAND,
    IslandConfig,
    MapElitesIsland,
)
from gigaevo.evolution.strategies.island_selector import WeightedIslandSelector
from gigaevo.evolution.strategies.mutant_router import RandomMutantRouter
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


class MapElitesMultiIsland(EvolutionStrategy):
    """Multi-island MAP-Elites strategy (updated to new island API)."""

    def __init__(
        self,
        island_configs: list[IslandConfig],
        program_storage: RedisProgramStorage,
        migration_interval: int = 50,
        enable_migration: bool = True,
        max_migrants_per_island: int = 5,
        island_selector: WeightedIslandSelector | None = None,
        mutant_router: RandomMutantRouter | None = None,
    ):
        if not island_configs:
            raise ValueError("At least one island configuration is required")

        self.islands: dict[str, MapElitesIsland] = {
            cfg.island_id: MapElitesIsland(cfg, program_storage)
            for cfg in island_configs
        }

        self.program_storage = program_storage
        self.migration_interval = int(migration_interval)
        self.enable_migration = bool(enable_migration)
        self.max_migrants_per_island = int(max_migrants_per_island)

        self.generation = 0
        self.last_migration = 0

        # Pluggables (kept for API completeness)
        self.island_selector = island_selector or WeightedIslandSelector()
        self.mutant_router = mutant_router or RandomMutantRouter()

        capped = [cfg.max_size for cfg in island_configs if cfg.max_size is not None]
        self.max_size = sum(capped) if capped else None

        logger.info(
            "Initialized MAP-Elites with {} island(s), global max_size={}",
            len(self.islands),
            self.max_size,
        )

    # --------------------------- Public API ---------------------------

    async def add(self, program: Program, island_id: str | None = None) -> bool:
        """Add a program to a specific island or route it automatically."""
        logger.debug(
            "MultiIsland: adding program {} (island_id={})",
            program.id,
            island_id or "auto-route",
        )

        if island_id is not None and island_id not in self.islands:
            logger.debug(
                "MultiIsland: program {} rejected (unknown island '{}')",
                program.id,
                island_id,
            )
            return False

        island = (
            self.islands[island_id]
            if island_id is not None
            else await self.mutant_router.route_mutant(
                program, list(self.islands.values())
            )
        )

        if island is None:
            logger.debug(
                "MultiIsland: program {} rejected (router returned None)",
                program.id,
            )
            return False

        logger.debug(
            "MultiIsland: routing program {} to island '{}'",
            program.id,
            island.config.island_id,
        )
        result = await island.add(program)

        if result:
            logger.debug(
                "MultiIsland: program {} successfully added to island '{}'",
                program.id,
                island.config.island_id,
            )
        else:
            logger.debug(
                "MultiIsland: program {} rejected by island '{}'",
                program.id,
                island.config.island_id,
            )

        return result

    async def select_elites(self, total: int = 10) -> list[Program]:
        """
        Sample elites from all islands (migration & enforcement on schedule).
        Returns up to `total` elite programs.
        """
        logger.debug(
            "MultiIsland: selecting elites (gen={}, total={}, islands={})",
            self.generation,
            total,
            len(self.islands),
        )

        # Check if migration is due
        if self.enable_migration:
            gens_since_migration = self.generation - self.last_migration
            logger.debug(
                "MultiIsland: migration check (gens_since_last={}, interval={})",
                gens_since_migration,
                self.migration_interval,
            )

            if gens_since_migration >= self.migration_interval:
                logger.info(
                    "MultiIsland: triggering migration (generation {}, last migration at {})",
                    self.generation,
                    self.last_migration,
                )
                await self._perform_migration()
                await self._enforce_all_island_size_limits()
                self.last_migration = self.generation

        # Calculate per-island quotas
        quotas = self._calculate_island_quotas(total)
        logger.debug(
            "MultiIsland: island quotas: {}",
            {k: v for k, v in quotas.items() if v > 0},
        )

        tasks = [
            asyncio.create_task(self.islands[island_id].select_elites(quota))
            for island_id, quota in quotas.items()
            if quota > 0
        ]
        if not tasks:
            logger.debug("MultiIsland: no elites to select (all islands empty)")
            return []

        selections = await asyncio.gather(*tasks)
        results = [p for group in selections for p in group]

        logger.debug(
            "MultiIsland: collected {} elites from {} islands",
            len(results),
            len(tasks),
        )

        # Shuffle and sample if needed
        random.shuffle(results)
        if len(results) > total:
            logger.debug(
                "MultiIsland: sampling {} from {} collected elites",
                total,
                len(results),
            )
            results = random.sample(results, total)

        if results:
            self.generation += 1
            logger.debug(
                "MultiIsland: selected {} elites (generation {} -> {})",
                len(results),
                self.generation - 1,
                self.generation,
            )

        return results

    async def select_migrants(self, count: int) -> list[Program]:
        """Select migrants across all islands (utility)."""
        tasks = [
            asyncio.create_task(island.select_migrants(count))
            for island in self.islands.values()
        ]
        groups = await asyncio.gather(*tasks)
        return [p for g in groups for p in g]

    async def get_program_ids(self) -> list[str]:
        tasks = [
            asyncio.create_task(island.get_elite_ids())
            for island in self.islands.values()
        ]
        groups = await asyncio.gather(*tasks)  # list[list[str]]
        ids: list[str] = [pid for group in groups for pid in group]
        return list(set(ids))

    async def get_global_archive_size(self) -> int:
        """Total elites across all islands (fast path via island counts)."""
        tasks = [
            asyncio.create_task(island.__len__()) for island in self.islands.values()
        ]
        sizes = await asyncio.gather(*tasks)
        return sum(int(s) for s in sizes)

    async def remove_program_by_id(self, program_id: str) -> bool:
        """Remove a program (by id) from whichever island holds it and transition to DISCARDED."""
        for island in self.islands.values():
            if await island.archive_storage.remove_elite_by_id(program_id):
                prog = await self.program_storage.get(program_id)
                if prog is not None:
                    if prog.metadata.get(METADATA_KEY_CURRENT_ISLAND):
                        prog.metadata[METADATA_KEY_CURRENT_ISLAND] = None
                        await island.state_manager.update_program(prog)
                    await island.state_manager.set_program_state(
                        prog, ProgramState.DISCARDED
                    )
                return True
        return False

    async def get_metrics(self) -> StrategyMetrics:
        # per-island counts concurrently
        island_ids = list(self.islands.keys())
        counts = await asyncio.gather(*[self.islands[i].__len__() for i in island_ids])
        population_sizes = {f"size/{i}": int(c) for i, c in zip(island_ids, counts)}
        total_programs = sum(population_sizes.values())

        return StrategyMetrics(
            total_programs=total_programs,
            active_populations=len(self.islands),
            strategy_specific_metrics={
                "generation": self.generation,
                "migration_enabled": self.enable_migration,
                "migration_interval": self.migration_interval,
                "max_migrants_per_island": self.max_migrants_per_island,
                "global_max_size": self.max_size,
                **population_sizes,
            },
        )

    def _calculate_island_quotas(self, total: int) -> dict[str, int]:
        """Evenly distribute selection quotas across islands."""
        island_ids = list(self.islands.keys())
        if not island_ids or total <= 0:
            return {}
        base, rem = divmod(total, len(island_ids))
        random.shuffle(island_ids)
        return {
            island_id: base + (1 if i < rem else 0)
            for i, island_id in enumerate(island_ids)
        }

    async def _perform_migration(self) -> None:
        """Migrate elites between islands to improve diversity."""
        logger.info(
            "MultiIsland: starting migration (max_migrants_per_island={})",
            self.max_migrants_per_island,
        )

        # Collect migrants from all islands
        tasks = [
            asyncio.create_task(island.select_migrants(self.max_migrants_per_island))
            for island in self.islands.values()
        ]
        groups = await asyncio.gather(*tasks)
        migrants = [p for g in groups for p in g]

        if not migrants:
            logger.info("MultiIsland: no migrants selected")
            return

        logger.info(
            "MultiIsland: collected {} migrants from {} islands",
            len(migrants),
            len(self.islands),
        )

        # Track migration statistics
        successful_migrations = 0
        failed_migrations = 0
        rollbacks = 0

        random.shuffle(migrants)
        for migrant in migrants:
            source_island_id = migrant.get_metadata("current_island")
            logger.debug(
                "MultiIsland: migrating program {} from island '{}'",
                migrant.id,
                source_island_id,
            )

            candidates = [
                i
                for i in self.islands.values()
                if i.config.island_id != source_island_id
            ]
            if not candidates:
                logger.debug(
                    "MultiIsland: no candidate islands for migrant {} (only 1 island?)",
                    migrant.id,
                )
                failed_migrations += 1
                continue

            destination = await self.mutant_router.route_mutant(migrant, candidates)
            if destination is None:
                logger.debug(
                    "MultiIsland: router returned None for migrant {}",
                    migrant.id,
                )
                failed_migrations += 1
                continue

            logger.debug(
                "MultiIsland: migrant {} routed to island '{}'",
                migrant.id,
                destination.config.island_id,
            )

            if await destination.add(migrant):
                # Successfully added to destination, remove from source
                removed = await self.islands[
                    source_island_id
                ].archive_storage.remove_elite_by_id(migrant.id)

                if not removed:
                    # Rollback: remove from destination to avoid duplicates
                    logger.warning(
                        "MultiIsland: migration rollback for {} (failed to remove from source '{}')",
                        migrant.id,
                        source_island_id,
                    )
                    await destination.archive_storage.remove_elite_by_id(migrant.id)
                    rollbacks += 1
                else:
                    logger.debug(
                        "MultiIsland: migrant {} successfully moved: '{}' -> '{}'",
                        migrant.id,
                        source_island_id,
                        destination.config.island_id,
                    )
                    successful_migrations += 1
            else:
                logger.debug(
                    "MultiIsland: migrant {} rejected by destination island '{}'",
                    migrant.id,
                    destination.config.island_id,
                )
                failed_migrations += 1

        logger.info(
            "MultiIsland: migration complete (success={}, failed={}, rollbacks={})",
            successful_migrations,
            failed_migrations,
            rollbacks,
        )

    async def _enforce_all_island_size_limits(self) -> None:
        """Enforce size limits on all capped islands."""
        tasks = [
            asyncio.create_task(self._enforce_one_island(island))
            for island in self.islands.values()
        ]
        if tasks:
            await asyncio.gather(*tasks)

    async def _enforce_one_island(self, island: MapElitesIsland) -> None:
        await island._enforce_size_limit()
