import random
from typing import Dict, List, Optional

from loguru import logger

from src.database.redis_program_storage import RedisProgramStorage
from src.evolution.strategies.base import (
    EvolutionStrategy,
    StrategyMetrics,
    StrategyStatus,
)
from src.programs.program import Program

from .island import IslandConfig, MapElitesIsland
from .island_selector import WeightedIslandSelector
from .mutant_router import RandomMutantRouter


class MapElitesMultiIsland(EvolutionStrategy):
    """
    Multi-island MAP-Elites implementation.
    """

    def __init__(
        self,
        island_configs: List[IslandConfig],
        program_storage: RedisProgramStorage,
        migration_interval: int = 50,
        enable_migration: bool = True,
        max_migrants_per_island: int = 5,
        island_selector: Optional[WeightedIslandSelector] = None,
        mutant_router: Optional[RandomMutantRouter] = None,
    ):
        if not island_configs:
            raise ValueError("At least one island configuration is required")

        self.islands: Dict[str, MapElitesIsland] = {
            cfg.island_id: MapElitesIsland(cfg, program_storage)
            for cfg in island_configs
        }

        self.program_storage = program_storage
        self.migration_interval = migration_interval
        self.enable_migration = enable_migration
        self.max_migrants_per_island = max_migrants_per_island
        self.generation = 0
        self.last_migration = 0

        # Initialize selectors and routers
        self.island_selector = island_selector or WeightedIslandSelector()
        self.mutant_router = mutant_router or RandomMutantRouter()

        # Calculate global max_size
        total_max_size = 0
        has_size_limits = False
        for cfg in island_configs:
            if cfg.max_size is not None:
                total_max_size += cfg.max_size
                has_size_limits = True

        self.max_size = total_max_size if has_size_limits else None

        logger.info(
            f"Initialized MAP-Elites with {len(self.islands)} islands, global max_size={self.max_size}"
        )

    async def add(
        self, program: Program, island_id: Optional[str] = None
    ) -> bool:
        """Add a program to the best-matching island (or specific one)."""
        # Manual island assignment
        if island_id is not None and island_id not in self.islands:
            logger.debug(
                f"Program {program.id} rejected — invalid island_id: {island_id}"
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
                f"Program {program.id} rejected — no compatible island found"
            )
            return False

        try:
            accepted = await island.add(program)
            if accepted:
                logger.debug(
                    f"Program {program.id} accepted by island {island.config.island_id}"
                )
            else:
                logger.debug(
                    f"Program {program.id} rejected by island {island.config.island_id}"
                )
            return accepted
        except Exception as e:
            logger.warning(
                f"Failed to add program {program.id} to island {island.config.island_id}: {e}"
            )
            return False

    async def select_elites(self, total: int = 10) -> List[Program]:
        """Sample elites from all islands (with optional migration)."""
        if (
            self.enable_migration
            and self.generation - self.last_migration >= self.migration_interval
        ):
            await self._perform_migration()
            self.last_migration = self.generation
            await self._enforce_all_island_size_limits()
        if self.generation % 10 == 0:
            await self._enforce_all_island_size_limits()

        # FIXED: Collect all candidates with their island info, then fairly select
        island_candidates = []
        quotas = self._calculate_island_quotas(total)

        for island_id, quota in quotas.items():
            try:
                if quota > 0:
                    selected = await self.islands[island_id].select_elites(
                        quota
                    )
                    island_candidates.append((island_id, selected))
            except Exception as e:
                logger.warning(
                    f"Failed to select elites from island {island_id}: {e}"
                )

        # FIXED: Randomize order to prevent bias toward specific islands
        random.shuffle(island_candidates)

        all_elites: List[Program] = []
        for island_id, selected in island_candidates:
            all_elites.extend(selected)

        # FIXED: If we have more than requested, fairly sample rather than truncate
        if len(all_elites) > total:
            all_elites = random.sample(all_elites, total)

        if all_elites:
            self.generation += 1

        return all_elites

    def _calculate_island_quotas(self, total: int) -> Dict[str, int]:
        """Evenly distribute elite selection across islands."""
        if not self.islands:
            return {}
        island_ids = list(self.islands.keys())
        base = total // len(island_ids)
        rem = total % len(island_ids)
        random.shuffle(island_ids)
        return {
            island_id: base + (1 if i < rem else 0)
            for i, island_id in enumerate(island_ids)
        }

    async def _perform_migration(self) -> None:
        """Migrate best elites across islands to improve diversity."""
        logger.info("Starting migration round")
        island_ids = list(self.islands.keys())
        random.shuffle(island_ids)

        all_migrants = []
        for island_id in island_ids:
            try:
                migrants = await self.islands[island_id].select_migrants(
                    self.max_migrants_per_island
                )
                all_migrants.extend(migrants)
            except Exception as e:
                logger.error(f"Error collecting migrants from {island_id}: {e}")

        if not all_migrants:
            logger.info("No migrants available for migration")
            return

        logger.info(f"Migrating {len(all_migrants)} programs")

        successful, failed = 0, 0
        # FIXED: Shuffle migrants to prevent processing order bias
        random.shuffle(all_migrants)

        for migrant in all_migrants:
            source_island = migrant.metadata.get("current_island")

            # FIXED: Handle missing current_island metadata
            if not source_island:
                logger.warning(
                    f"Migrant {migrant.id} has no current_island metadata, skipping migration"
                )
                failed += 1
                continue

            # Get all islands except the source island
            available_islands = [
                island
                for island in self.islands.values()
                if island.config.island_id != source_island
            ]

            if not available_islands:
                logger.debug(
                    f"No available destination islands for migrant {migrant.id} from {source_island}"
                )
                failed += 1
                continue

            # Route to a different island
            destination = await self.mutant_router.route_mutant(
                migrant, available_islands
            )

            if not destination:
                logger.debug(
                    f"No compatible destination island found for migrant {migrant.id}"
                )
                failed += 1
                continue

            try:
                accepted = await destination.add(migrant)
                if accepted:
                    source_island_obj = self.islands[source_island]
                    removal_success = await source_island_obj.archive_storage.remove_elite_by_id(
                        migrant.id
                    )

                    if removal_success:
                        successful += 1
                        logger.debug(
                            f"Successfully migrated program {migrant.id} from {source_island} to {destination.config.island_id}"
                        )
                    else:
                        logger.error(
                            f"CRITICAL: Program {migrant.id} added to {destination.config.island_id} but failed to remove from {source_island} - potential duplicate!"
                        )
                        failed += 1
                        try:
                            await destination.archive_storage.remove_elite_by_id(
                                migrant.id
                            )
                            logger.info(
                                f"Cleaned up duplicate program {migrant.id} from destination island {destination.config.island_id}"
                            )
                        except Exception as cleanup_exc:
                            logger.error(
                                f"Failed to cleanup duplicate program {migrant.id}: {cleanup_exc}"
                            )
                else:
                    failed += 1
                    logger.debug(
                        f"Destination island {destination.config.island_id} rejected migrant {migrant.id}"
                    )
            except Exception as e:
                logger.warning(
                    f"Migration failed for program {migrant.id}: {e}"
                )
                failed += 1

        logger.info(
            f"Migration complete: {successful} succeeded, {failed} failed"
        )

    async def _enforce_all_island_size_limits(self) -> None:
        """Enforce size limits on all islands after migration."""
        violations_found = False

        for island_id, island in self.islands.items():
            if island.config.max_size is None:
                continue
            try:
                current_count = await island.get_elite_count()
                if current_count > island.config.max_size:
                    violations_found = True
                    logger.warning(
                        f"Enforcing size limit on island {island_id}: {current_count} > {island.config.max_size}"
                    )
                    await island.enforce_size_limit()

                    # Verify enforcement worked
                    post_enforcement_count = await island.get_elite_count()
                    if post_enforcement_count > island.config.max_size:
                        logger.error(
                            f"CRITICAL: Size enforcement failed for island {island_id}! "
                            f"Still has {post_enforcement_count} > {island.config.max_size} after enforcement"
                        )
                else:
                    logger.debug(
                        f"Island {island_id} size OK: {current_count}/{island.config.max_size}"
                    )
            except Exception as e:
                logger.error(
                    f"Failed to enforce size limit on island {island_id}: {e}"
                )

        if not violations_found:
            logger.debug("All island size limits are within bounds")

    async def get_global_archive_size(self) -> int:
        """Get total number of elites across all islands."""
        total_size = 0
        for island in self.islands.values():
            try:
                programs = await island.get_all_elites()
                total_size += len(programs)
            except Exception as e:
                logger.warning(
                    f"Error getting elite count from island {island.config.island_id}: {e}"
                )

        return total_size

    async def get_program_ids(self) -> List[Program]:
        """Get all programs across all islands."""
        all_programs = []
        for island in self.islands.values():
            try:
                programs = await island.get_all_elites()
                all_programs.extend(programs)
            except Exception as e:
                logger.warning(
                    f"Error getting programs from island {island.config.island_id}: {e}"
                )

        return all_programs

    async def get_metrics(self) -> Optional[StrategyMetrics]:
        """Get multi-island strategy metrics."""
        try:
            total_programs = await self.get_global_archive_size()
            active_populations = len(self.islands)

            return StrategyMetrics(
                total_programs=total_programs,
                active_populations=active_populations,
                strategy_specific_metrics={
                    "generation": self.generation,
                    "migration_enabled": self.enable_migration,
                    "migration_interval": self.migration_interval,
                    "max_migrants_per_island": self.max_migrants_per_island,
                    "global_max_size": self.max_size,
                },
            )
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return None

    async def get_status(self) -> StrategyStatus:
        """Get MAP-Elites strategy status."""
        try:
            metrics = await self.get_metrics()

            is_healthy = True
            health_issues = []

            if metrics and metrics.active_populations == 0:
                is_healthy = False
                health_issues.append("No active islands")

            if metrics and metrics.total_programs == 0:
                health_issues.append("No programs in archive")

            status_details = {
                "health_issues": health_issues,
                "capabilities": {
                    "migration": self.enable_migration,
                },
            }

            return StrategyStatus(
                strategy_type="MapElitesMultiIsland",
                is_healthy=is_healthy,
                error_message=(
                    "; ".join(health_issues) if health_issues else None
                ),
                metrics=metrics,
                status_details=status_details,
            )

        except Exception as e:
            return StrategyStatus(
                strategy_type="MapElitesMultiIsland",
                is_healthy=False,
                error_message=f"Status check failed: {e}",
            )

    async def cleanup(self) -> None:
        """Perform cleanup operations."""
        logger.info("[MapElitesMultiIsland] Cleanup completed")
