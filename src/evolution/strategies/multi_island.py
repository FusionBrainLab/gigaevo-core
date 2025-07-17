import random
import time
from typing import Dict, List, Optional, Tuple, Callable, Any
import asyncio

from loguru import logger

from src.database.program_storage import RedisProgramStorage
from src.evolution.strategies.base import (
    EvolutionStrategy,
    StrategyStatus,
)
from src.programs.program import Program
from .island import IslandConfig, MapElitesIsland


class MapElitesMultiIsland(EvolutionStrategy):
    """
    Multi-island MAP-Elites implementation.

    Responsibilities:
    - Maintain multiple behaviorally diverse islands.
    - Route programs to best-matching island.
    - Coordinate inter-island migration.
    """

    def __init__(
        self,
        island_configs: List[IslandConfig],
        program_storage: RedisProgramStorage,
        migration_interval: int = 50,
        enable_migration: bool = True,
        max_migrants_per_island: int = 5,
        should_consider_program_filter: Optional[Callable[[Program], bool]] = None,
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
        self._migration_paused = False
        self.should_consider_program_filter = should_consider_program_filter
        
        # Add rate limiting to prevent Redis connection pool exhaustion
        self._redis_semaphore = asyncio.Semaphore(len(island_configs) * 2)  # Allow 2x concurrent ops per island
        
        # OPTIMIZATION: Add caching for island sizes and elite counts
        self._island_size_cache: Dict[str, Tuple[int, float]] = {}  # {island_id: (size, timestamp)}
        self._cache_ttl = 10.0  # Cache valid for 10 seconds
        self._cache_lock = asyncio.Lock()
        
        # Calculate global max_size, handling None values properly
        total_max_size = 0
        has_size_limits = False
        for cfg in island_configs:
            if cfg.max_size is not None:
                total_max_size += cfg.max_size
                has_size_limits = True
        
        self.max_size = total_max_size if has_size_limits else None

        logger.info(f"Initialized MAP-Elites with {len(self.islands)} islands, global max_size={self.max_size}")

    async def _rate_limited_operation(self, operation_name: str, operation_func):
        """Execute a Redis operation with rate limiting."""
        async with self._redis_semaphore:
            try:
                return await operation_func()
            except Exception as e:
                if "too many connections" in str(e).lower():
                    logger.warning(f"Connection pool exhausted during {operation_name}, retrying after delay...")
                    await asyncio.sleep(0.5)
                    return await operation_func()
                raise

    async def add(self, program: Program, island_id: Optional[str] = None) -> bool:
        """Add a program to the best-matching island (or specific one)."""
        if self.should_consider_program_filter and not self.should_consider_program_filter(program):
            logger.debug(f"Program {program.id} filtered out by user-defined criteria")
            return False

        island = (
            self.islands.get(island_id)
            if island_id and island_id in self.islands
            else await self._select_best_island_for_program(program)
        )

        if island is None:
            logger.debug(f"Program {program.id} rejected — no compatible island found")
            return False

        try:
            accepted = await island.add(program)
            if accepted:
                # OPTIMIZATION: Invalidate cache when program is successfully added
                self._invalidate_island_cache(island.config.island_id)
                logger.debug(f"Program {program.id} accepted by island {island.config.island_id}")
            else:
                logger.debug(f"Program {program.id} rejected by island {island.config.island_id} (no improvement)")
            return accepted
        except Exception as e:
            logger.warning(f"Failed to add program {program.id} to island {island.config.island_id}: {e}")
            return False

    async def _select_best_island_for_program(self, program: Program) -> Optional[MapElitesIsland]:
        """Select the best island for a program with efficient caching and batch operations."""
        accepting_islands: List[Tuple[MapElitesIsland, str]] = []

        # First pass: check which islands can accept the program
        for island in self.islands.values():
            try:
                required_keys = set(island.config.behavior_space.behavior_keys)
                if not required_keys.issubset(program.metrics.keys()):
                    continue

                cell = island.config.behavior_space.get_cell(program.metrics)
                cell_key = island._cell_key(cell)
                elite = await island.archive_storage.get_elite(cell_key)

                if elite is None:
                    accepting_islands.append((island, "empty_cell"))
                elif island.config.archive_selector(program, elite):
                    accepting_islands.append((island, "improvement"))

            except Exception as e:
                logger.warning(f"Error evaluating island {island.config.island_id} for program {program.id}: {e}")

        if not accepting_islands:
            return None

        # Second pass: get sizes for accepting islands and calculate weights
        island_info = []  # List of (island, reason, size, weight)
        
        for island, reason in accepting_islands:
            size = await self._get_cached_island_size(island.config.island_id)
            island_info.append((island, reason, size))

        # Calculate weights inversely proportional to island size
        # Smaller islands get higher weight, but with bounds to prevent extreme bias
        total_sizes = sum(info[2] for info in island_info)
        avg_size = total_sizes / len(island_info) if island_info else 1
        
        weighted_islands = []
        weights = []
        
        for island, reason, size in island_info:
            # Inverse weight with floor to prevent division by zero and extreme bias
            # Prefer empty cells over improvements
            reason_multiplier = 2.0 if reason == "empty_cell" else 1.0
            
            # Inverse size weight: smaller islands get higher probability
            # Use max(size, 1) to avoid division by zero
            # Add avg_size to prevent extreme bias towards very small islands
            size_weight = (avg_size + 1) / (size + 1)
            
            # Additional penalty for islands close to capacity
            if island.config.max_size is not None:
                capacity_ratio = size / island.config.max_size
                if capacity_ratio > 0.8:  # Penalize islands that are >80% full
                    capacity_penalty = 1.0 - (capacity_ratio - 0.8) * 5.0  # Reduce weight by up to 100%
                    size_weight *= max(capacity_penalty, 0.1)  # Minimum 10% weight
            
            final_weight = size_weight * reason_multiplier
            
            weighted_islands.append(island)
            weights.append(final_weight)
            
            logger.debug(f"Island {island.config.island_id}: size={size}, reason={reason}, weight={final_weight:.3f}")

        # Weighted random selection
        if weighted_islands:
            chosen_island = random.choices(weighted_islands, weights=weights, k=1)[0]
            logger.debug(f"Selected island {chosen_island.config.island_id} via weighted random selection")
            return chosen_island
        
        return None

    async def select_elites(self, total: int = 10) -> List[Program]:
        """Sample elites from all islands (with optional migration)."""
        if (
            self.enable_migration and
            not self._migration_paused and
            self.generation - self.last_migration >= self.migration_interval
        ):
            await self._perform_migration()
            self.last_migration = self.generation

        # Periodic size enforcement every 10 generations
        if self.generation % 10 == 0:
            await self._enforce_all_island_size_limits()

        all_elites: List[Program] = []
        quotas = self._calculate_island_quotas(total)

        for island_id, quota in quotas.items():
            try:
                if quota > 0:
                    selected = await self.islands[island_id].select_elites(quota)
                    all_elites.extend(selected)
            except Exception as e:
                logger.warning(f"Failed to select elites from island {island_id}: {e}")

        if all_elites:
            self.generation += 1

        return all_elites[:total]

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

        migration_tasks = [
            self._collect_migrants_from_island(island_id, self.islands[island_id])
            for island_id in island_ids
        ]
        migrant_batches = await asyncio.gather(*migration_tasks)

        all_migrants = [m for batch in migrant_batches if batch for m in batch[1]]

        if not all_migrants:
            logger.info("No migrants available for migration")
            return

        logger.info(f"Migrating {len(all_migrants)} programs")

        successful, failed = 0, 0
        for migrant in all_migrants:
            source_island = migrant.metadata.get("current_island")
            destination = await self._select_best_island_for_program(migrant)

            if not destination or destination.config.island_id == source_island:
                failed += 1
                continue

            try:
                accepted = await destination.add(migrant)
                if accepted:
                    # Remove from source island if different
                    if source_island and source_island != destination.config.island_id:
                        try:
                            source_island_obj = self.islands.get(source_island)
                            if source_island_obj:
                                await source_island_obj.archive_storage.remove_elite_by_id(migrant.id)
                        except Exception as exc:
                            logger.warning(f"Failed to remove program {migrant.id} from source island {source_island}: {exc}")
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logger.warning(f"Migration failed for program {migrant.id}: {e}")
                failed += 1

        logger.info(f"Migration complete: {successful} succeeded, {failed} failed")
        
        # Post-migration size enforcement to ensure no islands exceed limits
        await self._enforce_all_island_size_limits()

    async def _enforce_all_island_size_limits(self) -> None:
        """Enforce size limits on all islands after migration."""
        for island_id, island in self.islands.items():
            if island.config.max_size is not None:
                try:
                    current_count = await island.get_elite_count()
                    if current_count > island.config.max_size:
                        logger.warning(f"Enforcing size limit on island {island_id}: {current_count} > {island.config.max_size}")
                        await island.enforce_size_limit()
                except Exception as e:
                    logger.error(f"Failed to enforce size limit on island {island_id}: {e}")

    async def _collect_migrants_from_island(
        self, island_id: str, island: MapElitesIsland
    ) -> Optional[Tuple[str, List[Program]]]:
        try:
            migrants = await island.select_migrants(self.max_migrants_per_island)
            return (island_id, migrants)
        except Exception as e:
            logger.error(f"Error collecting migrants from {island_id}: {e}")
            return None

    async def get_global_archive_size(self) -> int:
        """Get total number of elites across all islands efficiently."""
        async def _get_size():
            size_tasks = [island.get_elite_count() for island in self.islands.values()]
            sizes = await asyncio.gather(*size_tasks, return_exceptions=True)
            
            total_size = 0
            for size in sizes:
                if isinstance(size, Exception):
                    logger.warning(f"Error getting elite count from island: {size}")
                    continue
                total_size += size
            
            return total_size
        
        return await self._rate_limited_operation("get_global_archive_size", _get_size)

    async def get_program_ids(self) -> List[Program]:
        """Get all programs across all islands (for compatibility with EvolutionEngine)."""
        async def _get_programs():
            # Stagger the operations to reduce concurrent load
            all_programs = []
            for island in self.islands.values():
                try:
                    programs = await island.get_all_elites()
                    all_programs.extend(programs)
                    # Small delay between islands to reduce concurrent load
                    await asyncio.sleep(0.01)
                except Exception as e:
                    logger.warning(f"Error getting programs from island {island.config.island_id}: {e}")
            
            return all_programs
        
        return await self._rate_limited_operation("get_program_ids", _get_programs)

    async def get_status(self) -> StrategyStatus:
        """Get comprehensive MAP-Elites strategy status."""
        try:
            metrics = await self.get_metrics()

            is_healthy = True
            health_issues = []

            if metrics.active_populations == 0:
                is_healthy = False
                health_issues.append("No active islands")

            if metrics.total_programs == 0:
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
                error_message=("; ".join(health_issues) if health_issues else None),
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

    async def pause(self) -> None:
        """Pause MAP-Elites operations."""
        self._migration_paused = True
        logger.info("[MapElitesMultiIsland] Migration paused")

    async def resume(self) -> None:
        """Resume MAP-Elites operations."""
        self._migration_paused = False
        logger.info("[MapElitesMultiIsland] Migration resumed")

    async def log_status(self) -> None:
        """Log periodic status summary with QD metrics per island."""
        try:
            logger.info(f"=== Multi-Island MAP-Elites Status (Generation {self.generation}) ===")
            
            total_programs = 0
            total_qd_score = 0
            
            for island_id, island in self.islands.items():
                try:
                    elite_count = await island.get_elite_count()
                    total_programs += elite_count
                    
                    # Check for overpopulation
                    max_size = island.config.max_size
                    if max_size is not None and elite_count > max_size:
                        logger.error(f"🚨 Island {island_id} OVERPOPULATED: {elite_count} elites (max: {max_size})")
                        # Force size enforcement
                        try:
                            await island.enforce_size_limit()
                            logger.info(f"✅ Enforced size limit for island {island_id}")
                        except Exception as e:
                            logger.error(f"❌ Failed to enforce size limit for island {island_id}: {e}")
                    
                    try:
                        qd_metrics = await island.compute_qd_metrics()
                        total_qd_score += qd_metrics.qd_score
                        
                        size_status = f"{elite_count}"
                        if max_size is not None:
                            utilization = (elite_count / max_size) * 100
                            size_status += f"/{max_size} ({utilization:.1f}%)"
                            if utilization > 100:
                                size_status += " ⚠️ OVER"
                        
                        logger.info(
                            f"Island {island_id}: {size_status} elites, "
                            f"QD={qd_metrics.qd_score:.2f}, "
                            f"Coverage={qd_metrics.coverage:.3f}, "
                            f"Max fitness={qd_metrics.maximum_fitness:.2f}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to compute QD metrics for island {island_id}: {e}")
                        logger.info(f"Island {island_id}: {elite_count} elites (QD metrics unavailable)")
                        
                except Exception as e:
                    logger.error(f"Failed to get status for island {island_id}: {e}")
            
            size_info = f"Total: {total_programs} programs"
            if self.max_size:
                utilization = (total_programs / self.max_size) * 100
                size_info += f" ({utilization:.1f}% of global max_size)"
            
            logger.info(f"{size_info}, Global QD={total_qd_score:.2f}")
            
            # Check for severely imbalanced islands
            if len(self.islands) > 1:
                elite_counts = []
                for island in self.islands.values():
                    try:
                        count = await island.get_elite_count()
                        elite_counts.append(count)
                    except Exception as e:
                        logger.warning(f"Failed to get elite count for imbalance check: {e}")
                
                if elite_counts:
                    min_count = min(elite_counts)
                    max_count = max(elite_counts)
                    if min_count > 0 and max_count / min_count > 3:
                        logger.warning(f"⚠️ Island size imbalance detected: min={min_count}, max={max_count}")
            
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Error logging status: {e}")

    async def get_island_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed statistics for each island."""
        async def _get_stats():
            stats = {}
            
            for island_id, island in self.islands.items():
                try:
                    elite_count = await island.get_elite_count()
                    
                    island_stats = {
                        'elite_count': elite_count,
                        'max_size': island.config.max_size,
                        'is_overpopulated': False,
                        'utilization_percent': 0.0
                    }
                    
                    if island.config.max_size is not None:
                        island_stats['utilization_percent'] = (elite_count / island.config.max_size) * 100
                        island_stats['is_overpopulated'] = elite_count > island.config.max_size
                    
                    try:
                        qd_metrics = await island.compute_qd_metrics()
                        island_stats.update({
                            'qd_score': qd_metrics.qd_score,
                            'coverage': qd_metrics.coverage,
                            'max_fitness': qd_metrics.maximum_fitness,
                            'avg_fitness': qd_metrics.average_fitness,
                            'filled_cells': qd_metrics.filled_cells,
                            'total_cells': qd_metrics.total_cells
                        })
                    except Exception as e:
                        logger.warning(f"Failed to get QD metrics for island {island_id}: {e}")
                    
                    stats[island_id] = island_stats
                    
                    # Small delay between islands
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    logger.error(f"Failed to get statistics for island {island_id}: {e}")
                    stats[island_id] = {'error': str(e)}
            
            return stats
        
        return await self._rate_limited_operation("get_island_statistics", _get_stats)

    async def _get_cached_island_size(self, island_id: str) -> int:
        """Get island size with caching to reduce Redis calls."""
        current_time = time.time()
        
        async with self._cache_lock:
            if island_id in self._island_size_cache:
                size, timestamp = self._island_size_cache[island_id]
                if current_time - timestamp < self._cache_ttl:
                    return size
            
            # Cache miss or expired - fetch fresh data
            try:
                island = self.islands[island_id]
                size = await island.get_elite_count()
                self._island_size_cache[island_id] = (size, current_time)
                return size
            except Exception as e:
                logger.warning(f"Error getting size for island {island_id}: {e}")
                return float("inf")  # Penalize islands with errors

    def _invalidate_island_cache(self, island_id: str):
        """Invalidate cache entry for a specific island."""
        if island_id in self._island_size_cache:
            del self._island_size_cache[island_id] 