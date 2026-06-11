"""Shared utilities for chain evolution problems."""

from __future__ import annotations

import asyncio

from gigaevo.database.factory import RedisRunConfig, build_readonly_redis_storage


def get_best_program(
    config: RedisRunConfig,
    fitness_col: str = "metric_fitness",
    minimize: bool = False,
) -> dict | None:
    """Extract the best program from Redis by fitness.

    Args:
        config: Redis connection config
        fitness_col: Name of the fitness metric column
        minimize: If True, lower fitness is better

    Returns:
        Dict with program info: {id, code, fitness, metrics, metadata} or None if no programs
    """

    async def _fetch():
        storage = build_readonly_redis_storage(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            key_prefix=config.redis_prefix,
        )
        async with storage:
            return await storage.get_all()

    programs = asyncio.run(_fetch())

    if not programs:
        return None

    # Filter to programs with valid fitness
    metric_name = fitness_col.replace("metric_", "")
    valid_programs = [
        p
        for p in programs
        if metric_name in p.metrics and p.metrics[metric_name] is not None
    ]

    if not valid_programs:
        return None

    # Find best by fitness
    if minimize:
        best = min(valid_programs, key=lambda p: p.metrics[metric_name])
    else:
        best = max(valid_programs, key=lambda p: p.metrics[metric_name])

    return {
        "id": best.id,
        "code": best.code,
        "fitness": best.metrics[metric_name],
        "metrics": best.metrics,
        "metadata": best.metadata,
    }
