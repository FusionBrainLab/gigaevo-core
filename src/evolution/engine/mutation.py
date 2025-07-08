"""Helpers for mutation generation and persistence."""
from __future__ import annotations

import asyncio
from typing import List

from loguru import logger

from src.programs.program import Program
from src.evolution.mutation.base import MutationOperator
from src.database.program_storage import ProgramStorage

__all__ = ["generate_mutations"]


async def _produce_children(mutator: MutationOperator, elites: List[Program]) -> List[Program]:
    return [Program.from_mutation_spec(spec) for spec in await mutator.mutate_batch(elites)]


async def generate_mutations(
    elites: List[Program],
    *,
    mutator: MutationOperator,
    storage: ProgramStorage,
    limit: int,
) -> int:
    """Generate at most *limit* mutations from *elites* and persist them.

    Returns number of persisted mutations.
    """
    if not elites or limit <= 0:
        return 0

    try:
        children = await _produce_children(mutator, elites)
    except Exception as exc:  # pragma: no cover
        logger.error(f"[mutation] mutate_batch failed: {exc}.")
        return 0

    children = children[:limit]

    persisted = 0
    for spec in children:
        try:
            await storage.add(spec)  # type: ignore[arg-type]
            persisted += 1
        except Exception as exc:  # pragma: no cover
            logger.error(f"[mutation] Failed to persist mutation: {exc}")

    logger.info(f"[mutation] Created {persisted} mutations")
    return persisted 