"""Helpers for mutation generation and persistence."""

from __future__ import annotations

import asyncio
from typing import List

from loguru import logger

from src.database.program_storage import ProgramStorage
from src.evolution.mutation.base import MutationOperator
from src.evolution.mutation.parent_selector import ParentSelector
from src.programs.program import Program

__all__ = ["generate_mutations"]


async def generate_mutations(
    elites: List[Program],
    *,
    mutator: MutationOperator,
    storage: ProgramStorage,
    parent_selector: ParentSelector,
    limit: int,
    iteration: int,
) -> int:
    """Generate at most *limit* mutations from *elites* and persist them immediately.

    This function now uses parallel execution for efficient mutation generation
    while maintaining proper error handling and respecting the limit.

    Args:
        elites: List of elite programs to use as parents
        mutator: Mutation operator to use for generating mutations
        storage: Storage backend for persisting mutations
        parent_selector: Strategy for selecting parents from elites
        limit: Maximum number of mutations to generate
        iteration: Current iteration number
    Returns:
        Number of persisted mutations.
    """
    if not elites or limit <= 0:
        return 0

    try:
        # Create parent iterator - no need for reset/state management
        parent_iterator = parent_selector.create_parent_iterator(elites)

        # Collect parent selections up to the limit
        parent_selections = []
        for parents in parent_iterator:
            if len(parent_selections) >= limit:
                break
            parent_selections.append(parents)

        if not parent_selections:
            logger.info("[mutation] No valid parent selections available")
            return 0

        logger.info(f"[mutation] Generated {len(parent_selections)} parent selections for parallel mutation")

        # Create mutation tasks for parallel execution
        async def generate_and_persist_mutation(parents: List[Program], task_id: int) -> bool:
            """Generate a single mutation and persist it. Returns True if successful."""
            try:
                mutation_spec = await mutator.mutate_single(parents)

                if mutation_spec is None:
                    logger.debug(f"[mutation] Task {task_id}: Failed to generate mutation")
                    return False

                # Immediately persist the mutation
                program = Program.from_mutation_spec(mutation_spec)
                program.set_metadata("iteration", iteration)
                await storage.add(program)

                logger.debug(f"[mutation] Task {task_id}: Persisted mutation: {mutation_spec.name}")
                return True

            except Exception as exc:
                logger.error(f"[mutation] Task {task_id}: Failed to generate/persist mutation: {exc}")
                return False

        # Execute mutations in parallel
        tasks = [
            generate_and_persist_mutation(parents, i) 
            for i, parents in enumerate(parent_selections)
        ]

        # Use gather with return_exceptions=True to handle individual failures gracefully
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successful mutations
        persisted = sum(1 for result in results if result is True)

        logger.info(f"[mutation] Created {persisted} mutations in parallel (immediately persisted)")
        return persisted

    except Exception as exc:  # pragma: no cover
        logger.error(f"[mutation] Mutation generation failed: {exc}.")
        return 0
