from __future__ import annotations

import asyncio

from loguru import logger

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.evolution.mutation.base import MutationOperator
from gigaevo.evolution.mutation.parent_selector import ParentSelector
from gigaevo.programs.program import Program


async def generate_mutations(
    elites: list[Program],
    *,
    mutator: MutationOperator,
    storage: ProgramStorage,
    state_manager: ProgramStateManager,
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
        parent_iterator = parent_selector.create_parent_iterator(elites)

        parent_selections = []
        for parents in parent_iterator:
            if len(parent_selections) >= limit:
                break
            parent_selections.append(parents)

        if not parent_selections:
            logger.info("[mutation] No valid parent selections available")
            return 0

        logger.info(
            f"[mutation] Generated {len(parent_selections)} parent selections for parallel mutation"
        )

        async def generate_and_persist_mutation(
            parents: list[Program], task_id: int
        ) -> bool:
            """Generate a single mutation and persist it. Returns True if successful."""
            try:
                mutation_spec = await mutator.mutate_single(parents)

                if mutation_spec is None:
                    return False

                program = Program.from_mutation_spec(mutation_spec)
                program.set_metadata("iteration", iteration)

                await storage.add(program)

                for parent in parents:
                    fresh_parent = await storage.get(parent.id)
                    if fresh_parent:
                        fresh_parent.lineage.add_child(program.id)
                        await state_manager.update_program(fresh_parent)

                return True

            except Exception as exc:
                logger.error(
                    f"[mutation] Task {task_id}: Failed to generate/persist mutation: {exc}"
                )
                return False

        tasks = [
            generate_and_persist_mutation(parents, i)
            for i, parents in enumerate(parent_selections)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        persisted = sum(1 for result in results if result)

        logger.info(
            f"[mutation] Created {persisted} mutations in parallel (immediately persisted)"
        )
        return persisted

    except Exception as exc:  # pragma: no cover
        logger.error(f"[mutation] Mutation generation failed: {exc}.")
        return 0
