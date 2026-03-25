from __future__ import annotations

import asyncio

from loguru import logger

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.evolution.mutation.base import MutationOperator, MutationSpec
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
) -> list[str]:
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
        List of program IDs for persisted mutations.
    """
    if not elites or limit <= 0:
        return []

    try:
        parent_iterator = parent_selector.create_parent_iterator(elites)

        parent_selections = []
        for parents in parent_iterator:
            if len(parent_selections) >= limit:
                break
            parent_selections.append(parents)

        if not parent_selections:
            logger.info("[mutation] No valid parent selections available")
            return []

        logger.info(
            "[mutation] Generated {} parent selections for parallel mutation",
            len(parent_selections),
        )

        async def generate_and_persist_mutation(
            parents: list[Program], task_id: int
        ) -> str | None:
            """Generate a single mutation and persist it. Returns program ID if successful."""
            try:
                mutation_spec = await mutator.mutate_single(parents)

                if mutation_spec is None:
                    logger.debug(
                        "[mutation] Task {}: mutate_single returned None (parents={})",
                        task_id,
                        [p.short_id for p in parents],
                    )
                    return None

                program = Program.from_mutation_spec(mutation_spec)
                program.set_metadata("iteration", iteration)

                await storage.add(program)
                prompt_id = mutation_spec.metadata.get(MutationSpec.META_PROMPT_ID, "")
                logger.info(
                    "[mutation] Task {}: {} → {} (model={}, archetype={}, prompt_id={})",
                    task_id,
                    [p.short_id for p in parents],
                    program.short_id,
                    mutation_spec.mutation_model or "?",
                    mutation_spec.mutation_archetype or "?",
                    prompt_id or "default",
                )

                for parent in parents:
                    fresh_parent = await storage.get(parent.id)
                    if fresh_parent:
                        fresh_parent.lineage.add_child(program.id)
                        await state_manager.update_program(fresh_parent)

                return program.id

            except Exception as exc:
                logger.error(
                    "[mutation] Task {}: Failed to generate/persist mutation: {}",
                    task_id,
                    exc,
                )
                return None

        tasks = [
            generate_and_persist_mutation(parents, i)
            for i, parents in enumerate(parent_selections)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        mutation_ids = [r for r in results if isinstance(r, str)]

        logger.info(
            "[mutation] Created {} mutations in parallel (immediately persisted)",
            len(mutation_ids),
        )
        return mutation_ids

    except Exception as exc:  # pragma: no cover
        logger.error("[mutation] Mutation generation failed: {}", exc)
        return []
