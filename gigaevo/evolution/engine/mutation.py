from __future__ import annotations

import asyncio

from loguru import logger

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.evolution.mutation.context import MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY
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
    memory_instructions: str | None = None,
    memory_used: bool = False,
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
        memory_instructions: Optional memory string to guide mutation
        memory_used: Whether to mark resulting programs as memory-based mutations
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
            """Generate a single mutation and persist it. Returns program ID if successful.

            Once ``storage.add()`` succeeds the program exists in Redis. Any
            failure after that point (including ``asyncio.CancelledError``, which
            is a ``BaseException``) must still return the program ID so the engine
            can track it — otherwise the program becomes an orphan ghost.
            """
            persisted_id: str | None = None
            try:
                mutation_spec = await mutator.mutate_single(
                    parents, memory_instructions=memory_instructions
                )

                if mutation_spec is None:
                    logger.debug(
                        "[mutation] Task {}: mutate_single returned None (parents={})",
                        task_id,
                        [p.short_id for p in parents],
                    )
                    return None

                program = Program.from_mutation_spec(mutation_spec)
                program.set_metadata("iteration", iteration)
                program.set_metadata("memory_used", bool(memory_used))
                selected_ids = program.get_metadata(
                    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY
                )
                if not isinstance(selected_ids, list):
                    selected_ids = []
                program.set_metadata(
                    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY, selected_ids
                )

                await storage.add(program)
                persisted_id = program.id  # Point of no return — ID must be returned

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

                # Update parent lineages. Failures here are non-critical — the
                # program is already persisted and will be evaluated by DagRunner.
                # If parent no longer exists or lineage update fails, we still
                # return the program ID so steady-state engine can track it.
                try:
                    for parent in parents:
                        fresh_parent = await storage.get(parent.id)
                        if fresh_parent:
                            fresh_parent.lineage.add_child(program.id)
                            await state_manager.update_program(fresh_parent)
                except Exception as lineage_exc:
                    logger.warning(
                        "[mutation] Task {}: Lineage update failed (program {} still valid): {}",
                        task_id,
                        program.short_id,
                        lineage_exc,
                    )

                return program.id

            except BaseException as exc:
                if persisted_id is not None:
                    # Program is in Redis — return its ID to prevent orphan.
                    logger.warning(
                        "[mutation] Task {}: post-persist {} ({}), returning ID to avoid orphan",
                        task_id,
                        type(exc).__name__,
                        persisted_id[:8],
                    )
                    return persisted_id
                # Not yet persisted — safe to handle normally.
                if isinstance(exc, Exception):
                    logger.error(
                        "[mutation] Task {}: Failed to generate/persist mutation: {}",
                        task_id,
                        exc,
                    )
                    return None
                raise  # CancelledError before persist — propagate

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
