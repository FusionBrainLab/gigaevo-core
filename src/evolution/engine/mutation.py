"""Helpers for mutation generation and persistence."""

from __future__ import annotations

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
) -> int:
    """Generate at most *limit* mutations from *elites* and persist them immediately.

    This function now uses a parent selector for clean parent selection logic,
    and persists mutations one by one as they're generated, preventing
    loss of expensive LLM work when timeouts occur.

    Args:
        elites: List of elite programs to use as parents
        mutator: Mutation operator to use for generating mutations
        storage: Storage backend for persisting mutations
        parent_selector: Strategy for selecting parents from elites
        limit: Maximum number of mutations to generate

    Returns:
        Number of persisted mutations.
    """
    if not elites or limit <= 0:
        return 0

    persisted = 0

    try:
        # Reset the parent selector to start fresh
        parent_selector.reset()

        while persisted < limit:
            try:
                # Generate a single mutation using the parent selector
                mutation_spec = await mutator.mutate_single(
                    elites, parent_selector
                )

                if mutation_spec is None:
                    # No mutation could be generated
                    if not parent_selector.has_more_selections():
                        logger.info(
                            f"[mutation] Parent selector exhausted after {persisted} mutations"
                        )
                        break
                    else:
                        logger.debug(
                            f"[mutation] Failed to generate mutation, but parent selector has more options"
                        )
                        continue

                # Immediately persist the mutation
                program = Program.from_mutation_spec(mutation_spec)
                await storage.add(program)
                persisted += 1

                logger.debug(
                    f"[mutation] Persisted mutation {persisted}/{limit}: {mutation_spec.name}"
                )

                # Check if we should continue (for exhaustive selectors)
                if not parent_selector.has_more_selections():
                    logger.info(
                        f"[mutation] No more parent selections available after {persisted} mutations"
                    )
                    break

            except Exception as exc:
                logger.error(
                    f"[mutation] Failed to generate/persist mutation {persisted + 1}: {exc}"
                )

                # For exhaustive selectors, we might want to continue with remaining combinations
                if parent_selector.has_more_selections():
                    logger.debug(
                        "[mutation] Continuing with next parent selection"
                    )
                    continue
                else:
                    logger.info(
                        "[mutation] No more parent selections available after error"
                    )
                    break

    except Exception as exc:  # pragma: no cover
        logger.error(f"[mutation] Mutation generation failed: {exc}.")

    logger.info(
        f"[mutation] Created {persisted} mutations (immediately persisted)"
    )
    return persisted
