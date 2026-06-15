from __future__ import annotations

from loguru import logger

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.evolution.mutation.base import MutationOperator, MutationSpec
from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_INJECTED_IDS_METADATA_KEY,
    MUTATION_MEMORY_LINEAGE_APPLIED_IDS_METADATA_KEY,
    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
    MUTATION_MEMORY_USED_METADATA_KEY,
    MUTATION_PARENT_STAGE_OUTPUTS_METADATA_KEY,
)
from gigaevo.evolution.mutation.parent_selector import ParentSelector
from gigaevo.evolution.mutation.parent_snapshot import snapshot_parent_stage_outputs
from gigaevo.programs.program import Program


def lineage_applied_closure(
    *, injected_ids: list[str], parents: list[Program]
) -> list[str]:
    """Transitive closure of every card applied to this child or any ancestor.

    Built from frozen inputs only: the child's just-computed ``injected_ids``
    unioned with each parent's own (birth-frozen) lineage-applied closure.
    """
    closure: set[str] = {cid for cid in injected_ids if cid}
    for parent in parents:
        for cid in (
            parent.get_metadata(MUTATION_MEMORY_LINEAGE_APPLIED_IDS_METADATA_KEY) or []
        ):
            if cid:
                closure.add(cid)
    return sorted(closure)


async def generate_one_mutation(
    parents: list[Program],
    *,
    mutator: MutationOperator,
    storage: ProgramStorage,
    state_manager: ProgramStateManager,
    iteration: int,
    task_id: int = 0,
) -> str | None:
    """Generate a single mutation and persist it. Returns program ID if successful.

    Runs inline — no ``asyncio.gather`` wrapping. This is the primitive
    invoked by ``mutant_task.run_one_mutant`` (which always wants exactly
    one mutant per call). Keeping the call path linear means that a
    ``CancelledError`` raised in any inner ``await`` is caught by the
    local ``except BaseException`` arm, which can return ``persisted_id``
    directly to the caller — no outer ``gather`` exists to swallow the
    return value.

    Once ``storage.add()`` succeeds the program exists in Redis. Any
    failure after that point (including ``asyncio.CancelledError``, which
    is a ``BaseException``) must still return the program ID so the engine
    can track it — otherwise the program becomes an orphan ghost.

    Memory usage is auto-derived from parent metadata: if any parent has
    ``MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY`` set by the DAG-based
    MemoryContextStage, the child is marked ``memory_used=True``.
    """
    persisted_id: str | None = None
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
        program.iteration = iteration

        # Freeze the birth-time card slate: parents' selected-ids metadata is
        # overwritten on every NO_CACHE requeue, so attribution must read the
        # child's stamp, never the parents' current state.
        injected_ids = sorted(
            {
                card_id
                for parent in parents
                for card_id in (
                    parent.get_metadata(MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY) or []
                )
                if card_id
            }
        )
        program.set_metadata(MUTATION_MEMORY_INJECTED_IDS_METADATA_KEY, injected_ids)
        program.set_metadata(MUTATION_MEMORY_USED_METADATA_KEY, bool(injected_ids))
        program.set_metadata(
            MUTATION_MEMORY_LINEAGE_APPLIED_IDS_METADATA_KEY,
            lineage_applied_closure(injected_ids=injected_ids, parents=parents),
        )

        # Freeze the parent stage outputs that produced this child (debug only —
        # must never block the mutation, so failures are swallowed).
        try:
            stage_outputs = await snapshot_parent_stage_outputs(parents, storage)
            if stage_outputs:
                program.set_metadata(
                    MUTATION_PARENT_STAGE_OUTPUTS_METADATA_KEY, stage_outputs
                )
        except Exception as snap_exc:
            logger.warning(
                "[mutation] Task {}: stage-output snapshot failed (non-critical): {}",
                task_id,
                snap_exc,
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
    """Generate at most *limit* mutations from *elites* and persist them.

    Batch wrapper around :func:`generate_one_mutation`. Each mutant is
    produced sequentially — the steady-state engine never calls this with
    ``limit > 1`` (see ``mutant_task.run_one_mutant``), so the loss of
    intra-batch parallelism is irrelevant for production. Tests that use
    ``limit > 1`` exercise correctness, not throughput.

    Sequential dispatch is deliberate: the prior ``asyncio.gather`` shape
    discarded already-persisted IDs whenever the outer awaiter was
    cancelled (the "ghost-persist" race), because gather re-raises
    ``CancelledError`` to its caller before the caller can read the
    children's return values. Sequential calls let each mutant's
    ``except BaseException`` handler return the persisted ID directly to
    the caller.

    Returns:
        List of program IDs for persisted mutations.
    """
    if not elites or limit <= 0:
        return []

    try:
        parent_iterator = parent_selector.create_parent_iterator(elites)

        parent_selections: list[list[Program]] = []
        for parents in parent_iterator:
            if len(parent_selections) >= limit:
                break
            parent_selections.append(parents)

        if not parent_selections:
            logger.info("[mutation] No valid parent selections available")
            return []

        logger.info(
            "[mutation] Generated {} parent selections for sequential mutation",
            len(parent_selections),
        )

        mutation_ids: list[str] = []
        for i, parents in enumerate(parent_selections):
            try:
                mid = await generate_one_mutation(
                    parents,
                    mutator=mutator,
                    storage=storage,
                    state_manager=state_manager,
                    iteration=iteration,
                    task_id=i,
                )
            except BaseException as exc:
                # ``generate_one_mutation`` only re-raises CancelledError
                # when no program was persisted (pre-persist cancel). Treat
                # any propagated exception as "batch interrupted" — return
                # whatever IDs were already persisted rather than dropping
                # them on the floor.
                logger.warning(
                    "[mutation] Batch interrupted by {} on item {} of {}: "
                    "returning {} ids accumulated so far",
                    type(exc).__name__,
                    i,
                    len(parent_selections),
                    len(mutation_ids),
                )
                break
            if mid is not None:
                mutation_ids.append(mid)

        logger.info(
            "[mutation] Created {} mutations (immediately persisted)",
            len(mutation_ids),
        )
        return mutation_ids

    except Exception as exc:  # pragma: no cover
        logger.error("[mutation] Mutation generation failed: {}", exc)
        return []
