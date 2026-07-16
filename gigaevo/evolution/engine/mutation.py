from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from loguru import logger

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.evolution.mutation.base import MutationOperator, MutationSpec
from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_ASSIGNMENT_METADATA_KEY,
    MUTATION_MEMORY_BASE_ID_METADATA_KEY,
    MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
    MUTATION_MEMORY_BASE_SCORE_SIGNATURE_METADATA_KEY,
    MUTATION_MEMORY_BASE_SCORES_METADATA_KEY,
    MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY,
    MUTATION_MEMORY_CARD_PROVENANCE_METADATA_KEY,
    MUTATION_MEMORY_DECISION_ID_METADATA_KEY,
    MUTATION_MEMORY_INJECTED_IDS_METADATA_KEY,
    MUTATION_MEMORY_LINEAGE_APPLIED_IDS_METADATA_KEY,
    MUTATION_MEMORY_MUTATION_ASSIGNMENT_METADATA_KEY,
    MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY,
    MUTATION_MEMORY_PARENT_ASSIGNMENTS_METADATA_KEY,
    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
    MUTATION_MEMORY_USED_METADATA_KEY,
    MUTATION_PARENT_STAGE_OUTPUTS_METADATA_KEY,
)
from gigaevo.evolution.mutation.parent_selector import ParentSelector
from gigaevo.evolution.mutation.parent_snapshot import snapshot_parent_stage_outputs
from gigaevo.exceptions import StorageError
from gigaevo.memory.cards import (
    AssignmentRecord,
    CardAssignmentSource,
    DecisionContext,
    MutationAssignmentRecord,
)
from gigaevo.memory.selection_leases import SelectionLease
from gigaevo.programs.metrics.paired import (
    PER_SAMPLE_SCORES_KEY,
    PER_SAMPLE_SIGNATURE_KEY,
)
from gigaevo.programs.program import Program


@dataclass(frozen=True)
class MutationFailure:
    """Classification for an attempt that ended before child persistence."""

    status: Literal["invalid", "censored"]
    stage: str


def _pre_persist_failure_status(
    exc: Exception, *, failure_stage: str
) -> Literal["invalid", "censored"]:
    """Classify failures before a child becomes observable in storage.

    Only typed infrastructure failures are independent of the selected action
    and therefore censorable. Unknown exceptions remain invalid evidence: they
    may be deterministic consequences of malformed generated content.
    """

    if failure_stage == "mutation_persistence" and isinstance(
        exc, (StorageError, ConnectionError, TimeoutError, OSError)
    ):
        return "censored"
    return "invalid"


def lineage_applied_closure(
    *, applied_ids: list[str], parents: list[Program]
) -> list[str]:
    """Transitive closure of every card applied to this child or any ancestor.

    Built from frozen inputs only: the child's just-computed ``applied_ids``
    unioned with each parent's own (birth-frozen) lineage-applied closure.
    """
    closure: set[str] = {cid for cid in applied_ids if cid}
    for parent in parents:
        for cid in (
            parent.get_metadata(MUTATION_MEMORY_LINEAGE_APPLIED_IDS_METADATA_KEY) or []
        ):
            if cid:
                closure.add(cid)
    return sorted(closure)


def base_parent_index(value) -> int:
    """1-based base-parent index from mutator output: wire-JSON emits an int,
    structured diffs a namespace letter ('A' = parent 1); anything else falls
    back to parent 1."""
    if isinstance(value, str) and len(value) == 1 and value.isalpha():
        return ord(value.upper()) - ord("A") + 1
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning(
            "base_parent={!r} is neither an index nor a namespace letter; "
            "attributing credit to parent 1.",
            value,
        )
        return 1


def applied_memory_ids(injected_ids: list[str], mutation_output: object) -> list[str]:
    """Cards the mutator actually applied, limited to the prompt-time slate.

    ``injected_ids`` remains the full exposure slate for write-time attribution.
    Lineage exclusion is stricter: descendants should not re-see cards the branch
    truly used, but merely showing a card should not ban it forever. Current
    structured mutation output carries ``card_ids_used``; legacy/non-structured
    mutations fall back to the injected slate because they have no use signal.
    """
    injected = {cid.strip() for cid in injected_ids if cid.strip()}
    if not injected:
        return []
    if not isinstance(mutation_output, dict):
        return sorted(injected)
    raw_used = mutation_output.get("card_ids_used")
    if not isinstance(raw_used, list):
        return []
    used = {str(cid).strip() for cid in raw_used if str(cid).strip()}
    return sorted(injected & used)


def proposed_probe_ids(parents: list[Program]) -> list[str]:
    """Frozen randomized arms whose card revisions must survive to outcome."""

    proposed: set[str] = set()
    for parent in parents:
        raw = parent.get_metadata(MUTATION_MEMORY_ASSIGNMENT_METADATA_KEY)
        if not isinstance(raw, dict):
            continue
        try:
            assignment = AssignmentRecord.model_validate(raw)
        except Exception:
            continue
        if assignment.probe_arm in ("treated", "control"):
            proposed.update(card_id for card_id in assignment.propensities if card_id)
    return sorted(proposed)


def freeze_base_parent_snapshot(parents, base_parent: int) -> dict:
    """Snapshot the base parent's selected ids and metrics for use-attribution.

    The base parent is the one the mutator named (1-based ``base_parent``); its own
    metadata is overwritten on NO_CACHE requeue, so reward/context must read the
    child's stamp. ``base_fitness`` is derived from ``base_metrics`` at the write
    seam (where the fitness key is known), so it is not frozen here. The per-sample
    score vector, when the eval emits one, is frozen with the metrics — it must
    describe the same evaluation, and the parent's live vector is overwritten on
    re-eval.
    """
    if not parents:
        return {}
    index = base_parent - 1
    if index < 0 or index >= len(parents):
        logger.warning(
            "base_parent={} out of range for {} parent(s); attributing credit to "
            "parent 1. The mutator named a base it was not given.",
            base_parent,
            len(parents),
        )
        index = 0
    base = parents[index]
    snapshot = {
        MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY: [
            card_id
            for card_id in (
                base.get_metadata(MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY) or []
            )
            if card_id
        ],
        MUTATION_MEMORY_BASE_METRICS_METADATA_KEY: dict(base.metrics or {}),
        MUTATION_MEMORY_BASE_ID_METADATA_KEY: base.id,
        MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY: bool(
            base.get_metadata(MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY)
        ),
    }
    decision_id = base.get_metadata(MUTATION_MEMORY_DECISION_ID_METADATA_KEY)
    if isinstance(decision_id, str) and decision_id:
        snapshot[MUTATION_MEMORY_DECISION_ID_METADATA_KEY] = decision_id
    assignment = base.get_metadata(MUTATION_MEMORY_ASSIGNMENT_METADATA_KEY)
    if isinstance(assignment, dict):
        snapshot[MUTATION_MEMORY_ASSIGNMENT_METADATA_KEY] = dict(assignment)
    raw_scores = base.get_metadata(PER_SAMPLE_SCORES_KEY)
    if isinstance(raw_scores, list) and raw_scores:
        snapshot[MUTATION_MEMORY_BASE_SCORES_METADATA_KEY] = list(raw_scores)
        signature = base.get_metadata(PER_SAMPLE_SIGNATURE_KEY)
        if isinstance(signature, str) and signature:
            snapshot[MUTATION_MEMORY_BASE_SCORE_SIGNATURE_METADATA_KEY] = signature

    parent_assignments: dict[str, dict] = {}
    card_sources: dict[str, dict] = {}
    # Base parent first: a card cited by both parents is credited against the
    # anchor baseline (the same parent the child's overall gain uses), not by
    # arbitrary parent order.
    ordered_parents = [
        parents[index],
        *(p for i, p in enumerate(parents) if i != index),
    ]
    for parent in ordered_parents:
        raw_assignment = parent.get_metadata(MUTATION_MEMORY_ASSIGNMENT_METADATA_KEY)
        assignment: AssignmentRecord | None = None
        if isinstance(raw_assignment, dict):
            try:
                assignment = AssignmentRecord.model_validate(raw_assignment)
            except Exception:
                logger.warning(
                    "[mutation] ignoring malformed memory assignment on parent {}",
                    parent.short_id,
                )
        selected_ids = parent.get_metadata(MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY)
        parent_metrics = dict(parent.metrics or {})
        context = (
            assignment.context
            if assignment is not None
            else DecisionContext(parent_id=parent.id, parent_metrics=parent_metrics)
        )
        if not context.parent_id or not context.parent_metrics:
            context = context.model_copy(
                update={
                    "parent_id": context.parent_id or parent.id,
                    "parent_metrics": context.parent_metrics or parent_metrics,
                }
            )
        if assignment is not None:
            assignment = assignment.model_copy(update={"context": context})
            parent_assignments[parent.id] = assignment.model_dump(mode="json")
        scores = parent.get_metadata(PER_SAMPLE_SCORES_KEY)
        try:
            frozen_scores = (
                tuple(float(value) for value in scores)
                if isinstance(scores, list) and scores
                else None
            )
        except (TypeError, ValueError):
            frozen_scores = None
        raw_signature = parent.get_metadata(PER_SAMPLE_SIGNATURE_KEY)
        frozen_signature = (
            raw_signature
            if frozen_scores is not None
            and isinstance(raw_signature, str)
            and raw_signature
            else ""
        )
        for raw_card_id in selected_ids if isinstance(selected_ids, list) else ():
            card_id = str(raw_card_id).strip()
            if not card_id or card_id in card_sources:
                continue
            source = CardAssignmentSource(
                source_card_id=card_id,
                parent_id=parent.id,
                decision_id=assignment.decision_id if assignment is not None else "",
                source_context=context,
                bd_cell=assignment.bd_cell if assignment is not None else None,
                parent_metrics=dict(context.parent_metrics or parent_metrics),
                parent_scores=frozen_scores,
                parent_score_signature=frozen_signature,
            )
            card_sources[card_id] = source.model_dump(mode="json")
    snapshot[MUTATION_MEMORY_PARENT_ASSIGNMENTS_METADATA_KEY] = parent_assignments
    snapshot[MUTATION_MEMORY_CARD_PROVENANCE_METADATA_KEY] = card_sources
    return snapshot


async def generate_one_mutation(
    parents: list[Program],
    *,
    mutator: MutationOperator,
    storage: ProgramStorage,
    state_manager: ProgramStateManager,
    iteration: int,
    task_id: int = 0,
    selection_lease: SelectionLease | None = None,
    failure_observer: Callable[[MutationFailure], None] | None = None,
    child_observer: Callable[[str], None] | None = None,
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
    failure_stage = "mutation_generation"
    failure_observed = False

    def observe_failure(failure: MutationFailure) -> None:
        nonlocal failure_observed
        if failure_observer is not None and not failure_observed:
            failure_observed = True
            failure_observer(failure)

    try:
        mutation_spec = await mutator.mutate_single(parents)

        if mutation_spec is None:
            observe_failure(MutationFailure(status="invalid", stage=failure_stage))
            logger.debug(
                "[mutation] Task {}: mutate_single returned None (parents={})",
                task_id,
                [p.short_id for p in parents],
            )
            return None

        failure_stage = "mutation_materialization"
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
        retained_probe_ids = sorted(set(injected_ids) | set(proposed_probe_ids(parents)))
        mutation_output = mutation_spec.metadata.get(MutationSpec.META_OUTPUT)
        base_parent = 1
        if isinstance(mutation_output, dict):
            base_parent = base_parent_index(mutation_output.get("base_parent", 1) or 1)
        applied_ids = applied_memory_ids(injected_ids, mutation_output)
        program.set_metadata(MUTATION_MEMORY_INJECTED_IDS_METADATA_KEY, injected_ids)
        program.set_metadata(MUTATION_MEMORY_USED_METADATA_KEY, bool(injected_ids))
        program.set_metadata(
            MUTATION_MEMORY_LINEAGE_APPLIED_IDS_METADATA_KEY,
            lineage_applied_closure(applied_ids=applied_ids, parents=parents),
        )
        memory_snapshot = freeze_base_parent_snapshot(parents, base_parent)
        for key, value in memory_snapshot.items():
            program.set_metadata(key, value)
        card_sources = {
            card_id: CardAssignmentSource.model_validate(source)
            for card_id, source in memory_snapshot.get(
                MUTATION_MEMORY_CARD_PROVENANCE_METADATA_KEY, {}
            ).items()
        }
        mutation_assignment = MutationAssignmentRecord(
            mutation_id=program.id,
            parent_ids=tuple(parent.id for parent in parents),
            delivered_ids=tuple(injected_ids),
            source_decision_ids=tuple(
                sorted(
                    {
                        source.decision_id
                        for source in card_sources.values()
                        if source.decision_id
                    }
                )
            ),
            card_sources=card_sources,
        )
        program.set_metadata(
            MUTATION_MEMORY_MUTATION_ASSIGNMENT_METADATA_KEY,
            mutation_assignment.model_dump(mode="json"),
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

        failure_stage = "mutation_persistence"
        if child_observer is not None:
            # Durable causal handoff precedes Redis persistence. A crash after
            # this point leaves either a linked child or a linked-missing child
            # that startup reconciliation can close without guessing.
            child_observer(program.id)
        await storage.add(program)
        persisted_id = program.id  # Point of no return — ID must be returned
        if selection_lease is not None:
            selection_lease.transfer_to_child(
                program.id,
                retained_probe_ids,
            )

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
        if isinstance(exc, asyncio.CancelledError):
            observe_failure(
                MutationFailure(status="censored", stage="mutation_cancelled")
            )
            raise
        if isinstance(exc, Exception):
            observe_failure(
                MutationFailure(
                    status=_pre_persist_failure_status(
                        exc, failure_stage=failure_stage
                    ),
                    stage=failure_stage,
                )
            )
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
