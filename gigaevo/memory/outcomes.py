"""Production child-level terminal outcomes for memory assignments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_BASE_ID_METADATA_KEY,
    MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
    MUTATION_MEMORY_DECISION_ID_METADATA_KEY,
    MUTATION_MEMORY_OUTCOME_METADATA_KEY,
    MUTATION_MEMORY_PARENT_ASSIGNMENTS_METADATA_KEY,
)
from gigaevo.memory.cards import AssignmentRecord
from gigaevo.memory.events import (
    MemoryOutcome,
    MemoryOutcomeUpdate,
    emit_memory_event,
)
from gigaevo.programs.metrics.context import MetricsContext

if TYPE_CHECKING:
    from gigaevo.database.program_storage import ProgramStorage
    from gigaevo.programs.program import Program

OutcomeEmission = Literal["emitted", "duplicate", "updated", "not_applicable"]


def _outcome_payload(
    program: Program,
    metrics_context: MetricsContext | None,
    *,
    decision_id: str,
    base_id: str,
    base_metrics: dict[str, float],
    ope_eligible: bool,
) -> dict[str, Any]:
    primary_metric = ""
    higher_is_better = True
    fitness_delta: float | None = None
    invalid = False
    censor_reason = ""
    status: Literal["outcome", "invalid", "censored"] = "censored"

    if metrics_context is None:
        censor_reason = "metrics_context_unavailable"
    else:
        primary_metric = metrics_context.get_primary_key()
        higher_is_better = metrics_context.is_higher_better(primary_metric)
        invalid = metrics_context.is_evaluated_invalid(program.metrics, primary_metric)
        if invalid:
            status = "invalid"
        else:
            child_fitness = metrics_context.strict_fitness(
                program.metrics, primary_metric
            )
            base_fitness = metrics_context.strict_fitness(base_metrics, primary_metric)
            if child_fitness is None:
                censor_reason = "child_fitness_unavailable"
            elif base_fitness is None:
                censor_reason = "base_fitness_unavailable"
            else:
                status = "outcome"
                fitness_delta = (
                    child_fitness - base_fitness
                    if higher_is_better
                    else base_fitness - child_fitness
                )

    return {
        "schema_version": 1,
        "decision_id": decision_id,
        "program_id": program.id,
        "status": status,
        "fitness_delta": fitness_delta,
        "invalid": invalid,
        "censor_reason": censor_reason,
        "child_id": program.id,
        "base_id": base_id,
        "primary_metric": primary_metric,
        "higher_is_better": higher_is_better,
        "ope_eligible": ope_eligible,
    }


def _decision_sources(
    program: Program,
) -> list[tuple[str, str, dict[str, float]]]:
    raw_assignments = program.get_metadata(
        MUTATION_MEMORY_PARENT_ASSIGNMENTS_METADATA_KEY
    )
    sources: list[tuple[str, str, dict[str, float]]] = []
    seen: set[str] = set()
    if isinstance(raw_assignments, dict):
        for parent_id, raw_assignment in raw_assignments.items():
            if not isinstance(parent_id, str) or not isinstance(raw_assignment, dict):
                continue
            try:
                assignment = AssignmentRecord.model_validate(raw_assignment)
            except Exception:
                continue
            if not assignment.decision_id or assignment.decision_id in seen:
                continue
            seen.add(assignment.decision_id)
            sources.append(
                (
                    assignment.decision_id,
                    parent_id,
                    dict(assignment.context.parent_metrics),
                )
            )
    if sources:
        return sources

    decision_id = program.get_metadata(MUTATION_MEMORY_DECISION_ID_METADATA_KEY)
    if not isinstance(decision_id, str) or not decision_id:
        return []
    base_id = program.get_metadata(MUTATION_MEMORY_BASE_ID_METADATA_KEY)
    raw_base_metrics = program.get_metadata(MUTATION_MEMORY_BASE_METRICS_METADATA_KEY)
    return [
        (
            decision_id,
            base_id if isinstance(base_id, str) else "",
            dict(raw_base_metrics) if isinstance(raw_base_metrics, dict) else {},
        )
    ]


def _probe_decision_ids(program: Program) -> set[str]:
    """Distinct randomized-probe (treated/control) decision ids on this child."""
    raw_assignments = program.get_metadata(
        MUTATION_MEMORY_PARENT_ASSIGNMENTS_METADATA_KEY
    )
    probe_ids: set[str] = set()
    if isinstance(raw_assignments, dict):
        for raw_assignment in raw_assignments.values():
            if not isinstance(raw_assignment, dict):
                continue
            try:
                assignment = AssignmentRecord.model_validate(raw_assignment)
            except Exception:
                continue
            if assignment.decision_id and assignment.probe_arm in (
                "treated",
                "control",
            ):
                probe_ids.add(assignment.decision_id)
    return probe_ids


async def record_program_memory_outcome(
    program: Program,
    *,
    storage: ProgramStorage,
    metrics_context: MetricsContext | None,
) -> OutcomeEmission:
    """Emit at most one terminal row for a child's frozen memory decision.

    The durable marker is claimed before emission. An identical re-evaluation is
    a duplicate and emits nothing; a changed re-evaluation emits the explicitly
    non-terminal ``MEMORY_OUTCOME_UPDATE`` while the first terminal Y stays frozen.
    """
    sources = _decision_sources(program)
    if not sources:
        return "not_applicable"

    # A child born from >1 randomized-probe decision cannot be attributed to any
    # single probe arm: its one outcome would enter multiple estimator arms. Mark
    # all its terminals ineligible so the DR-AIPW estimator excludes them.
    child_ope_eligible = len(_probe_decision_ids(program)) <= 1
    payloads = {
        decision_id: _outcome_payload(
            program,
            metrics_context,
            decision_id=decision_id,
            base_id=base_id,
            base_metrics=base_metrics,
            ope_eligible=child_ope_eligible,
        )
        for decision_id, base_id, base_metrics in sources
    }
    previous_marker = program.get_metadata(MUTATION_MEMORY_OUTCOME_METADATA_KEY)
    previous_by_decision: dict[str, dict[str, Any]] = {}
    if isinstance(previous_marker, dict):
        raw_by_decision = previous_marker.get("by_decision_id")
        if isinstance(raw_by_decision, dict):
            previous_by_decision = {
                str(key): dict(value)
                for key, value in raw_by_decision.items()
                if isinstance(value, dict)
            }
        elif isinstance(previous_marker.get("decision_id"), str):
            previous_by_decision[previous_marker["decision_id"]] = dict(previous_marker)

    new_terminals: list[dict[str, Any]] = []
    updates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    next_by_decision = dict(previous_by_decision)
    for decision_id, payload in payloads.items():
        previous = previous_by_decision.get(decision_id)
        next_by_decision[decision_id] = payload
        if previous is None:
            new_terminals.append(payload)
        elif previous != payload:
            updates.append((previous, payload))
    if not new_terminals and not updates:
        return "duplicate"

    program.set_metadata(
        MUTATION_MEMORY_OUTCOME_METADATA_KEY,
        {"schema_version": 2, "by_decision_id": next_by_decision},
    )
    try:
        await storage.update(program)
    except Exception:
        if previous_marker is None:
            program.metadata.pop(MUTATION_MEMORY_OUTCOME_METADATA_KEY, None)
        else:
            program.set_metadata(MUTATION_MEMORY_OUTCOME_METADATA_KEY, previous_marker)
        raise
    for payload in new_terminals:
        emit_memory_event(
            MemoryOutcome(**{k: v for k, v in payload.items() if k != "schema_version"})
        )
    for previous, payload in updates:
        event_payload = {k: v for k, v in payload.items() if k != "schema_version"}
        emit_memory_event(
            MemoryOutcomeUpdate(
                **event_payload,
                previous_status=previous.get("status", "censored"),
                previous_fitness_delta=previous.get("fitness_delta"),
            )
        )
    return "emitted" if new_terminals else "updated"
