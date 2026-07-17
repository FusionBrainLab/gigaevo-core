"""Production child-level terminal outcomes for memory assignments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import numpy as np

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_ASSIGNMENT_METADATA_KEY,
    MUTATION_MEMORY_BASE_ID_METADATA_KEY,
    MUTATION_MEMORY_BASE_METRICS_METADATA_KEY,
    MUTATION_MEMORY_BASE_SCORE_SIGNATURE_METADATA_KEY,
    MUTATION_MEMORY_BASE_SCORES_METADATA_KEY,
    MUTATION_MEMORY_DECISION_ID_METADATA_KEY,
    MUTATION_MEMORY_MUTATION_ASSIGNMENT_METADATA_KEY,
    MUTATION_MEMORY_OUTCOME_METADATA_KEY,
    MUTATION_MEMORY_PARENT_ASSIGNMENTS_METADATA_KEY,
)
from gigaevo.evolution.mutation.terminal_failure import (
    get_mutation_terminal_failure,
)
from gigaevo.memory.cards import AssignmentRecord, MutationAssignmentRecord
from gigaevo.memory.events import (
    MemoryOutcome,
    MemoryOutcomeUpdate,
    emit_memory_event,
)
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.metrics.paired import (
    COHERENCE_TOL,
    PER_SAMPLE_SCORES_KEY,
    PER_SAMPLE_SIGNATURE_KEY,
)
from gigaevo.programs.program_state import ProgramState

if TYPE_CHECKING:
    from gigaevo.database.program_storage import ProgramStorage
    from gigaevo.programs.program import Program

OutcomeEmission = Literal["emitted", "duplicate", "updated", "not_applicable"]


class MemoryOutcomeSink(Protocol):
    """Durable terminal sink injected into the evolution engine."""

    def record_memory_outcome(self, event: MemoryOutcome) -> None: ...


@runtime_checkable
class MemoryAttemptLifecycleSink(Protocol):
    """Optional durable attempt/child lifecycle used by causal memory policies."""

    def record_attempt_failure(
        self,
        *,
        attempt_id: str,
        status: Literal["invalid", "censored"],
        failure_stage: str,
        completion_ordinal: int,
    ) -> bool: ...

    def link_attempt_child(
        self,
        *,
        attempt_id: str,
        child_id: str,
        completion_ordinal: int,
    ) -> bool: ...

    def record_missing_child(self, child_id: str, *, failure_stage: str) -> bool: ...

    def reconcile_unlinked_attempts(self, *, completion_ordinal: int) -> int: ...

    def pending_child_ids(self) -> tuple[str, ...]: ...


class NullMemoryOutcomeSink:
    def record_memory_outcome(self, event: MemoryOutcome) -> None:
        del event


def record_memory_attempt_failure(
    parents: Sequence[Program],
    *,
    outcome_sink: MemoryOutcomeSink | None,
    metrics_context: MetricsContext | None,
    status: Literal["invalid", "censored"],
    failure_stage: str,
    completion_ordinal: int,
    attempt_id: str | None = None,
) -> int:
    """Close decisions whose mutation attempt ended before a child existed."""

    if outcome_sink is None:
        return 0
    if attempt_id and isinstance(outcome_sink, MemoryAttemptLifecycleSink):
        return int(
            outcome_sink.record_attempt_failure(
                attempt_id=attempt_id,
                status=status,
                failure_stage=failure_stage,
                completion_ordinal=completion_ordinal,
            )
        )
    assignments: list[tuple[Program, AssignmentRecord]] = []
    seen: set[str] = set()
    for parent in parents:
        raw = parent.get_metadata(MUTATION_MEMORY_ASSIGNMENT_METADATA_KEY)
        if not isinstance(raw, dict):
            continue
        try:
            assignment = AssignmentRecord.model_validate(raw)
        except Exception:
            continue
        if not assignment.decision_id or assignment.decision_id in seen:
            continue
        seen.add(assignment.decision_id)
        assignments.append((parent, assignment))
    probe_count = sum(
        assignment.probe_arm in ("treated", "control") for _, assignment in assignments
    )
    primary_metric = (
        metrics_context.get_primary_key() if metrics_context is not None else ""
    )
    higher_is_better = (
        metrics_context.is_higher_better(primary_metric)
        if metrics_context is not None
        else True
    )
    for parent, assignment in assignments:
        event = MemoryOutcome(
            decision_id=assignment.decision_id,
            program_id=parent.id,
            parent_ids=(parent.id,),
            status=status,
            invalid=status == "invalid",
            censor_reason=failure_stage if status == "censored" else "",
            failure_stage=failure_stage,
            completion_ordinal=completion_ordinal,
            child_id=f"no-child:{assignment.decision_id}",
            base_id=parent.id,
            primary_metric=primary_metric,
            higher_is_better=higher_is_better,
            ope_eligible=probe_count <= 1,
        )
        outcome_sink.record_memory_outcome(event)
        emit_memory_event(event)
    return len(assignments)


def _paired_uncertainty(
    program: Program,
    *,
    child_fitness: float,
    base_fitness: float,
    higher_is_better: bool,
) -> tuple[float | None, int | None, Literal["scalar", "paired"], str]:
    """Analytic paired SE only when the ordered evaluation cohorts match."""

    child_raw = program.get_metadata(PER_SAMPLE_SCORES_KEY)
    base_raw = program.get_metadata(MUTATION_MEMORY_BASE_SCORES_METADATA_KEY)
    child_signature = program.get_metadata(PER_SAMPLE_SIGNATURE_KEY)
    base_signature = program.get_metadata(
        MUTATION_MEMORY_BASE_SCORE_SIGNATURE_METADATA_KEY
    )
    if (
        not isinstance(child_signature, str)
        or not child_signature
        or child_signature != base_signature
    ):
        return None, None, "scalar", ""
    try:
        child = np.asarray(child_raw, dtype=float)
        base = np.asarray(base_raw, dtype=float)
    except (TypeError, ValueError):
        return None, None, "scalar", ""
    if (
        child.ndim != 1
        or base.ndim != 1
        or child.shape != base.shape
        or child.size < 2
        or not np.isfinite(child).all()
        or not np.isfinite(base).all()
        or abs(float(child.mean()) - child_fitness) > COHERENCE_TOL
        or abs(float(base.mean()) - base_fitness) > COHERENCE_TOL
    ):
        return None, None, "scalar", ""
    differences = child - base if higher_is_better else base - child
    se = float(np.std(differences, ddof=1) / np.sqrt(child.size))
    if not np.isfinite(se) or se < 0.0:
        return None, None, "scalar", ""
    return se, int(child.size), "paired", child_signature


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
    fitness_delta_se: float | None = None
    n_pairs: int | None = None
    measurement_kind: Literal["scalar", "paired"] = "scalar"
    pairing_signature = ""
    invalid = False
    censor_reason = ""
    failure_stage = ""
    status: Literal["outcome", "invalid", "censored"] = "censored"

    if metrics_context is not None:
        primary_metric = metrics_context.get_primary_key()
        higher_is_better = metrics_context.is_higher_better(primary_metric)

    if program.state == ProgramState.DISCARDED:
        failure = get_mutation_terminal_failure(program)
        if failure is None:
            censor_reason = "child_discarded_unclassified"
            failure_stage = "child_discarded_unclassified"
        else:
            status = failure.status
            invalid = failure.status == "invalid"
            failure_stage = failure.stage.value
            if failure.status == "censored":
                censor_reason = failure.stage.value
    elif metrics_context is None:
        censor_reason = "metrics_context_unavailable"
    else:
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
                (
                    fitness_delta_se,
                    n_pairs,
                    measurement_kind,
                    pairing_signature,
                ) = _paired_uncertainty(
                    program,
                    child_fitness=child_fitness,
                    base_fitness=base_fitness,
                    higher_is_better=higher_is_better,
                )

    return {
        "schema_version": 1,
        "decision_id": decision_id,
        "program_id": program.id,
        "status": status,
        "fitness_delta": fitness_delta,
        "fitness_delta_se": fitness_delta_se,
        "n_pairs": n_pairs,
        "measurement_kind": measurement_kind,
        "pairing_signature": pairing_signature,
        "invalid": invalid,
        "censor_reason": censor_reason,
        "child_id": program.id,
        "base_id": base_id,
        "primary_metric": primary_metric,
        "higher_is_better": higher_is_better,
        "ope_eligible": ope_eligible,
        "failure_stage": failure_stage,
        "completion_ordinal": program.iteration,
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
    outcome_sink: MemoryOutcomeSink | None = None,
) -> OutcomeEmission:
    """Emit at most one terminal row for a child's frozen memory decision.

    The durable marker is claimed before emission. An identical re-evaluation is
    a duplicate and emits nothing; a changed re-evaluation emits the explicitly
    non-terminal ``MEMORY_OUTCOME_UPDATE`` while the first terminal Y stays frozen.
    """
    raw_mutation_assignment = program.get_metadata(
        MUTATION_MEMORY_MUTATION_ASSIGNMENT_METADATA_KEY
    )
    if not isinstance(raw_mutation_assignment, dict):
        return "not_applicable"
    mutation_assignment = MutationAssignmentRecord.model_validate(
        raw_mutation_assignment
    )
    if mutation_assignment.mutation_id != program.id:
        raise ValueError(
            "memory mutation assignment does not belong to terminal child "
            f"{program.id!r}"
        )

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

    terminal_events = tuple(
        MemoryOutcome(
            **{key: value for key, value in payload.items() if key != "schema_version"}
        )
        for payload in new_terminals
    )
    sink = outcome_sink if outcome_sink is not None else NullMemoryOutcomeSink()
    for event in terminal_events:
        # The causal sink is the source of truth and must succeed before the
        # program claims the at-most-once marker. SQLite inserts are idempotent.
        sink.record_memory_outcome(event)

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
    for event in terminal_events:
        emit_memory_event(event)
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
