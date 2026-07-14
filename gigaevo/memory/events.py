"""Typed canonical events for the memory system.

Memory events are ordinary :class:`~gigaevo.monitoring.events.BaseEvent`
subclasses — defining one registers it in ``CANONICAL_EVENTS`` and buys the
loguru ``[EVENT_NAME] {json}`` line plus Redis minute-bucket counters for
free. On top of that shared machinery, memory adds two things:

- **Correlation.** A prompt-time memory decision spans several components
  (research, auction, budget). :func:`memory_event_context` attaches a
  ``decision_id`` / ``program_id`` / ``parent_ids`` triple via contextvars so
  every event emitted inside the block can be joined offline.
- **A JSONL sink.** :func:`emit_memory_event` additionally appends the event
  as one row to ``memory_events.jsonl`` (per-run, under the Hydra output
  dir), the stable shape the analysis tools consume across a whole run.

Emission must never affect evolution: sink failures are logged and swallowed.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar, Literal
from uuid import uuid4

from loguru import logger
from pydantic import Field

from gigaevo.memory.cards import AssignmentRecord
from gigaevo.monitoring.emit import emit
from gigaevo.monitoring.events import BaseEvent

MEMORY_EVENTS_FILENAME = "memory_events.jsonl"

# Serializes JSONL sink appends: the writer emits from worker threads
# (stats/eviction off the event loop) while the reader emits from the loop, and
# a row can exceed the atomic-append size (PIPE_BUF), so concurrent appends
# could otherwise interleave into a torn line.
_sink_lock = Lock()

_decision_id: ContextVar[str] = ContextVar("memory_event_decision_id", default="")
_program_id: ContextVar[str] = ContextVar("memory_event_program_id", default="")
_parent_ids: ContextVar[tuple[str, ...]] = ContextVar(
    "memory_event_parent_ids", default=()
)
_event_path: ContextVar[Path | None] = ContextVar("memory_event_path", default=None)


def new_decision_id() -> str:
    return f"memsel-{uuid4().hex}"


def resolve_memory_event_path(
    checkpoint_dir: str | Path | None = None,
) -> Path | None:
    """The JSONL sink path, if one is configured.

    Precedence: an explicit ``checkpoint_dir``, else the Hydra runtime output
    dir when Hydra is initialized, else ``None`` (sink disabled — e.g. tests
    and ad-hoc scripts).
    """
    if checkpoint_dir is not None:
        return Path(checkpoint_dir) / MEMORY_EVENTS_FILENAME

    try:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            return (
                Path(HydraConfig.get().runtime.output_dir)
                / "memory"
                / MEMORY_EVENTS_FILENAME
            )
    except Exception:
        return None

    return None


@contextmanager
def memory_event_context(
    *,
    decision_id: str | None = None,
    program_id: str | None = None,
    parent_ids: Sequence[str] | None = None,
    event_path: str | Path | None = None,
) -> Iterator[None]:
    """Attach correlation fields (and/or the sink path) to nested emits."""
    tokens: list[tuple[ContextVar[Any], Any]] = []
    if decision_id is not None:
        tokens.append((_decision_id, _decision_id.set(decision_id)))
    if program_id is not None:
        tokens.append((_program_id, _program_id.set(program_id)))
    if parent_ids is not None:
        tokens.append((_parent_ids, _parent_ids.set(tuple(parent_ids))))
    if event_path is not None:
        tokens.append((_event_path, _event_path.set(Path(event_path))))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


class MemoryEvent(BaseEvent):
    """Abstract base carrying the cross-component correlation fields.

    Empty ``event`` ClassVar → not registered; concrete subclasses set a
    unique name and auto-register. ``timestamp_utc`` and unset correlation
    fields are stamped by :func:`emit_memory_event` at emission time.
    """

    timestamp_utc: str = ""
    decision_id: str = ""
    program_id: str = ""
    parent_ids: tuple[str, ...] = ()


class BufferedMemoryEvents:
    def __init__(self) -> None:
        self._events: list[tuple[MemoryEvent, str | Path | None]] = []

    def append(self, event: MemoryEvent, event_path: str | Path | None) -> None:
        self._events.append((event, event_path))

    @staticmethod
    def _retain_assignment(
        assignment: AssignmentRecord, selected_ids: tuple[str, ...]
    ) -> AssignmentRecord:
        assigned_ids = tuple(sorted(set(selected_ids)))
        retained = set(assigned_ids)
        return assignment.model_copy(
            update={
                "assigned_ids": assigned_ids,
                "delivered_ids": assigned_ids,
                "arm": "injected" if assigned_ids else "none",
                "probe_arm": assignment.probe_arm,
                "propensities": dict(assignment.propensities),
                "predicted_help": dict(assignment.predicted_help),
                "predicted_gain": dict(assignment.predicted_gain),
                "predicted_no_card_gain": dict(assignment.predicted_no_card_gain),
                "pending_by_card": {
                    card_id: value
                    for card_id, value in assignment.pending_by_card.items()
                    if card_id in retained
                },
                "pending_discount_by_card": {
                    card_id: value
                    for card_id, value in assignment.pending_discount_by_card.items()
                    if card_id in retained
                },
            }
        )

    def rewrite_terminal_selection(
        self,
        *,
        decision_id: str,
        selected_ids: tuple[str, ...],
        assignment: AssignmentRecord | None,
    ) -> AssignmentRecord | None:
        corrected_assignment = (
            self._retain_assignment(assignment, selected_ids)
            if assignment is not None
            else None
        )
        rewritten: list[tuple[MemoryEvent, str | Path | None]] = []
        for event, event_path in self._events:
            if event.decision_id == decision_id and isinstance(
                event, MemoryReadSelection
            ):
                empty_reason = event.empty_reason
                if selected_ids:
                    empty_reason = ""
                elif event.selected_ids:
                    empty_reason = "lease_vanished"
                event = event.model_copy(
                    update={
                        "selected_ids": selected_ids,
                        "empty_reason": empty_reason,
                    }
                )
            elif event.decision_id == decision_id and isinstance(
                event, MemoryAssignment
            ):
                event = event.model_copy(
                    update={
                        "assignment": self._retain_assignment(
                            event.assignment, selected_ids
                        )
                    }
                )
            rewritten.append((event, event_path))
        self._events = rewritten
        return corrected_assignment

    def commit(self) -> None:
        events, self._events = self._events, []
        for event, event_path in events:
            _emit_stamped_memory_event(event, event_path=event_path)


_event_buffer: ContextVar[BufferedMemoryEvents | None] = ContextVar(
    "memory_event_buffer", default=None
)


@contextmanager
def memory_event_buffer() -> Iterator[BufferedMemoryEvents]:
    buffer = BufferedMemoryEvents()
    token = _event_buffer.set(buffer)
    try:
        yield buffer
    finally:
        _event_buffer.reset(token)


def _finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def emit_memory_event(
    event: MemoryEvent, *, event_path: str | Path | None = None
) -> MemoryEvent:
    """Stamp, emit canonically, and append to the JSONL sink.

    Returns the stamped copy so call sites and tests can inspect exactly what
    was recorded.
    """
    stamped = event.model_copy(
        update={
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "decision_id": event.decision_id or _decision_id.get(),
            "program_id": event.program_id or _program_id.get(),
            "parent_ids": event.parent_ids or _parent_ids.get(),
        }
    )
    buffer = _event_buffer.get()
    if buffer is not None:
        buffer.append(stamped, event_path)
        return stamped
    return _emit_stamped_memory_event(stamped, event_path=event_path)


def _emit_stamped_memory_event(
    stamped: MemoryEvent, *, event_path: str | Path | None = None
) -> MemoryEvent:
    emit(stamped)

    target = Path(event_path) if event_path is not None else _event_path.get()
    if target is None:
        target = resolve_memory_event_path()
    if target is not None:
        row = {
            "event": type(stamped).event,
            "schema_version": type(stamped).schema_version,
            **stamped.model_dump(mode="json"),
        }
        line = json.dumps(_finite(row), ensure_ascii=False) + "\n"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with _sink_lock, target.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception as exc:
            logger.warning(
                "[Memory][Event] failed to record {}: {}", type(stamped).event, exc
            )
    return stamped


class MemoryStoreWrite(MemoryEvent):
    event: ClassVar[str] = "MEMORY_STORE_WRITE"
    description: ClassVar[str] = (
        "A card-bank mutation (save/update/delete/merge) completed."
    )
    health_question: ClassVar[str] = "Is the write path landing cards in the bank?"

    # {"save", "update", "delete", "merge"}
    op: str
    # {"ok", "not_found", "noop"}
    outcome: str
    card_ids: tuple[str, ...] = ()
    bank_count: int = 0


class MemoryStoreSync(MemoryEvent):
    event: ClassVar[str] = "MEMORY_STORE_SYNC"
    description: ClassVar[str] = "A store persist/refresh/rebuild pass completed."
    health_question: ClassVar[str] = "Are bank persistence and index rebuilds healthy?"

    # {"refresh", "rebuild"}
    op: str
    # {"ok", "noop", "failed"}
    outcome: str
    card_count: int = 0
    duration_ms: float = 0.0
    error: str = ""


class MemoryResearch(MemoryEvent):
    event: ClassVar[str] = "MEMORY_RESEARCH"
    description: ClassVar[str] = "One research() call over the store completed."
    health_question: ClassVar[str] = "Is agentic retrieval returning candidates?"

    # {"ok", "empty", "failed"}
    outcome: str
    iterations: int = 0
    query_chars: int = 0
    exclude_count: int = 0
    candidate_ids: tuple[str, ...] = ()
    duration_ms: float = 0.0
    error: str = ""


class MemoryResearchStep(MemoryEvent):
    event: ClassVar[str] = "MEMORY_RESEARCH_STEP"
    description: ClassVar[str] = (
        "One plan → retrieve → reflect iteration of the research agent."
    )
    health_question: ClassVar[str] = (
        "Is the research loop planning real queries and converging?"
    )

    step: int
    scopes: tuple[str, ...] = ()
    query_count: int = 0
    hit_ids: tuple[str, ...] = ()
    # {"final", "continue"}
    decision: str = ""
    duration_ms: float = 0.0


class MemoryAuctionRun(MemoryEvent):
    event: ClassVar[str] = "MEMORY_AUCTION_RUN"
    description: ClassVar[str] = (
        "One Thompson auction over retrieved candidates completed."
    )
    health_question: ClassVar[str] = "Are auction winners emerging at a healthy rate?"

    # {"thompson", "thompson_ev", "thompson_bootstrap"}
    auction: str
    candidate_count: int = 0
    winner_count: int = 0
    winner_ids: tuple[str, ...] = ()
    baseline_prior: tuple[float, float] = (0.0, 0.0)
    cold_magnitude: float | None = None
    ev_floor: float | None = None
    bids: tuple[dict[str, Any], ...] = ()


class MemoryBudgetCap(MemoryEvent):
    event: ClassVar[str] = "MEMORY_BUDGET_CAP"
    description: ClassVar[str] = (
        "The budgeter capped auction winners to the injection budget."
    )
    health_question: ClassVar[str] = "Is the injection budget dropping strong winners?"

    # {"theta", "bid"}
    rank_key: str
    winner_count: int = 0
    max_cards: int = 0
    kept_ids: tuple[str, ...] = ()
    dropped_ids: tuple[str, ...] = ()
    rank_by_card_id: dict[str, float] = Field(default_factory=dict)


class MemoryGainRestamp(MemoryEvent):
    event: ClassVar[str] = "MEMORY_GAIN_RESTAMP"
    description: ClassVar[str] = (
        "Use-attributed gain events recomputed from the pool and restamped."
    )
    health_question: ClassVar[str] = (
        "Is gain attribution crediting cards with injection events?"
    )

    credited_card_count: int = 0
    event_count_by_card_id: dict[str, int] = Field(default_factory=dict)


class MemoryEvictionSweep(MemoryEvent):
    event: ClassVar[str] = "MEMORY_EVICTION_SWEEP"
    description: ClassVar[str] = (
        "A configured eviction sweep deleted cards from the bank."
    )
    health_question: ClassVar[str] = "Is memory eviction firing, and how often?"

    bank_count: int = 0
    evicted_ids: tuple[str, ...] = ()


class MemoryPriorCohort(MemoryEvent):
    event: ClassVar[str] = "MEMORY_PRIOR_COHORT"
    description: ClassVar[str] = (
        "Live and evicted card counts backing an empirical-Bayes cold prior."
    )
    health_question: ClassVar[str] = (
        "Is cold-prior evidence retaining the evicted cohort?"
    )

    live_card_count: int = 0
    evicted_card_count: int = 0
    cohort_card_count: int = 0


class MemoryConsolidationPass(MemoryEvent):
    event: ClassVar[str] = "MEMORY_CONSOLIDATION_PASS"
    description: ClassVar[str] = (
        "One background near-duplicate consolidation pass finished."
    )
    health_question: ClassVar[str] = (
        "Is consolidation folding drifted near-duplicates (or failing)?"
    )

    # {"ok", "failed"}
    outcome: str
    merged: int = 0
    failures: int = 0
    error: str = ""


class MemoryAssignment(MemoryEvent):
    event: ClassVar[str] = "MEMORY_ASSIGNMENT"
    description: ClassVar[str] = "One durable memory read-policy assignment."
    health_question: ClassVar[str] = (
        "Can every memory decision be joined to exactly one child outcome?"
    )

    assignment: AssignmentRecord


class MemoryOutcome(MemoryEvent):
    event: ClassVar[str] = "MEMORY_OUTCOME"
    description: ClassVar[str] = (
        "The first terminal child evaluation for one memory assignment."
    )
    health_question: ClassVar[str] = (
        "Does every durable assignment receive exactly one honest child-level outcome?"
    )

    status: Literal["outcome", "invalid", "censored"]
    fitness_delta: float | None = None
    invalid: bool = False
    censor_reason: str = ""
    child_id: str
    base_id: str
    primary_metric: str = ""
    higher_is_better: bool = True


class MemoryOutcomeUpdate(MemoryEvent):
    event: ClassVar[str] = "MEMORY_OUTCOME_UPDATE"
    description: ClassVar[str] = (
        "A changed re-evaluation of a child whose terminal memory outcome is frozen."
    )
    health_question: ClassVar[str] = (
        "Are re-evaluations changing frozen memory-assignment outcomes?"
    )

    status: Literal["outcome", "invalid", "censored"]
    previous_status: Literal["outcome", "invalid", "censored"]
    fitness_delta: float | None = None
    previous_fitness_delta: float | None = None
    invalid: bool = False
    censor_reason: str = ""
    child_id: str
    base_id: str
    primary_metric: str = ""
    higher_is_better: bool = True


class MemoryDelivery(MemoryEvent):
    event: ClassVar[str] = "MEMORY_DELIVERY"
    description: ClassVar[str] = (
        "The final mutation-prompt delivery after downstream no-card withholding."
    )
    health_question: ClassVar[str] = (
        "Which assigned cards actually reached the mutator?"
    )

    assignment: AssignmentRecord
    assigned_ids: tuple[str, ...] = ()
    delivered_ids: tuple[str, ...] = ()
    withheld_for_control: bool = False
    no_card_control_probability: float = 0.0


class MemoryReadSelection(MemoryEvent):
    event: ClassVar[str] = "MEMORY_READ_SELECTION"
    description: ClassVar[str] = (
        "One end-to-end read decision (research → auction → budget → render)."
    )
    health_question: ClassVar[str] = (
        "Is the read path injecting cards — and when empty, at which stage?"
    )

    mutation_mode: str = ""
    max_cards: int = 0
    exclude_ids: tuple[str, ...] = ()
    research_iterations: int = 0
    candidate_ids: tuple[str, ...] = ()
    auction_winner_ids: tuple[str, ...] = ()
    budgeted_ids: tuple[str, ...] = ()
    render_dropped_ids: tuple[str, ...] = ()
    selected_ids: tuple[str, ...] = ()
    slate: tuple[dict[str, Any], ...] = ()
    # one of: "", "max_cards_nonpositive", "research_empty", "auction_rejected",
    # "budget_empty", "render_empty", "lease_vanished", "exception"
    empty_reason: str = ""
    timing_ms: dict[str, float] = Field(default_factory=dict)
    error: str = ""
