"""General canonical-events registry.

Events are Pydantic subclasses of `BaseEvent`. Subclassing a concrete event
(one with a non-empty `event: ClassVar[str]`) auto-registers the class in
`CANONICAL_EVENTS` via `__init_subclass__`. There is no central list to
maintain — defining a subclass IS registration.

This module is role-agnostic: no G/D labels, no adversarial-specific fields.
Experiments tag their runs via the optional `run_label` field; downstream
tooling (e.g. the `gigaevo events plot` CLI) groups by user-supplied regex.

Adversarial-specific events (TRACKER_WRITE, HOF_FETCH, HOF_ROTATE, CELL_PICK)
live in `gigaevo.adversarial.events` so they can carry role-invariant
validators without polluting the general registry.

Emission seams (each event has exactly one):
- EXCEPTION           -> loguru exception sink hook
- STAGE_EXEC          -> Stage.__call__ base-class wrapper
- LLM_CALL            -> LLM client request/response wrapper
- METRIC_EMIT         -> metric logging helper
- BACKPRESSURE_SAMPLE -> backpressure_sampler periodic tick

``GENERATION_BOUNDARY`` is parse-only — its Pydantic schema is retained so
``log_audit`` can validate archived run logs, but no current code path
emits it.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt, model_validator

CANONICAL_EVENTS: dict[str, type[BaseEvent]] = {}


class BaseEvent(BaseModel):
    """Base class for all canonical events.

    Subclasses declare ClassVars for metadata (`event`, `description`,
    `health_question`, `expected_after_gen`, `schema_version`) and Pydantic
    fields for payload. Subclassing triggers auto-registration by class name.
    """

    model_config = ConfigDict(extra="forbid")

    # Metadata (class-level). `event` must be set on concrete subclasses.
    event: ClassVar[str] = ""
    description: ClassVar[str] = ""
    health_question: ClassVar[str] = ""
    expected_after_gen: ClassVar[int] = 0
    schema_version: ClassVar[int] = 1

    # Free-form run tag — experiments populate this with whatever convention
    # they use (e.g. "K5_1_G"). The plot tool groups by user-supplied regex.
    run_label: str | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        event_name = getattr(cls, "event", "") or ""
        if not event_name:
            # Intermediate/abstract subclass — skip registration.
            return
        if event_name in CANONICAL_EVENTS:
            raise ValueError(
                f"Duplicate canonical event name: {event_name!r} "
                f"(already registered as {CANONICAL_EVENTS[event_name].__name__})"
            )
        CANONICAL_EVENTS[event_name] = cls


# --------------------------------------------------------------------------- #
# General events — one emission site each.                                    #
# --------------------------------------------------------------------------- #


class GenerationBoundary(BaseEvent):
    """Parse-only schema for archived run logs.

    No current emitter; retained so ``log_audit`` can validate
    ``GENERATION_BOUNDARY`` entries in archived logs.
    """

    event: ClassVar[str] = "GENERATION_BOUNDARY"
    description: ClassVar[str] = "Archived event — engine generation tick."
    health_question: ClassVar[str] = "Where are we in the run?"

    gen: int


class Exception_(BaseEvent):
    # Underscore to avoid shadowing the builtin; registered as "EXCEPTION".
    event: ClassVar[str] = "EXCEPTION"
    description: ClassVar[str] = "An exception was logged via loguru."
    health_question: ClassVar[str] = "What is silently failing?"

    where: str
    exc_type: str
    msg_head: str
    program_id: str | None = None


class StageExec(BaseEvent):
    event: ClassVar[str] = "STAGE_EXEC"
    description: ClassVar[str] = "A pipeline stage executed (hit/miss/rerun)."
    health_question: ClassVar[str] = "Are stages caching and rerunning as intended?"

    stage: str
    program_id: str
    # {"hit", "miss", "no_cache", "rerun_invalidated"}
    decision: str
    cache_key_hash: str | None = None
    upstream_changed: bool = False
    duration_ms: float


class LLMCall(BaseEvent):
    event: ClassVar[str] = "LLM_CALL"
    description: ClassVar[str] = "An LLM request completed (or failed)."
    health_question: ClassVar[str] = "Are LLMs firing, succeeding, and how long?"

    stage: str
    program_id: str | None = None
    endpoint: str
    model: str
    attempt: int = 1
    ok: bool
    latency_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    error_type: str | None = None


class MetricEmit(BaseEvent):
    event: ClassVar[str] = "METRIC_EMIT"
    description: ClassVar[str] = "A metric was written to program.metrics."
    health_question: ClassVar[str] = "Are engine-consumed values plausible?"

    program_id: str
    metric: str
    value: Any


class BackpressureSample(BaseEvent):
    """One snapshot of the steady-state two-sema backpressure state.

    Emitted periodically by the engine so a runner log carries a time
    series of held producer / buffer / in-flight counts. ``llm_active``
    breaks down producer occupancy into LLM inference vs DAG evaluation
    phases. Without this the only published evidence that ``max_in_flight``
    is enforced is the boot banner — concurrency-over-time is invisible.
    """

    event: ClassVar[str] = "BACKPRESSURE_SAMPLE"
    description: ClassVar[str] = (
        "Periodic snapshot of held producer/buffer slots + in_flight + llm_active."
    )
    health_question: ClassVar[str] = (
        "Is max_in_flight actually being utilised? LLM vs DAG split?"
    )

    producer_held: NonNegativeInt
    buffer_held: NonNegativeInt
    in_flight: NonNegativeInt
    max_in_flight: PositiveInt
    llm_active: NonNegativeInt

    @model_validator(mode="after")
    def _hold_within_cap(self) -> BackpressureSample:
        # held > cap means accounting is broken — surface loudly rather than
        # silently logging rubbish that downstream dashboards then plot as truth.
        cap = self.max_in_flight
        if self.producer_held > cap:
            raise ValueError(
                f"producer_held={self.producer_held} exceeds max_in_flight={cap}"
            )
        if self.buffer_held > cap:
            raise ValueError(
                f"buffer_held={self.buffer_held} exceeds max_in_flight={cap}"
            )
        if self.in_flight > cap:
            raise ValueError(f"in_flight={self.in_flight} exceeds max_in_flight={cap}")
        if self.llm_active > self.producer_held:
            raise ValueError(
                f"llm_active={self.llm_active} exceeds producer_held={self.producer_held}"
            )
        return self
