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
- GENERATION_BOUNDARY -> engine generation tick
- EXCEPTION           -> loguru exception sink hook
- STAGE_EXEC          -> Stage.__call__ base-class wrapper
- LLM_CALL            -> LLM client request/response wrapper
- METRIC_EMIT         -> metric logging helper
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

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
    event: ClassVar[str] = "GENERATION_BOUNDARY"
    description: ClassVar[str] = "Engine ticked a generation boundary."
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
