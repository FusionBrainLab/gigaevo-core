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
from typing import Any, ClassVar
from uuid import uuid4

from loguru import logger

from gigaevo.monitoring.emit import emit
from gigaevo.monitoring.events import BaseEvent

MEMORY_EVENTS_FILENAME = "memory_events.jsonl"

_decision_id: ContextVar[str] = ContextVar("memory_event_decision_id", default="")
_program_id: ContextVar[str] = ContextVar("memory_event_program_id", default="")
_parent_ids: ContextVar[tuple[str, ...]] = ContextVar(
    "memory_event_parent_ids", default=()
)
_event_path: ContextVar[Path | None] = ContextVar("memory_event_path", default=None)


def new_decision_id() -> str:
    return f"memsel-{uuid4().hex[:12]}"


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
    emit(stamped)

    target = Path(event_path) if event_path is not None else _event_path.get()
    if target is None:
        target = resolve_memory_event_path()
    if target is not None:
        row = {"event": type(stamped).event, **stamped.model_dump(mode="json")}
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_finite(row), ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning(
                "[Memory][Event] failed to record {}: {}", type(stamped).event, exc
            )
    return stamped


class MemoryStoreWrite(MemoryEvent):
    event: ClassVar[str] = "MEMORY_STORE_WRITE"
    description: ClassVar[str] = "A card-bank mutation (save/delete/merge) completed."
    health_question: ClassVar[str] = "Is the write path landing cards in the bank?"

    # {"save", "delete", "merge"}
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
