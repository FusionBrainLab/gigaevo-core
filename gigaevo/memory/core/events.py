"""Canonical structured events for memory decisions.

Memory logs are useful for humans, but future debugging needs one stable JSONL
shape that scripts can join across read decisions, auctions, budget caps, write
gates, and posterior updates. Emission must never affect evolution: failures are
logged and swallowed.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence, Set
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from enum import Enum
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

MEMORY_EVENT_SCHEMA_VERSION = "memory_event.v1"
DEFAULT_MEMORY_EVENTS_FILENAME = "memory_events.jsonl"

_decision_id: ContextVar[str] = ContextVar("memory_event_decision_id", default="")
_program_id: ContextVar[str] = ContextVar("memory_event_program_id", default="")
_parent_ids: ContextVar[tuple[str, ...]] = ContextVar(
    "memory_event_parent_ids", default=()
)
_event_path: ContextVar[Path | None] = ContextVar("memory_event_path", default=None)


class MemoryEventRecord(BaseModel):
    """One append-only canonical memory event row."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=MEMORY_EVENT_SCHEMA_VERSION)
    event_id: str = Field(description="Unique id for this event row.")
    timestamp_utc: str = Field(description="ISO-8601 UTC timestamp.")
    component: str = Field(description="Memory component that made the decision.")
    event_type: str = Field(description="Stable event name, e.g. read.selection.")
    decision_id: str = Field(
        default="",
        description="Correlation id for a prompt-time memory decision.",
    )
    program_id: str = Field(
        default="", description="Primary program id when the event is program-scoped."
    )
    parent_ids: list[str] = Field(
        default_factory=list, description="Parent/program ids in the active context."
    )
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Component-specific JSON-safe details."
    )


def new_memory_decision_id(prefix: str = "memsel") -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


def resolve_memory_event_path(
    checkpoint_dir: str | Path | None = None,
) -> Path | None:
    """Return the JSONL path for canonical memory events, if configured.

    Precedence:
    1. Explicit ``checkpoint_dir``.
    2. Hydra runtime output dir, when Hydra is initialized.
    """

    if checkpoint_dir is not None:
        return Path(checkpoint_dir) / DEFAULT_MEMORY_EVENTS_FILENAME

    try:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            return (
                Path(HydraConfig.get().runtime.output_dir)
                / "memory"
                / DEFAULT_MEMORY_EVENTS_FILENAME
            )
    except Exception:
        return None

    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump(mode="json"))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, Set):
        return sorted((_json_safe(v) for v in value), key=repr)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


@contextmanager
def memory_event_context(
    *,
    decision_id: str | None = None,
    program_id: str | None = None,
    parent_ids: Sequence[str] | None = None,
    event_path: str | Path | None = None,
) -> Iterator[None]:
    """Temporarily attach correlation fields to emitted memory events."""

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


def emit_memory_event(
    *,
    component: str,
    event_type: str,
    payload: Mapping[str, Any] | None = None,
    level: str = "DEBUG",
    event_path: str | Path | None = None,
) -> MemoryEventRecord:
    """Emit one canonical event to JSONL and the log sink.

    The returned record makes unit tests and future call sites easy to inspect.
    File write failures are swallowed after a warning.
    """

    record = MemoryEventRecord(
        event_id=uuid4().hex,
        timestamp_utc=datetime.now(UTC).isoformat(),
        component=component,
        event_type=event_type,
        decision_id=_decision_id.get(),
        program_id=_program_id.get(),
        parent_ids=list(_parent_ids.get()),
        payload=_json_safe(dict(payload or {})),
    )

    target = Path(event_path) if event_path is not None else _event_path.get()
    if target is None:
        target = resolve_memory_event_path()

    if target is not None:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
        except Exception as exc:
            logger.warning("[Memory][Event] failed to record {}: {}", event_type, exc)

    logger.bind(memory_event=record.model_dump(mode="json")).log(
        level,
        "[Memory][Event] component={} event_type={} decision_id={} program_id={} payload={}",
        component,
        event_type,
        record.decision_id,
        record.program_id,
        record.payload,
    )
    return record
