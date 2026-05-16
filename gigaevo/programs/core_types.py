from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
import hashlib
import traceback
from typing import Any

import cloudpickle
from pydantic import BaseModel, Field, field_serializer

from gigaevo.programs.utils import pickle_b64_deserialize, pickle_b64_serialize


class StageIO(BaseModel):
    """Strict base for stage inputs/outputs (used by stage classes & DAG typing).

    Provides a `content_hash` property for cache invalidation.
    """

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(cloudpickle.dumps(self.model_dump())).hexdigest()[:16]


class VoidInput(StageIO):
    pass


class VoidOutput(StageIO):
    pass


class StageError(BaseModel):
    type: str = Field(..., description="Exception class or category")
    message: str = Field(..., description="Human-readable message")
    stage: str | None = Field(default=None, description="Stage class name, if known")
    traceback: str | None = Field(default=None, description="Formatted traceback")

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        stage: str | None = None,
        include_traceback: bool = True,
    ) -> StageError:
        tb_str = None
        if include_traceback:
            tb_str = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
        msg = str(exc) or repr(exc)
        return cls(type=type(exc).__name__, message=msg, stage=stage, traceback=tb_str)

    def pretty(self, include_traceback: bool = False) -> str:
        head = f"[{self.stage or 'unknown'}] {self.type}: {self.message}"
        if include_traceback and self.traceback:
            return f"{head}\n\nTraceback:\n{self.traceback}"
        return head


class StageState(StrEnum):
    """Status of a processing stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


FINAL_STATES = {
    StageState.COMPLETED,
    StageState.FAILED,
    StageState.CANCELLED,
    StageState.SKIPPED,
}


class ProgramStageResult(BaseModel):
    status: StageState = Field(default=StageState.PENDING)
    output: Any | None = None
    error: StageError | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    input_hash: str | None = Field(
        default=None,
        description="Hash of inputs when stage was executed (for cache invalidation)",
    )

    def duration_seconds(self) -> float | None:
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None

    def mark_started(self) -> None:
        self.started_at = datetime.now(UTC)
        self.status = StageState.RUNNING

    def mark_completed(self, output: Any | None = None) -> None:
        self.finished_at = datetime.now(UTC)
        self.status = StageState.COMPLETED
        if output is not None:
            self.output = output

    def mark_failed(self, error: StageError) -> None:
        self.finished_at = datetime.now(UTC)
        self.status = StageState.FAILED
        self.error = error

    @classmethod
    def success(
        cls, *, output: Any | None = None, started_at: datetime | None = None
    ) -> ProgramStageResult:
        res = cls(started_at=started_at or datetime.now(UTC))
        res.mark_completed(output=output)
        return res

    @classmethod
    def failure(
        cls, *, error: StageError, started_at: datetime | None = None
    ) -> ProgramStageResult:
        res = cls(started_at=started_at or datetime.now(UTC))
        res.mark_failed(error=error)
        return res

    @classmethod
    def skipped(
        cls,
        *,
        message: str = "Stage skipped",
        stage: str | None = None,
        error_type: str = "Skip",
    ) -> ProgramStageResult:
        """Create a result indicating the stage was skipped (e.g. no input data)."""
        now = datetime.now(UTC)
        return cls(
            status=StageState.SKIPPED,
            error=StageError(type=error_type, message=message, stage=stage),
            started_at=now,
            finished_at=now,
        )

    @field_serializer("output", when_used="json")
    def _ser_output(self, value: Any | None) -> str | None:
        return pickle_b64_serialize(value) if value is not None else None

    @field_serializer("error", when_used="json")
    def _ser_error(self, value: StageError | None) -> str | None:
        return pickle_b64_serialize(value) if value is not None else None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProgramStageResult:
        d = dict(data)
        for key in ("output", "error"):
            if isinstance(d.get(key), str):
                d[key] = pickle_b64_deserialize(d[key])
        return cls.model_validate(d)
