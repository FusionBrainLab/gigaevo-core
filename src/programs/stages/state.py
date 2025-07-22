from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import (
    BaseModel,
    Field,
    field_serializer,
)

from src.programs.stages.utils import (
    pickle_b64_deserialize,
    pickle_b64_serialize,
)


class StageState(str, Enum):
    """Status of a processing stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class ProgramStageResult(BaseModel):
    """Result of a program processing stage."""

    status: StageState = Field(
        StageState.PENDING, description="Current status of the stage"
    )
    output: Optional[Any] = Field(
        None, description="Output data from the stage"
    )
    error: Optional[Any] = Field(
        None, description="Error information if stage failed"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional stage metadata"
    )
    started_at: Optional[datetime] = Field(
        None, description="When the stage started"
    )
    finished_at: Optional[datetime] = Field(
        None, description="When the stage finished"
    )

    def duration_seconds(self) -> Optional[float]:
        """Calculate the duration of the stage in seconds."""
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None

    def is_running(self) -> bool:
        """Check if the stage is currently running."""
        return self.status == StageState.RUNNING

    def is_completed(self) -> bool:
        """Check if the stage completed successfully."""
        return self.status == StageState.COMPLETED

    def is_failed(self) -> bool:
        """Check if the stage failed."""
        return self.status == StageState.FAILED

    def is_skipped(self) -> bool:
        """Check if the stage was skipped."""
        return self.status == StageState.SKIPPED

    def mark_started(self) -> None:
        """Mark the stage as started."""
        self.started_at = datetime.now(timezone.utc)
        self.status = StageState.RUNNING

    def mark_completed(
        self, output: Any = None, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Mark the stage as completed."""
        self.finished_at = datetime.now(timezone.utc)
        self.status = StageState.COMPLETED
        if output is not None:
            self.output = output
        if metadata is not None:
            self.metadata = metadata or {}

    def mark_failed(
        self, error: Any = None, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Mark the stage as failed."""
        self.finished_at = datetime.now(timezone.utc)
        self.status = StageState.FAILED
        if error is not None:
            self.error = error
        if metadata is not None:
            self.metadata = metadata or {}

    @field_serializer("output", when_used="json")
    def serialize_output(self, value: Any) -> Optional[str]:
        return pickle_b64_serialize(value) if value is not None else None

    @field_serializer("error", when_used="json")
    def serialize_error(self, value: Any) -> Optional[str]:
        return pickle_b64_serialize(value) if value is not None else None

    @field_serializer("metadata", when_used="json")
    def serialize_metadata(self, value: Any) -> Optional[str]:
        return pickle_b64_serialize(value) if value is not None else None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProgramStageResult":
        """Create a ProgramStageResult from a dictionary."""
        data = dict(data)
        for key in ("output", "error", "metadata"):
            if isinstance(data.get(key), str):
                data[key] = pickle_b64_deserialize(data[key])
        return cls.model_validate(data)
