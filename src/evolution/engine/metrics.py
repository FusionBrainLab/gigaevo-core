from __future__ import annotations

from collections import deque
from datetime import datetime

from pydantic import BaseModel, Field, computed_field


class EngineMetrics(BaseModel):
    """Simplified metrics tracking (extracted)."""

    total_generations: int = Field(
        default=0, description="Total number of generations run"
    )
    programs_processed: int = Field(
        default=0, description="Total number of programs processed"
    )
    mutations_created: int = Field(
        default=0, description="Total number of mutations created"
    )
    errors_encountered: int = Field(
        default=0, description="Total number of errors encountered"
    )
    last_generation_time: datetime | None = Field(
        default=None, description="Timestamp of last generation"
    )
    novel_programs_per_generation: deque = Field(
        default_factory=lambda: deque(maxlen=5),
        description="Rolling window of novel programs per generation",
    )

    @computed_field
    @property
    def avg_novel_programs(self) -> float:
        """Average number of novel programs over the rolling window."""
        return sum(self.novel_programs_per_generation) / max(
            1, len(self.novel_programs_per_generation)
        )

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "total_generations": self.total_generations,
            "programs_processed": self.programs_processed,
            "mutations_created": self.mutations_created,
            "errors_encountered": self.errors_encountered,
            "last_generation_time": self.last_generation_time,
            "avg_novel_programs": self.avg_novel_programs,
        }

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}
