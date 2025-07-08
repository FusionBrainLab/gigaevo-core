from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel, Field
from collections import deque

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
        default_factory=lambda: deque(maxlen=100),
        description="Rolling window of novel programs per generation",
    )

    @property
    def avg_novel_programs(self) -> float:
        """Average number of novel programs over the rolling window."""
        return (
            sum(self.novel_programs_per_generation) / max(1, len(self.novel_programs_per_generation))
        )

    def to_dict(self) -> Dict[str, Any]:
        data = self.model_dump()
        data["avg_novel_programs"] = self.avg_novel_programs
        return data

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"} 