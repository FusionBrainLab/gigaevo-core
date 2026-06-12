from __future__ import annotations

import math
from typing import Any

from pydantic import ConfigDict, Field

from gigaevo.memory.shared_memory.models import (
    CardStatsBlock,
    EfficacyMetrics,
    Quartile,
)


def coerce_metric(value: Any) -> float | None:
    """Numeric read of a stats field for decision comparisons.

    Mirrors the pandas ``to_numeric(..., errors="coerce")`` semantics the
    admitters were pinned against: missing / non-numeric / NaN values compare
    False against any threshold, so they return None here.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


class IdeaStats(EfficacyMetrics):
    """One per-(idea, quartile) row of the origin-analysis summary.

    Identity fields plus the full :class:`EfficacyMetrics` vocabulary;
    aggregation is the sole producer, so undeclared keys are rejected
    (``extra="forbid"``). ``as_row()`` returns the full dict with NaN
    preserved; ``as_json_row()`` is the JSON-writer shape with NaN as None.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    idea_id: str = Field(description="Stable identifier of the idea this row scores.")
    quartile: Quartile = Field(
        description="Run-progress slice the row aggregates: a quarter or the ALL aggregate."
    )
    description: str = Field(
        default="", description="Human-readable idea description from the tracker."
    )

    def as_row(self) -> dict[str, Any]:
        return self.model_dump()

    def as_json_row(self) -> dict[str, Any]:
        """Full row dump with NaN converted to None — the best_ideas.json shape."""
        return {
            name: (None if isinstance(value, float) and math.isnan(value) else value)
            for name, value in self.model_dump().items()
        }

    def to_stats_block(self) -> CardStatsBlock:
        """Project the row's metric vocabulary into a card stats block.

        Identity fields stay behind; NaN becomes None at this typed boundary
        (banks.json carries nulls, not NaN).
        """
        metrics = {
            name: (None if isinstance(value, float) and math.isnan(value) else value)
            for name, value in self.model_dump().items()
            if name in EfficacyMetrics.model_fields
        }
        return CardStatsBlock.model_validate(metrics)
