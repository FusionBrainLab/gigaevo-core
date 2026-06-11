from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict


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


class IdeaStats(BaseModel):
    """One per-(idea, quartile) row of the origin-analysis summary.

    Typed fields are the ones admission/eviction decisions read; every other
    audit metric (e.g. ``TailRisk_best_median(min(gain,0))``) rides along as an
    extra. Construct via ``IdeaStats.model_validate(row)`` — several row keys
    are not valid identifiers. ``as_row()`` returns the full dict with NaN
    preserved; JSON writers convert NaN to None at the serialization boundary.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    idea_id: str
    quartile: str
    description: str = ""
    intro_events: int = 0
    IntroGain_best_median: float | None = None
    IntroGain_best_rel_median: float | None = None
    IntroGain_best_p10: float | None = None
    DownsideRate_best: float | None = None
    SiblingWinRate_allgens: float | None = None
    BornInElite_rate: float | None = None
    posterior_a: float | None = None
    posterior_b: float | None = None
    p_help_mean: float | None = None
    p_help_lo20: float | None = None
    efficacy_confident: bool | None = None

    def as_row(self) -> dict[str, Any]:
        return self.model_dump()
