"""Shared Beta-prior value type for memory context/read policies."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BetaPrior(BaseModel):
    """Validated Beta prior plus provenance for auction and no-card telemetry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    alpha: float = Field(gt=0.0, description="Beta alpha parameter.")
    beta: float = Field(gt=0.0, description="Beta beta parameter.")
    source: str = Field(default="fixed", description="Where this prior came from.")
    support_n: float = Field(
        default=0.0, ge=0.0, description="Effective evidence behind the prior."
    )

    @field_validator("alpha", "beta")
    @classmethod
    def _finite_positive(cls, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("Beta prior parameters must be finite and positive")
        return value

    def as_tuple(self) -> tuple[float, float]:
        return (self.alpha, self.beta)


def coerce_beta_prior(value: Any, *, source: str = "fixed") -> BetaPrior:
    """Accept legacy ``[alpha, beta]`` YAML values at policy seams."""

    if isinstance(value, BetaPrior):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return BetaPrior(alpha=float(value[0]), beta=float(value[1]), source=source)
    raise TypeError(f"expected BetaPrior or [alpha, beta], got {value!r}")
