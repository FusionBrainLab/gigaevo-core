"""Slate-calibrated cold-card prior for the EV Thompson auction.

The flat ``prior_magnitude`` in :class:`EVThompsonAuctioneer` is a per-substrate
constant: on heilbron its default 0.1 is ~25-80x the learned card magnitudes
(~0.001-0.004), so cold cards out-bid proven cards and the cold-winner fraction
*rises* over a run (docs/audits/bandit_health_report.md, RQ1). This auctioneer
removes the magic number: a cold card bids the ``cold_quantile`` (median by
default) of the *proven* magnitudes present on the same slate, so its bid tracks
the slate's own scale. When no proven magnitude is present (early run or an
all-cold slate) every cold card bids ``cold_floor`` and the EV auction degenerates
to the plain Thompson safety gate.

Calibration is a pure pre-pass over candidate magnitudes (no RNG); the filled
candidates then delegate to :meth:`EVThompsonAuctioneer.run` unchanged, so the
seed-exact draw-order contract (tests/memory/test_ev_auction.py) is preserved.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.core.auctioneer import (
    AuctionBid,
    AuctionCandidate,
    EVThompsonAuctioneer,
)


class CalibratedColdPriorAuctioneer(BaseModel):
    """EV Thompson auction whose cold-card magnitude is calibrated per slate."""

    model_config = ConfigDict(frozen=True)

    baseline_prior: tuple[float, float] = Field(
        default=(3.0, 3.0),
        description="(alpha, beta) of the no-card baseline arm each candidate must beat.",
    )
    ev_floor: float = Field(
        default=0.0,
        description="EV reserve: a winner's bid (theta x magnitude) must exceed this.",
    )
    cold_quantile: float = Field(
        default=0.5,
        description="Quantile of the slate's present (non-None) magnitudes used as the "
        "cold-card bid. 0.5 = median; lower is more conservative exploration.",
    )
    cold_floor: float = Field(
        default=1e-6,
        description="Strictly-positive cold bid when no proven magnitude is present "
        "(all-cold slate) or the calibrated quantile is non-positive; keeps a fresh "
        "card explorable instead of stranded at zero by the ev_floor.",
    )

    def run(
        self, candidates: list[AuctionCandidate], rng: Any
    ) -> tuple[list[str], list[AuctionBid]]:
        present = [c.magnitude for c in candidates if c.magnitude is not None]
        if present:
            cold_mag = max(
                float(np.quantile(present, self.cold_quantile)), self.cold_floor
            )
        else:
            cold_mag = self.cold_floor
        # Pre-pass is RNG-free; delegating filled candidates keeps the seed-exact
        # draw order of EVThompsonAuctioneer intact.
        filled = [
            c
            if c.magnitude is not None
            else c.model_copy(update={"magnitude": cold_mag})
            for c in candidates
        ]
        inner = EVThompsonAuctioneer(
            baseline_prior=self.baseline_prior,
            ev_floor=self.ev_floor,
            prior_magnitude=cold_mag,
        )
        return inner.run(filled, rng)
