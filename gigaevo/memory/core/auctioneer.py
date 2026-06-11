from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict


class ThompsonAuctioneer(BaseModel):
    """Thompson auction: each card's posterior draw competes against a no-card arm.

    For each ``(card_id, a, b)`` draw ``theta ~ Beta(a, b)`` and a fresh no-card
    ``base ~ Beta(*baseline_prior)``; select the card iff ``theta > base``. Winners
    are the emergent 0..N subset; ``records`` keep per-candidate draws for audit.
    Draw order (theta then base, per candidate) is part of the contract — it makes
    runs seed-exact reproducible against the legacy ``run_card_auction``.
    """

    model_config = ConfigDict(frozen=True)

    baseline_prior: tuple[float, float] = (3.0, 3.0)

    def run(
        self, candidates: list[tuple[str, float, float]], rng: Any
    ) -> tuple[list[str], list[dict]]:
        base_a, base_b = self.baseline_prior
        winners: list[str] = []
        records: list[dict] = []
        for card_id, a, b in candidates:
            theta = float(rng.beta(a, b))
            base_theta = float(rng.beta(base_a, base_b))
            selected = theta > base_theta
            if selected:
                winners.append(card_id)
            records.append(
                {
                    "card_id": card_id,
                    "a": float(a),
                    "b": float(b),
                    "theta": theta,
                    "baseline_a": float(base_a),
                    "baseline_b": float(base_b),
                    "baseline_theta": base_theta,
                    "selected": selected,
                }
            )
        if records:
            logger.debug(
                "[Memory][Auction] {}/{} candidate(s) beat baseline Beta{}: {}",
                len(winners),
                len(records),
                self.baseline_prior,
                "; ".join(
                    "{} a/b={:.3g}/{:.3g} theta={:.3f} base={:.3f} {}".format(
                        r["card_id"],
                        r["a"],
                        r["b"],
                        r["theta"],
                        r["baseline_theta"],
                        "WIN" if r["selected"] else "lose",
                    )
                    for r in records
                ),
            )
        return winners, records
