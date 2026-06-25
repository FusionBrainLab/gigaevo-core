"""Read-time BD-cell partitioned reputation (the contextual bandit's value channel).

``BDProximityReputation`` re-buckets each card's stored ``gain_events`` into the
query parent's *current* MAP-Elites cell via the run's own ``behavior_space.get_cell``
and bids over the in-cell subset only — a card that helped near cell A and hurt
near cell B bids high in A and abstains in B from the same stored list. A parent
cell with no in-cell event delegates byte-for-byte to ``fallback`` (today's
numbers, no regression).

The cell is recomputed every read from the immutable ``parent_metrics`` under the
held ``behavior_space``'s current bounds — the bandit reads the one tessellation,
never stores a cell id (``DynamicBehaviorSpace`` moves cells on every reindex).
"""

from __future__ import annotations

from math import isfinite

from pydantic import ConfigDict, Field

from gigaevo.evolution.strategies.models import BehaviorSpace
from gigaevo.memory.context import ContextualGain, DecisionContext
from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.efficacy import block_from_events
from gigaevo.memory.shared_memory.models import AnyCard, CardStatsBlock


class BDProximityReputation(BetaBinomialReputation):
    """Beta-Binomial reputation whose read-side value channel is partitioned by
    the query parent's BD cell. ``card_stats`` recomputes the in-cell block (the
    same MAD-noise-band harm predicate as the global path, measured over in-cell
    gains rather than the parent-fitness counterfactual); ``card_posterior`` and
    ``card_magnitude`` are views over it. A cold cell delegates to ``fallback``."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    behavior_space: BehaviorSpace = Field(
        description="The run's tessellation; bucketing reads its CURRENT bounds.",
    )
    fallback: BetaBinomialReputation = Field(
        default_factory=BetaBinomialReputation,
        description="Cold-cell delegate: the global event-derived reputation.",
    )

    def _cell(self, metrics: dict[str, float]) -> tuple[int, ...] | None:
        # A missing or non-finite behavior coord has no well-defined cell —
        # LinearBinning silently clamps NaN to bin 0, so guard here and abstain
        # to fallback rather than credit events to a spurious low-end cell.
        for key in self.behavior_space.behavior_keys:
            value = metrics.get(key)
            if value is None or not isfinite(value):
                return None
        return self.behavior_space.get_cell(metrics)

    def _in_cell(
        self, card: AnyCard, context: DecisionContext | None
    ) -> list[ContextualGain] | None:
        if context is None:
            return None
        events = card.gain_events
        if not events:
            return None
        parent_cell = self._cell(context.parent_metrics)
        if parent_cell is None:
            return None
        in_cell = [
            event
            for event in events
            if self._cell(event.context.parent_metrics) == parent_cell
        ]
        return in_cell or None

    def card_stats(
        self, card: AnyCard, context: DecisionContext | None = None
    ) -> CardStatsBlock | None:
        in_cell = self._in_cell(card, context)
        if in_cell is None:
            return self.fallback.card_stats(card, context)
        # Same global block math as the base reputation, but over the in-cell
        # subset only: the cell partition already controls for context, so the
        # MAD harm band and median magnitude are measured BD-locally rather than
        # against a parent-fitness counterfactual. Cold cells delegated above.
        return block_from_events(
            in_cell,
            noise_band_k=self.noise_band_k,
            confident_quantile=self.confident_quantile,
            confident_threshold=self.confident_threshold,
        )
