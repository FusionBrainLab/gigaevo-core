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
from statistics import median

from pydantic import ConfigDict, Field

from gigaevo.evolution.strategies.models import BehaviorSpace
from gigaevo.memory.context import ContextualGain, DecisionContext
from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.core.reward import AbsoluteMedianReward, RewardWeightedReputation
from gigaevo.memory.shared_memory.models import AnyCard


class BDProximityReputation(BetaBinomialReputation):
    """Beta-Binomial reputation whose read-side value channel is partitioned by
    the query parent's BD cell. Write-side aggregation (posterior, harm predicate,
    scorer) is inherited unchanged; only the per-call ``card_posterior`` /
    ``card_magnitude`` reads switch to the in-cell subset, with a ``fallback``
    reputation for cold cells."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    behavior_space: BehaviorSpace = Field(
        description="The run's tessellation; bucketing reads its CURRENT bounds.",
    )
    fallback: BetaBinomialReputation = Field(
        default_factory=lambda: RewardWeightedReputation(reward=AbsoluteMedianReward()),
        description="Cold-cell delegate (default arm-D absolute-median reputation).",
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
        events = card.evolution_statistics.gain_events
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

    def card_posterior(
        self, card: AnyCard, context: DecisionContext | None = None
    ) -> tuple[float, float]:
        in_cell = self._in_cell(card, context)
        if in_cell is None:
            return self.fallback.card_posterior(card)
        n_harm = sum(1 for e in in_cell if e.gain < 0 or e.invalid)
        n_help = len(in_cell) - n_harm
        return (1.0 + n_help, 1.0 + n_harm)

    def card_magnitude(
        self, card: AnyCard, context: DecisionContext | None = None
    ) -> float | None:
        in_cell = self._in_cell(card, context)
        if in_cell is None:
            return self.fallback.card_magnitude(card)
        # Value channel measures the gain of VALID outcomes only — invalid children
        # are harm in the posterior, not zero-gain successes; counting their stamped
        # 0.0 here would drag a harmful card's median toward non-negative. All-invalid
        # in-cell -> 0.0 (abstains under the default ev_floor), not the cold prior.
        # Non-finite gains (corrupt/legacy persisted events) are dropped: the
        # auction has no inf defense and median([inf, ...]) would win every slot.
        valid = [e.gain for e in in_cell if not e.invalid and isfinite(e.gain)]
        if not valid:
            return 0.0
        return float(median(valid))
