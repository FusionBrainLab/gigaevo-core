"""AOS credit-assignment reward for the EV-bid magnitude.

The auction bids ``theta x magnitude``; ``magnitude`` is the card's expected gain.
A ``RewardDefinition`` is the credit-assignment strategy that turns a card's
stamped ``CardStatsBlock`` into that magnitude — the swappable "reward" half of
the canonical AOS ``CreditAssignment ∘ OperatorSelection`` decomposition
(``plans/aos-credit-selection-architecture.md``).

``AdjustedMedianReward`` (default) reproduces today's cohort-adjusted read.
``AbsoluteMedianReward`` reads the absolute child-minus-parent median, so a card
that never beats its parent carries a non-positive bid and the auctioneer's
existing ``ev_floor`` abstains on it — closing the §L1 loophole
(``docs/audits/heilbron_decision_pipeline_integrity_2026-06-24.md``) with no
change to ``EVThompsonAuctioneer`` or ``BetaBinomialReputation``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import ConfigDict, Field

from gigaevo.memory.context import DecisionContext
from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.shared_memory.models import AnyCard, CardStatsBlock


class RewardDefinition(ABC):
    """Maps a card's stamped statistics to its EV-bid magnitude (``None`` cold)."""

    @abstractmethod
    def magnitude(self, block: CardStatsBlock) -> float | None: ...


class AdjustedMedianReward(RewardDefinition):
    """Today's reward: the cohort-adjusted child-minus-parent median gain."""

    def magnitude(self, block: CardStatsBlock) -> float | None:
        if block.IntroGain_best_adj_median is None:
            return None
        return float(block.IntroGain_best_adj_median)


class AbsoluteMedianReward(RewardDefinition):
    """Absolute objective: the raw child-minus-best-parent median gain. A card
    whose children never beat their parent bids non-positive (§L1 fix)."""

    def magnitude(self, block: CardStatsBlock) -> float | None:
        if block.IntroGain_best_median is None:
            return None
        return float(block.IntroGain_best_median)


class RewardWeightedReputation(BetaBinomialReputation):
    """``BetaBinomialReputation`` whose EV magnitude is supplied by a pluggable
    ``RewardDefinition`` instead of the hardwired cohort-adjusted median. Every
    other reputation behaviour (posterior, harm predicate, scorer) is inherited
    unchanged."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    reward: RewardDefinition = Field(
        default_factory=AdjustedMedianReward,
        description="Credit-assignment strategy turning the card's ALL block into "
        "the EV-bid magnitude. Default reproduces BetaBinomialReputation.",
    )

    def card_magnitude(
        self, card: AnyCard, context: DecisionContext | None = None
    ) -> float | None:
        block = card.evolution_statistics.ALL
        if block is None:
            return None
        return self.reward.magnitude(block)
