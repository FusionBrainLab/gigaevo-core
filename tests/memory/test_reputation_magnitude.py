"""Per-card EV magnitude read for the EV-bid auction.

``BetaBinomialReputation.card_magnitude`` surfaces the card's stamped
``IntroGain_best_adj_median`` (the cohort-adjusted expected gain) so the EV
auction can bid ``theta x magnitude``. Cards with no ALL block, or an ALL
block that never stamped the field, are cold (``None``) and fall back to the
auction's optimistic prior.
"""

from __future__ import annotations

from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.shared_memory.models import (
    MemoryCard,
    MemoryCardExplanation,
)


def _card(adj_median: float | None = None, *, has_all: bool = True) -> MemoryCard:
    all_block: dict = {}
    if adj_median is not None:
        all_block["IntroGain_best_adj_median"] = adj_median
    es = {"ALL": all_block} if has_all else {}
    return MemoryCard(
        id="m1",
        description="d",
        keywords=[],
        evolution_statistics=es,
        explanation=MemoryCardExplanation(summary=""),
    )


class TestCardMagnitude:
    def test_reads_stamped_adjusted_gain(self) -> None:
        assert BetaBinomialReputation().card_magnitude(_card(0.0123)) == 0.0123

    def test_negative_magnitude_passes_through(self) -> None:
        assert BetaBinomialReputation().card_magnitude(_card(-0.05)) == -0.05

    def test_no_all_block_is_cold_none(self) -> None:
        assert BetaBinomialReputation().card_magnitude(_card(has_all=False)) is None

    def test_all_block_without_field_is_cold_none(self) -> None:
        assert BetaBinomialReputation().card_magnitude(_card(None)) is None
