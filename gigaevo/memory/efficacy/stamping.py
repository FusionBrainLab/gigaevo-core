"""Stamping: project efficacy evidence onto cards and their statistics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from gigaevo.memory.shared_memory.models import (
    CardStatsBlock,
    EvolutionStatistics,
    Quartile,
)

if TYPE_CHECKING:
    from gigaevo.memory.core.idea_stats import IdeaStats
    from gigaevo.memory.shared_memory.models import AnyCard


class CardStatsStamper(BaseModel):
    """Single writer of card-side efficacy statistics.

    Cards carry only the whole-run ``ALL`` block in the decision vocabulary;
    per-quartile rows stay in the offline summary. Both producers go through
    here: the origin-analysis rows (``idea_statistics``) and the injection
    posterior (``stamp_posterior``).
    """

    model_config = ConfigDict(frozen=True)

    def idea_statistics(
        self, rows: Sequence[IdeaStats]
    ) -> dict[str, EvolutionStatistics]:
        """Per-idea card statistics from the analysis summary: each idea's
        ALL row becomes its card's ALL block."""
        return {
            row.idea_id: self.harm_statistics(row)
            for row in rows
            if row.quartile is Quartile.ALL
        }

    def harm_statistics(self, row: IdeaStats) -> EvolutionStatistics:
        """Lift one origin-analysis row into the typed statistics shape the
        reputation harm predicate reads (the row becomes the ALL block)."""
        return EvolutionStatistics(ALL=row.to_stats_block())

    def stamp_posterior(
        self, card: AnyCard, posteriors: dict[str, CardStatsBlock]
    ) -> AnyCard:
        """Card with its injection posterior written into the ALL block;
        cards without a posterior pass through unchanged."""
        posterior = posteriors.get(card.id.strip())
        if posterior is None:
            return card
        stamped = card.evolution_statistics.model_copy(update={"ALL": posterior})
        return card.model_copy(update={"evolution_statistics": stamped})
