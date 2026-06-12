from __future__ import annotations

from collections.abc import Mapping

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.shared_memory.models import AnyCard


class HarmEvictor(BaseModel):
    """Evicts cards whose injection posterior is confidently harmful — the
    write-side twin of the legacy ``save_card`` harm gate."""

    model_config = ConfigDict(frozen=True)

    reputation: BetaBinomialReputation = Field(
        default_factory=BetaBinomialReputation,
        description="Posterior model deciding the confidently-harmful verdict.",
    )

    def should_evict(self, card: AnyCard) -> bool:
        return self.reputation.is_confidently_harmful(card.evolution_statistics)

    def sweep(self, bank: Mapping[str, AnyCard]) -> list[str]:
        evicted = [cid for cid, card in bank.items() if self.should_evict(card)]
        if evicted:
            logger.info(
                "[Memory][Evictor] Sweep evicting {}/{} card(s) as confidently harmful: {}",
                len(evicted),
                len(bank),
                evicted,
            )
        return evicted
