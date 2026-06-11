from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.core.reputation import BetaBinomialReputation


class HarmEvictor(BaseModel):
    """Evicts cards whose injection posterior is confidently harmful — the
    write-side twin of the legacy ``save_card`` harm gate."""

    model_config = ConfigDict(frozen=True)

    reputation: BetaBinomialReputation = Field(default_factory=BetaBinomialReputation)

    def should_evict(self, card: Any) -> bool:
        if isinstance(card, Mapping):
            stats = card.get("evolution_statistics")
        else:
            stats = getattr(card, "evolution_statistics", None)
        return self.reputation.is_confidently_harmful(stats)

    def sweep(self, bank: Mapping[str, Any]) -> list[str]:
        evicted = [cid for cid, card in bank.items() if self.should_evict(card)]
        if evicted:
            logger.info(
                "[Memory][Evictor] Sweep evicting {}/{} card(s) as confidently harmful: {}",
                len(evicted),
                len(bank),
                evicted,
            )
        return evicted
