from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.core.events import emit_memory_event
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
        return self.reputation.is_confidently_harmful(
            self.reputation.card_stats(card, None)
        )

    def sweep(self, bank: Mapping[str, AnyCard]) -> list[str]:
        evicted = [cid for cid, card in bank.items() if self.should_evict(card)]
        if evicted:
            emit_memory_event(
                component="Evictor",
                event_type="evictor.sweep",
                payload={
                    "bank_count": len(bank),
                    "evicted_count": len(evicted),
                    "evicted_ids": evicted,
                },
                level="INFO",
            )
            logger.info(
                "[Memory][Evictor] Sweep evicting {}/{} card(s) as confidently harmful: {}",
                len(evicted),
                len(bank),
                evicted,
            )
        return evicted


class NullEvictor(BaseModel):
    """No-op evictor: never evicts. Selectable via ``writer/evictor=none`` to run
    the write path with the harm sweep disabled, the bank-maintenance twin of
    ``memory=none`` on the read side."""

    model_config = ConfigDict(frozen=True)

    reputation: Any = Field(
        default=None,
        description="Ignored; accepted only so MemorySystem completes this leaf with the shared reputation exactly as it does HarmEvictor.",
    )

    def should_evict(self, card: AnyCard) -> bool:
        return False

    def sweep(self, bank: Mapping[str, AnyCard]) -> list[str]:
        return []
