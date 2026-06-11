from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict

from gigaevo.memory.shared_memory.card_dedup import DedupAction, DedupDecision
from gigaevo.memory.shared_memory.card_update_dedup import CardUpdateDedupConfig


def _add(reason: str) -> DedupDecision:
    return DedupDecision(
        action=DedupAction.ADD, reason=reason, duplicate_of="", merges=[]
    )


class NullDeduplicator(BaseModel):
    """Every incoming card is a fresh add."""

    model_config = ConfigDict(frozen=True)

    def reconcile(self, card: Any, bank: Mapping[str, Any]) -> DedupDecision:
        return _add("dedup disabled")


class LLMDeduplicator:
    """Wraps the legacy CardDedup engine, owning the readiness gating that
    ``save_card`` performed inline: not configured / disabled / empty bank /
    no LLM all degrade to a plain add."""

    def __init__(
        self,
        config: CardUpdateDedupConfig | None = None,
        engine: Any = None,
    ) -> None:
        self.config = config if config is not None else CardUpdateDedupConfig()
        self.engine = engine
        self._warned_no_llm = False

    def reconcile(self, card: Any, bank: Mapping[str, Any]) -> DedupDecision:
        if self.engine is None:
            return _add("dedup engine unavailable")
        if not self.engine.config.enabled or not bank:
            return _add("dedup not applicable")
        if self.engine.llm_service is None:
            if not self._warned_no_llm:
                logger.warning(
                    "[Memory][CardUpdateDedup] card_update_dedup enabled but LLM unavailable; "
                    "falling back to regular save_card."
                )
                self._warned_no_llm = True
            return _add("dedup LLM unavailable")
        return self.engine.run_dedup_on_incoming_card(card)
