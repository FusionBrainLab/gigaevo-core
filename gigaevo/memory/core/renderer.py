from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

from gigaevo.memory.shared_memory.card_search import format_card_efficacy


class EfficacyCardRenderer(BaseModel):
    """Renders one card into the mutator-facing text block: description,
    mechanism (when it adds information), and the efficacy endorsement line
    (single source: ``card_search.format_card_efficacy``)."""

    model_config = ConfigDict(frozen=True)

    def render(self, card: Any) -> str:
        if card is None:
            return ""
        if isinstance(card, dict):
            description = str(card.get("description") or "")
            explanation = card.get("explanation")
        else:
            description = str(getattr(card, "description", "") or "")
            explanation = getattr(card, "explanation", None)
        description = description.strip()
        if isinstance(explanation, Mapping):
            mechanism = str(explanation.get("summary") or "").strip()
        else:
            mechanism = str(getattr(explanation, "summary", "") or "").strip()
        lines = [description] if description else []
        if mechanism and mechanism != description:
            lines.append(f"mechanism: {mechanism}")
        efficacy = format_card_efficacy(card)
        if efficacy:
            lines.append(efficacy)
        return "\n".join(lines)
