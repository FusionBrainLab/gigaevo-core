from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from gigaevo.memory.shared_memory.card_search import format_card_efficacy
from gigaevo.memory.shared_memory.models import AnyCard, MemoryCard


class EfficacyCardRenderer(BaseModel):
    """Renders one card into the mutator-facing text block: description,
    mechanism (when it adds information), and the efficacy endorsement line
    (single source: ``card_search.format_card_efficacy``)."""

    model_config = ConfigDict(frozen=True)

    def render(self, card: AnyCard | None) -> str:
        if card is None:
            return ""
        description = card.description.strip()
        mechanism = (
            card.explanation.summary.strip() if isinstance(card, MemoryCard) else ""
        )
        lines = [description] if description else []
        if mechanism and mechanism != description:
            lines.append(f"mechanism: {mechanism}")
        efficacy = format_card_efficacy(card)
        if efficacy:
            lines.append(efficacy)
        return "\n".join(lines)
