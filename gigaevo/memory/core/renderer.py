from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from gigaevo.memory.shared_memory.card_search import format_block_efficacy
from gigaevo.memory.shared_memory.models import AnyCard, CardStatsBlock


class EfficacyCardRenderer(BaseModel):
    """Renders one card into the mutator-facing text block: description and the
    efficacy endorsement line (single source:
    ``card_search.format_block_efficacy``)."""

    model_config = ConfigDict(frozen=True)

    def render(self, card: AnyCard | None, block: CardStatsBlock | None = None) -> str:
        if card is None:
            return ""
        description = card.description.strip()
        lines = [description] if description else []
        efficacy = format_block_efficacy(card, block)
        if efficacy:
            lines.append(efficacy)
        return "\n".join(lines)
