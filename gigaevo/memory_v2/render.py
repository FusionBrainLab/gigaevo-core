"""Fixed treatment renderer for memory v2."""

from __future__ import annotations

from gigaevo.memory_v2.models import CardSnapshot


class ImmutableCardRenderer:
    """Return exactly the content-addressed payload, with no posterior leakage."""

    def render(self, card: CardSnapshot) -> str:
        return card.delivered_text
