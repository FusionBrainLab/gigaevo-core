"""Mutator-facing card rendering: description plus the efficacy endorsement."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from gigaevo.memory.cards import Card, CardKind, CardStatsBlock


def format_block_efficacy(card: Card, block: CardStatsBlock | None) -> str:
    """One legible per-card endorsement line for the mutator, or empty.

    ``block`` is the reputation's ``card_stats`` for the decision context, so
    the rendered line reflects the same locality the auction bid on —
    cell-local under BD proximity, global otherwise. Insight cards: rendered
    only when the Beta-Binomial downside posterior is confident —
    non-confident and no-signal cards stay silent (description only). Program
    cards: exemplar fitness (block-independent).
    """
    if card.kind is CardKind.PROGRAM:
        if card.fitness is None:
            return ""
        return f"efficacy: exemplar fitness {card.fitness:.4f}"

    if block is None:
        return ""
    intros = block.intro_events
    median = block.IntroGain_best_median
    if intros <= 0 or median is None:
        return ""
    if not block.efficacy_confident:
        return ""
    # Gains are stored in "positive = improvement" space regardless of metric
    # direction (extraction negates for minimize metrics), so the wording must
    # be direction-neutral — "fitness change +x" would read inverted on minimize.
    line = (
        f"efficacy: introduced in {intros} children; "
        f"median improvement {float(median):+.4f}"
    )
    # A noise-band-confident posterior with a losing median must never read as
    # an endorsement.
    if float(median) <= 0:
        return line + " (caution: non-positive median)"
    return line + " (confident)"


class EfficacyCardRenderer(BaseModel):
    """Renders one card into the mutator-facing text block: description and the
    efficacy endorsement line (single source: :func:`format_block_efficacy`)."""

    model_config = ConfigDict(frozen=True)

    def render(self, card: Card | None, block: CardStatsBlock | None = None) -> str:
        if card is None:
            return ""
        description = card.description.strip()
        lines = [description] if description else []
        efficacy = format_block_efficacy(card, block)
        if efficacy:
            lines.append(efficacy)
        return "\n".join(lines)
