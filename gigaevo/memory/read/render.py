"""Mutator-facing card rendering: description plus the efficacy endorsement."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.cards import Card, CardKind, CardStatsBlock


def _format_event_count(value: float) -> str:
    count = float(value)
    if count.is_integer():
        return str(int(count))
    return f"{count:.2f}".rstrip("0").rstrip(".")


def format_block_efficacy(card: Card, block: CardStatsBlock | None) -> str:
    """One legible per-card endorsement line for the mutator, or empty.

    ``block`` is the reputation's ``card_stats`` for the decision context, so
    the rendered line reflects the same locality the auction bid on —
    cell-local under BD proximity, global otherwise. Insight cards: rendered
    only when the Beta-Binomial downside posterior is confident —
    non-confident and no-signal cards stay silent (description only). Program
    cards: exemplar fitness (block-independent).
    """
    foreign_line = ""
    if block is not None and block.foreign_total_events > 0.0:
        foreign_line = (
            f"helped in {_format_event_count(block.foreign_help_events)} of "
            f"{_format_event_count(block.foreign_total_events)} uses on other tasks"
        )

    if card.kind is CardKind.PROGRAM:
        if card.fitness is None:
            return foreign_line
        local_line = f"efficacy: exemplar fitness {card.fitness:.4f}"
        return "\n".join(line for line in (local_line, foreign_line) if line)

    if block is None:
        return foreign_line
    intros = block.intro_events
    value = block.IntroGain_bootstrap_ev_mean
    is_bootstrap_ev = value is not None
    if value is None:
        value = block.IntroGain_best_median
    if intros <= 0 or value is None:
        return foreign_line
    if not block.efficacy_confident:
        return foreign_line
    # Gains are stored in "positive = improvement" space regardless of metric
    # direction (extraction negates for minimize metrics), so the wording must
    # be direction-neutral — "fitness change +x" would read inverted on minimize.
    label = "expected improvement" if is_bootstrap_ev else "median improvement"
    line = (
        f"efficacy: introduced in {_format_event_count(intros)} children; "
        f"{label} {float(value):+.4f}"
    )
    # An efficacy-confident posterior with a losing value must never read as an
    # endorsement.
    if float(value) <= 0:
        descriptor = "expected improvement" if is_bootstrap_ev else "median"
        local_line = line + f" (caution: non-positive {descriptor})"
    else:
        local_line = line + " (confident)"
    return "\n".join(line for line in (local_line, foreign_line) if line)


class EfficacyCardRenderer(BaseModel):
    """Renders one card into the mutator-facing text block: description and the
    efficacy endorsement line (single source: :func:`format_block_efficacy`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_key: str = Field(default="", description="Current run's task key.")

    def render(self, card: Card | None, block: CardStatsBlock | None = None) -> str:
        if card is None:
            return ""
        description = card.description.strip()
        lines = [description] if description else []
        foreign = bool(
            self.task_key and card.task_key and self.task_key != card.task_key
        )
        # A foreign exemplar's fitness is on another task's scale — render it
        # fitness-less so only the sign-only foreign help line survives.
        efficacy_card = (
            card.model_copy(update={"fitness": None})
            if foreign and card.kind is CardKind.PROGRAM
            else card
        )
        efficacy = format_block_efficacy(efficacy_card, block)
        if efficacy:
            lines.append(efficacy)
        if foreign:
            lines.append(f"evidence from a different task ({card.task_key})")
        return "\n".join(lines)
