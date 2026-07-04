"""Memory domain models — the single ``Card`` type and its value objects.

Leaf layer of the memory package: pydantic + stdlib only. Everything above
(storage, read, write) speaks ``Card``; behavioral differences between insight
and program-exemplar cards are driven by ``card.kind``, never by type dispatch.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator
from pydantic_core.core_schema import SerializerFunctionWrapHandler


class CardKind(StrEnum):
    INSIGHT = "insight"
    PROGRAM = "program"


class DecisionContext(BaseModel):
    """The state a card-injection decision was made in.

    The base parent's id and metrics plus the crediting child's creation time —
    enough to identify which parent the decision was made against and to order
    events over the run. The extension point for richer contexting later.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    parent_metrics: dict[str, float] = Field(default_factory=dict)
    parent_id: str = Field(
        default="", description="Base parent's program id (whose metrics these are)."
    )
    timestamp: datetime | None = Field(
        default=None, description="Crediting child's creation time (UTC)."
    )


class ContextualGain(BaseModel):
    """One credited injection event: the gain a card earned in a context.

    "Gain" is always the child-minus-parent best-fitness delta in
    positive-is-improvement space (negated for minimize metrics).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    context: DecisionContext
    gain: float
    invalid: bool = Field(
        default=False,
        description="True for an evaluated-and-judged-invalid child — a forced "
        "harm event whose gain magnitude is meaningless.",
    )
    founding: bool = Field(
        default=False,
        description="True for the event seeded at authoring from the parent-child "
        "pair the card was distilled from. The founding child predates the card, "
        "so use-attribution can never re-credit it; it is preserved across the "
        "from-scratch restamp rather than recomputed each sweep.",
    )


class DecisionMetrics(BaseModel):
    """Efficacy metrics that decision paths read.

    Exactly the fields the Thompson auction, the reputation harm predicate, and
    the prompt renderer consume — the vocabulary reputation computes from a
    card's gain events, and nothing more. Field names are the serialized-card
    contract, including the mixed-case ``IntroGain_*`` keys.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    posterior_a: float | None = Field(
        default=None,
        description="Beta alpha of the downside posterior over per-introduction gains.",
    )
    posterior_b: float | None = Field(
        default=None, description="Beta beta of the downside posterior."
    )
    intro_events: int = Field(
        default=0,
        description="Introduction events (scorable children) backing this row.",
    )
    k_harm: int | None = Field(
        default=None,
        description="Introduction events whose baseline-adjusted gain was "
        "negative (harm, a strict sign test).",
    )
    p_help_mean: float | None = Field(
        default=None, description="Posterior mean P(gain >= threshold), a / (a + b)."
    )
    p_help_lo20: float | None = Field(
        default=None, description="20th-percentile lower credible bound of P(help)."
    )
    efficacy_confident: bool | None = Field(
        default=None,
        description="True when the lower credible bound clears the confidence threshold.",
    )
    IntroGain_best_median: float | None = Field(
        default=None, description="Median raw child-minus-parent best-fitness gain."
    )


class CardStatsBlock(DecisionMetrics):
    """A card's efficacy-statistics block, computed by reputation from the
    card's gain events."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_serializer(mode="wrap")
    def serialize_without_unset_defaults(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        """Serialize exactly the keys the source block carried: explicitly set
        fields (including explicit nulls) plus extras; unset defaults stay out
        so a serialized card block roundtrips to its original keys."""
        declared = type(self).model_fields
        return {
            key: value
            for key, value in handler(self).items()
            if key in self.model_fields_set or key not in declared
        }


class Card(BaseModel):
    """The one memory card.

    ``kind`` distinguishes distilled insights from program exemplars; the
    exemplar-only fields (``program_id``, ``code``, ``fitness``) are kind-gated
    so an insight card can never smuggle them in. Cards are frozen — the write
    path evolves them via ``model_copy(update=...)``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(description="Stable bank id of the card.")
    kind: CardKind = Field(default=CardKind.INSIGHT)
    category: str = Field(
        default="general",
        description="Free-form topical category assigned by the authoring librarian.",
    )
    description: str = Field(
        default="", description="The idea itself — the text injected into prompts."
    )
    explanation_summary: str = Field(
        default="",
        description="One-line condensed reason the lever works; a distinct "
        "retrieval channel from the fuller description.",
    )
    task_description: str = Field(
        default="", description="Task description of the run that produced the card."
    )
    task_description_summary: str = Field(
        default="", description="LLM-condensed one-line task summary."
    )
    keywords: tuple[str, ...] = Field(
        default=(), description="Search keywords for retrieval ranking."
    )
    programs: tuple[str, ...] = Field(
        default=(), description="Program ids that exhibited the idea."
    )
    absorbed_ids: tuple[str, ...] = Field(
        default=(),
        description="Bank ids merged/consolidated into this survivor; children's "
        "frozen card_ids_used pointing at them re-alias here at restamp.",
    )
    gain_events: tuple[ContextualGain, ...] = Field(
        default=(),
        description="Use-attributed base-relative injection events; reputation "
        "computes this card's efficacy block from them.",
    )
    program_id: str = Field(
        default="",
        description="Exemplar program's id in the run database (kind=program only).",
    )
    code: str = Field(
        default="", description="Exemplar program's source code (kind=program only)."
    )
    fitness: float | None = Field(
        default=None,
        description="Exemplar fitness at capture time (kind=program only).",
    )

    @model_validator(mode="after")
    def _gate_kind_fields(self) -> Card:
        if self.kind is CardKind.PROGRAM:
            if not self.program_id:
                raise ValueError("kind=program requires a non-empty program_id")
        elif self.program_id or self.code or self.fitness is not None:
            raise ValueError(
                "program_id/code/fitness are exemplar fields — set kind=program"
            )
        if self.id and self.id in self.absorbed_ids:
            raise ValueError("a card cannot absorb its own id")
        return self


def card_brief(card: Card) -> str:
    """Compact card projection for the librarian judging prompts (reconcile /
    consolidate): description + why-text + keywords on one line, empty fields
    omitted. The reconcile caller prepends the id (it needs it as the
    DUPLICATE/MERGE target); consolidate uses the body alone.
    """
    parts = [card.description]
    why = card.explanation_summary.strip()
    if why:
        parts.append(f"why: {why}")
    if card.keywords:
        parts.append(f"keywords: {', '.join(card.keywords)}")
    return " | ".join(parts)
