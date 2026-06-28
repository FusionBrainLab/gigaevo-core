from __future__ import annotations

from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_serializer
from pydantic_core.core_schema import SerializerFunctionWrapHandler

from gigaevo.memory.context import ContextualGain

Strategy = Literal["exploration", "exploitation", "hybrid"]


class DecisionMetrics(BaseModel):
    """Efficacy metrics that decision paths read.

    Exactly the fields the Thompson auction, the reputation harm predicate, and
    the prompt renderer consume — the vocabulary reputation computes from a
    card's gain events, and nothing more. Field names are the serialized-card
    contract, including the mixed-case ``IntroGain_*`` keys.

    "Gain" is always the child-minus-parent best-fitness delta in
    positive-is-improvement space (negated for minimize metrics).
    """

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
        description="Introduction events whose baseline-adjusted gain fell below the negative noise-band threshold.",
    )
    p_help_mean: float | None = Field(
        default=None, description="Posterior mean P(gain >= threshold), a / (a + b)."
    )
    p_help_lo20: float | None = Field(
        default=None,
        description="20th-percentile lower credible bound of P(help).",
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
    card's gain events.

    The metric vocabulary is exactly :class:`DecisionMetrics` — what decision
    paths read.
    """

    model_config = ConfigDict(extra="forbid")

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


class MemoryCard(BaseModel):
    """Canonical general memory card (ideas, insights)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(description="Stable bank id of the card.")
    category: str = Field(
        default="general",
        description="Free-form topical category assigned by the authoring librarian agent.",
    )
    description: str = Field(
        default="", description="The idea itself — the text injected into prompts."
    )
    explanation_summary: str = Field(
        default="",
        description="One-line condensed reason the lever works; a distinct retrieval channel from the fuller description.",
    )
    task_description: str = Field(
        default="", description="Task description of the run that produced the card."
    )
    task_description_summary: str = Field(
        default="", description="LLM-condensed one-line task summary."
    )
    programs: list[str] = Field(
        default_factory=list, description="Program ids that exhibited the idea."
    )
    keywords: list[str] = Field(
        default_factory=list, description="Search keywords for retrieval ranking."
    )
    gain_events: list[ContextualGain] | None = Field(
        default=None,
        description="Use-attributed base-relative injection events; reputation computes this card's efficacy block from them.",
    )


class ProgramCard(BaseModel):
    """Memory card representing a top-performing program."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(description="Stable bank id of the card.")
    category: str = Field(
        default="program",
        description="Always 'program' — distinguishes exemplar cards from idea cards.",
    )
    program_id: str = Field(
        default="", description="Id of the exemplar program in the run database."
    )
    task_description: str = Field(
        default="", description="Task description of the run that produced the card."
    )
    task_description_summary: str = Field(
        default="", description="LLM-condensed one-line task summary."
    )
    description: str = Field(
        default="", description="What the exemplar program does and why it scores well."
    )
    explanation_summary: str = Field(
        default="",
        description="One-line condensed reason the exemplar scores well; a distinct retrieval channel from the fuller description.",
    )
    fitness: float | None = Field(
        default=None, description="Fitness of the exemplar program at capture time."
    )
    code: str = Field(default="", description="Source code of the exemplar program.")
    keywords: list[str] = Field(
        default_factory=list, description="Search keywords for retrieval ranking."
    )
    gain_events: list[ContextualGain] | None = Field(
        default=None,
        description="Use-attributed base-relative injection events; reputation computes this card's efficacy block from them.",
    )


AnyCard = MemoryCard | ProgramCard
CardT = TypeVar("CardT", bound=AnyCard)


__all__ = [
    "AnyCard",
    "CardT",
    "CardStatsBlock",
    "DecisionMetrics",
    "MemoryCard",
    "ProgramCard",
    "Strategy",
]
