from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer
from pydantic_core.core_schema import SerializerFunctionWrapHandler

from gigaevo.memory.context import ContextualGain

Strategy = Literal["exploration", "exploitation", "hybrid"]


class DecisionMetrics(BaseModel):
    """Efficacy metrics that decision paths read.

    Exactly the fields the Thompson auction, the reputation harm predicate, and
    the prompt renderer consume — the vocabulary reputation computes from a
    card's gain events, and nothing more. Field names are the banks.json
    contract, including the analyzer-cased ``IntroGain_*`` keys.

    "Gain" is always the child-minus-parent best-fitness delta in
    positive-is-improvement space (the analysis negates for minimize metrics).
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
        so a banks.json block roundtrips to its original keys."""
        declared = type(self).model_fields
        return {
            key: value
            for key, value in handler(self).items()
            if key in self.model_fields_set or key not in declared
        }


class CardAlias(BaseModel):
    """Archived superseded version of a card: when a merge or update replaces
    the description, the previous wording is preserved here."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(description="Archive key, e.g. '<card_id>-update'.")
    description: str = Field(description="The superseded description text.")
    programs: list[str] = Field(
        default_factory=list,
        description="Program ids the card referenced at archival time.",
    )
    explanations: list[str] = Field(
        default_factory=list,
        description="Explanation entries the card carried at archival time.",
    )


class MemoryCardExplanation(BaseModel):
    """Explanation field with history and summary."""

    model_config = ConfigDict(extra="forbid")

    explanations: list[str] = Field(
        default_factory=list,
        description="Accumulated explanation entries, oldest first.",
    )
    summary: str = Field(
        default="", description="LLM-condensed summary of the explanation history."
    )


class MemoryCard(BaseModel):
    """Canonical general memory card (ideas, insights)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str = Field(description="Stable bank id of the card.")
    category: str = Field(
        default="general",
        description="Free-form topical category assigned by the producing analyzer.",
    )
    description: str = Field(
        default="", description="The idea itself — the text injected into prompts."
    )
    task_description: str = Field(
        default="", description="Task description of the run that produced the card."
    )
    task_description_summary: str = Field(
        default="", description="LLM-condensed one-line task summary."
    )
    strategy: str = Field(
        default="", description="Mutation archetype the idea originated from."
    )
    last_generation: int = Field(
        default=0, description="Last generation in which the idea was observed."
    )
    programs: list[str] = Field(
        default_factory=list, description="Program ids that exhibited the idea."
    )
    aliases: list[CardAlias] = Field(
        default_factory=list,
        description="Archived superseded versions of this card.",
    )
    keywords: list[str] = Field(
        default_factory=list, description="Search keywords for retrieval ranking."
    )
    gain_events: list[ContextualGain] | None = Field(
        default=None,
        description="Use-attributed base-relative injection events; reputation computes this card's efficacy block from them.",
    )
    explanation: MemoryCardExplanation = Field(
        default_factory=MemoryCardExplanation,
        description="Why the idea works, with history and summary.",
    )
    works_with: list[str] = Field(
        default_factory=list, description="Ids of cards observed to combine well."
    )
    links: list[str] = Field(
        default_factory=list, description="Ids of semantically related cards."
    )


class ConnectedIdea(BaseModel):
    """Reference to an idea card linked to a program."""

    model_config = ConfigDict(extra="forbid")

    idea_id: str = Field(default="", description="Bank id of the linked idea card.")
    description: str = Field(
        default="", description="Description of the linked idea at link time."
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
    fitness: float | None = Field(
        default=None, description="Fitness of the exemplar program at capture time."
    )
    code: str = Field(default="", description="Source code of the exemplar program.")
    connected_ideas: list[ConnectedIdea] = Field(
        default_factory=list,
        description="Idea cards observed in this program's lineage.",
    )
    keywords: list[str] = Field(
        default_factory=list, description="Search keywords for retrieval ranking."
    )
    strategy: str = Field(
        default="", description="Mutation archetype of the exemplar program."
    )
    links: list[str] = Field(
        default_factory=list, description="Ids of semantically related cards."
    )
    gain_events: list[ContextualGain] | None = Field(
        default=None,
        description="Use-attributed base-relative injection events; reputation computes this card's efficacy block from them.",
    )


AnyCard = MemoryCard | ProgramCard


class LocalMemorySnapshot(BaseModel):
    """Persisted local memory state."""

    model_config = ConfigDict(extra="forbid")

    memory_cards: dict[str, MemoryCard] = Field(
        default_factory=dict, description="All bank cards keyed by card id."
    )


__all__ = [
    "AnyCard",
    "CardAlias",
    "CardStatsBlock",
    "ConnectedIdea",
    "DecisionMetrics",
    "LocalMemorySnapshot",
    "MemoryCard",
    "MemoryCardExplanation",
    "ProgramCard",
    "Strategy",
]
