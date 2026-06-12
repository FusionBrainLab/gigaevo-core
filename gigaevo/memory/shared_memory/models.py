from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer
from pydantic_core.core_schema import SerializerFunctionWrapHandler

Strategy = Literal["exploration", "exploitation", "hybrid"]


class Quartile(StrEnum):
    """Run-progress slice of the origin analysis.

    Declaration order is canonical: chronological quarters first, then the
    whole-run ``ALL`` aggregate. Rank tables and iteration orders must derive
    from this enum, never restate it.
    """

    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"
    ALL = "ALL"

    @classmethod
    def quarters(cls) -> tuple[Quartile, ...]:
        """The four chronological slices, excluding the ALL aggregate."""
        return tuple(q for q in cls if q is not cls.ALL)


class EfficacyMetrics(BaseModel):
    """Single source of the per-(idea, quartile) efficacy-metric vocabulary.

    Every metric the origin analysis produces is a declared, described field —
    nothing rides along as an undeclared extra. ``IdeaStats`` (the analysis row)
    and ``CardStatsBlock`` (the card-stamped block) both inherit this vocabulary,
    so producer and consumer cannot drift apart. Field names are the banks.json
    contract, including the analyzer-cased ``IntroGain_*`` keys.

    "Gain" is always the child-minus-parent best-fitness delta in
    positive-is-improvement space (the analysis negates for minimize metrics);
    "adjusted" means the parent-fitness-local counterfactual median has been
    subtracted.
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
    IntroGain_best_p10: float | None = Field(
        default=None, description="10th percentile of raw per-introduction gains."
    )
    IntroGain_best_median: float | None = Field(
        default=None, description="Median raw child-minus-parent best-fitness gain."
    )
    IntroGain_best_adj_median: float | None = Field(
        default=None,
        description="Median cohort-adjusted gain (raw gain minus the parent-local counterfactual).",
    )
    IntroGain_best_rel_median: float | None = Field(
        default=None,
        description="Median gain relative to the parent's best fitness magnitude.",
    )
    IntroGain_best_p90: float | None = Field(
        default=None, description="90th percentile of raw per-introduction gains."
    )
    DownsideRate_best: float | None = Field(
        default=None,
        description="Fraction of introductions whose adjusted gain fell below the harm threshold.",
    )
    TailRisk_best_median: float | None = Field(
        default=None,
        description="Median of min(gain, 0): the typical downside magnitude per introduction.",
    )
    IntroGain_percentile_median_in_quartile: float | None = Field(
        default=None,
        description="Median percentile rank of the gain among same-quartile introductions.",
    )
    IntroGain_percentile_median_overall: float | None = Field(
        default=None,
        description="Median percentile rank of the gain among all introductions.",
    )
    IntroGain_z_median_in_quartile: float | None = Field(
        default=None,
        description="Median z-score of the gain within its quartile cohort.",
    )
    IntroGain_z_median_overall: float | None = Field(
        default=None, description="Median z-score of the gain over the whole run."
    )
    SiblingWinRate: float | None = Field(
        default=None,
        description="Fraction of introductions whose child beat its same-generation siblings.",
    )
    SiblingPercentile_median: float | None = Field(
        default=None,
        description="Median fitness percentile of the child among same-generation siblings.",
    )
    SiblingDelta_median: float | None = Field(
        default=None,
        description="Median fitness delta of the child versus its sibling median.",
    )
    SiblingWinRate_allgens: float | None = Field(
        default=None,
        description="Sibling win rate with siblings pooled across all generations.",
    )
    SiblingPercentile_allgens_median: float | None = Field(
        default=None,
        description="Median sibling percentile pooled across all generations.",
    )
    SiblingDelta_allgens_median: float | None = Field(
        default=None,
        description="Median sibling delta pooled across all generations.",
    )
    DescMaxLift_k_best_median: float | None = Field(
        default=None,
        description="Median best-fitness lift achieved by descendants within k generations.",
    )
    ReachesElite_k_rate: float | None = Field(
        default=None,
        description="Fraction of introductions whose lineage reaches the elite archive within k generations.",
    )
    TimeToElite_k_median: float | None = Field(
        default=None,
        description="Median generations until a descendant enters the elite archive.",
    )
    LineageReachesFinal_rate: float | None = Field(
        default=None,
        description="Fraction of introductions whose lineage survives to the final generation.",
    )
    DescendantCount_k_median: float | None = Field(
        default=None, description="Median descendant count within k generations."
    )
    BranchingFactor_median: float | None = Field(
        default=None, description="Median direct-children count per lineage program."
    )
    TimeToPeak_k_median: float | None = Field(
        default=None,
        description="Median generations until the lineage's peak fitness within k generations.",
    )
    ParentFitnessPercentile_within_gen_median: float | None = Field(
        default=None,
        description="Median fitness percentile of the parent within its generation (selection-pressure diagnostic).",
    )
    BornInElite_rate: float | None = Field(
        default=None,
        description="Fraction of introductions whose child was born directly into the elite archive.",
    )
    origin_programs: int = Field(
        default=0, description="Programs in which the idea first appeared."
    )
    origin_in_elite_rate: float | None = Field(
        default=None,
        description="Fraction of origin programs that sit in the elite archive.",
    )
    origin_generation_span: float | None = Field(
        default=None,
        description="Generations between the first and last origin program.",
    )
    origin_root_diversity: float | None = Field(
        default=None, description="Distinct lineage roots among origin programs."
    )
    reinvention_rate_origins_per_distinct_gen: float | None = Field(
        default=None,
        description="Origin programs per distinct generation (independent-rediscovery pressure).",
    )


class CardStatsBlock(EfficacyMetrics):
    """One efficacy-statistics block of a card (the ``ALL`` or a per-quartile entry).

    The metric vocabulary is inherited from :class:`EfficacyMetrics`; ``k_harm``
    is added by the injection-posterior producer.
    """

    model_config = ConfigDict(extra="forbid")

    k_harm: int | None = Field(
        default=None,
        description="Introduction events whose gain fell below the harm threshold.",
    )

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


class EvolutionStatistics(BaseModel):
    """Typed ``evolution_statistics`` payload of a card.

    Two writers populate it: the ideas-tracker origin analysis (per-quartile
    ``Q1``..``Q4`` plus ``ALL`` blocks) and the injection-posterior stamp
    (``ALL`` only). ``best_ideas_snapshot`` is the metric block of the
    best-ideas summary row the offline write pipeline merges in. Absent blocks
    stay absent on serialization so banks.json keeps its shape.
    """

    model_config = ConfigDict(extra="forbid")

    ALL: CardStatsBlock | None = Field(
        default=None,
        description="Whole-run aggregate block — the only block decision paths read.",
    )
    Q1: CardStatsBlock | None = Field(
        default=None, description="First run-quarter audit block (write-only)."
    )
    Q2: CardStatsBlock | None = Field(
        default=None, description="Second run-quarter audit block (write-only)."
    )
    Q3: CardStatsBlock | None = Field(
        default=None, description="Third run-quarter audit block (write-only)."
    )
    Q4: CardStatsBlock | None = Field(
        default=None, description="Fourth run-quarter audit block (write-only)."
    )
    best_ideas_snapshot: CardStatsBlock | None = Field(
        default=None,
        description="Metric block of the best-ideas summary row merged in by the offline write pipeline.",
    )

    @classmethod
    def from_blocks(cls, blocks: dict[Quartile, CardStatsBlock]) -> EvolutionStatistics:
        """Build statistics from per-quartile blocks without string keys."""
        stats = cls()
        for quartile, block in blocks.items():
            stats = stats.with_block(quartile, block)
        return stats

    def with_block(
        self, quartile: Quartile, block: CardStatsBlock
    ) -> EvolutionStatistics:
        """Copy with one quartile's block replaced; the enum member maps onto
        the field of the same name."""
        return self.model_copy(update={quartile.name: block})

    @model_serializer(mode="wrap")
    def serialize_without_absent_blocks(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        return {key: value for key, value in handler(self).items() if value is not None}


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
    evolution_statistics: EvolutionStatistics = Field(
        default_factory=EvolutionStatistics,
        description="Per-quartile and whole-run efficacy statistics.",
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
    evolution_statistics: EvolutionStatistics = Field(
        default_factory=EvolutionStatistics,
        description="Injection-efficacy statistics stamped onto this card.",
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
    "EfficacyMetrics",
    "EvolutionStatistics",
    "LocalMemorySnapshot",
    "MemoryCard",
    "MemoryCardExplanation",
    "ProgramCard",
    "Quartile",
    "Strategy",
]
