"""Typed event rows: one scored idea-introduction observation per (idea, child)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.shared_memory.models import Quartile

_NAN = float("nan")


class EfficacyEvent(BaseModel):
    """One idea-introduction observation: a child program carrying ``idea_id``
    where none of its parents did, with every event-level metric the per-idea
    aggregation consumes. A metric that could not be computed stays NaN."""

    model_config = ConfigDict(frozen=True)

    idea_id: str = Field(description="Idea the child introduced.")
    quartile: Quartile = Field(
        description="Run-progress slice of the child's generation."
    )
    child_id: str = Field(
        description="Introducing child program id; the cohort dedup key."
    )
    best_parent_fit: float = Field(
        description="Fitness of the child's best parent (direction-normalized)."
    )
    IntroGain_best: float = Field(
        description="Child fitness minus best-parent fitness, direction-normalized (positive = improvement)."
    )
    IntroGain_best_rel: float = Field(
        default=_NAN,
        description="IntroGain_best relative to |best-parent fitness|.",
    )
    IntroGain_percentile_in_quartile: float = Field(
        default=_NAN,
        description="Percentile of IntroGain_best within the quartile's gains.",
    )
    IntroGain_percentile_overall: float = Field(
        default=_NAN,
        description="Percentile of IntroGain_best within all gains.",
    )
    IntroGain_z_in_quartile: float = Field(
        default=_NAN,
        description="Robust z-score of IntroGain_best within the quartile.",
    )
    IntroGain_z_overall: float = Field(
        default=_NAN,
        description="Robust z-score of IntroGain_best within all gains.",
    )
    SiblingWin: float = Field(
        default=_NAN,
        description="1.0 iff the child beats its same-generation siblings' median.",
    )
    SiblingPercentile: float = Field(
        default=_NAN,
        description="Child's fitness percentile among same-generation siblings.",
    )
    SiblingDelta: float = Field(
        default=_NAN,
        description="Child fitness minus same-generation siblings' median.",
    )
    SiblingWin_allgens: float = Field(
        default=_NAN,
        description="1.0 iff the child beats its all-generation siblings' median.",
    )
    SiblingPercentile_allgens: float = Field(
        default=_NAN,
        description="Child's fitness percentile among all-generation siblings.",
    )
    SiblingDelta_allgens: float = Field(
        default=_NAN,
        description="Child fitness minus all-generation siblings' median.",
    )
    ParentFitnessPercentile_within_gen: float = Field(
        default=_NAN,
        description="Best parent's fitness percentile within its generation.",
    )
    BornInElite: float = Field(
        default=_NAN,
        description="1.0 iff the child itself lands in the elite set.",
    )
    DescMaxLift_k_best: float = Field(
        default=_NAN,
        description="Best descendant fitness within k generations minus best-parent fitness.",
    )
    ReachesElite_k: float = Field(
        default=_NAN,
        description="1.0 iff any descendant within k generations reaches the elite set.",
    )
    TimeToElite_k: float = Field(
        default=_NAN,
        description="Generations until the first elite descendant within k.",
    )
    LineageReachesFinal: float = Field(
        default=_NAN,
        description="1.0 iff the child's lineage survives to the final generation.",
    )
    DescendantCount_k: float = Field(
        default=_NAN,
        description="Number of descendants within k generations.",
    )
    BranchingFactor: float = Field(
        default=_NAN,
        description="Number of direct children of the introducing program.",
    )
    TimeToPeak_k: float = Field(
        default=_NAN,
        description="Generations until the lineage's best fitness within k.",
    )
