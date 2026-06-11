"""Shared pydantic models for the origin analysis pipeline."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from gigaevo.memory.core.idea_stats import IdeaStats


class IntroEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    idea_id: str
    child_id: str
    child_gen: int
    child_fit: float
    parents: list[str]
    best_parent_id: str
    best_parent_fit: float
    mean_parent_fit: float
    quartile: str


class DescMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    desc_max_fit_k: float
    time_to_peak_k: float
    desc_count_k: int
    reaches_elite_k: float
    time_to_elite_k: float
    lineage_reaches_final: float
    branching_factor: int


class AnalysisResult(BaseModel):
    """``summary`` holds every (idea, quartile) row; ``best_ideas`` is the
    admitter-selected subset."""

    summary: list[IdeaStats]
    best_ideas: list[IdeaStats]
