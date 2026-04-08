"""Shared dataclasses for the origin analysis pipeline."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class IntroEvent:
    idea_id: str
    child_id: str
    child_gen: int
    child_fit: float
    parents: list[str]
    best_parent_id: str
    best_parent_fit: float
    mean_parent_fit: float
    quartile: str


@dataclass
class DescMetrics:
    desc_max_fit_k: float
    time_to_peak_k: float
    desc_count_k: int
    reaches_elite_k: float
    time_to_elite_k: float
    lineage_reaches_final: float
    branching_factor: int


@dataclass
class AnalysisResult:
    summary_df: pd.DataFrame
    best_ideas_df: pd.DataFrame
