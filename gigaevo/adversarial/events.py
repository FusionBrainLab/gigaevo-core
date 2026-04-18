"""Adversarial-specific canonical events.

Subclasses of `BaseEvent` that describe adversarial-coevolution emission sites:
- TRACKER_WRITE — DGImprovementTracker finishes a batch write
- HOF_FETCH     — opponent HoF loaded from Redis for sampling
- HOF_ROTATE    — opponent HoF size/content changed between DAG steps
- CELL_PICK     — CellStratified opponent provider picked one elite per cell

Role-invariant validators (constructor G vs improver D expectations) attach
directly to the relevant events via Pydantic `@field_validator` /
`@model_validator` decorators. The general log auditor does NOT hold these
invariants — events validate themselves on construction.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field, model_validator

from gigaevo.monitoring.events import BaseEvent


class TrackerWrite(BaseEvent):
    event: ClassVar[str] = "TRACKER_WRITE"
    description: ClassVar[str] = (
        "DGImprovementTracker finished a pair-batch write to Redis."
    )
    health_question: ClassVar[str] = "Is the adversarial tracker updating?"
    expected_after_gen: ClassVar[int] = 1

    pairs_count: int = Field(ge=0)
    positive_count: int = Field(ge=0)
    d_wins_added: int = Field(ge=0)
    g_resisted_added: int = Field(ge=0)
    d_faced_added: int = Field(ge=0)
    gen: int | None = None

    @model_validator(mode="after")
    def _positive_le_pairs(self) -> TrackerWrite:
        if self.positive_count > self.pairs_count:
            raise ValueError(
                f"positive_count ({self.positive_count}) > pairs_count "
                f"({self.pairs_count}) — every positive is also a pair"
            )
        return self


class HofFetch(BaseEvent):
    event: ClassVar[str] = "HOF_FETCH"
    description: ClassVar[str] = "Opponent HoF loaded from Redis for sampling."
    health_question: ClassVar[str] = "Are we reading the opponent archive?"
    expected_after_gen: ClassVar[int] = 1

    label: str
    n_elites: int = Field(ge=0)
    fitness_key: str
    gen: int | None = None


class HofRotate(BaseEvent):
    event: ClassVar[str] = "HOF_ROTATE"
    description: ClassVar[str] = "Opponent HoF changed (new elites added or removed)."
    health_question: ClassVar[str] = "Is the archive actually rotating?"
    expected_after_gen: ClassVar[int] = 2

    label: str
    old_hof_size: int = Field(ge=0)
    new_hof_size: int = Field(ge=0)
    gen: int | None = None


class CellPick(BaseEvent):
    event: ClassVar[str] = "CELL_PICK"
    description: ClassVar[str] = (
        "CellStratified opponent provider picked one elite from a cell."
    )
    health_question: ClassVar[str] = "Is distinct-cell opponent selection working?"
    expected_after_gen: ClassVar[int] = 1

    label: str
    cell_id: str
    program_id: str
    fitness_key: str
    fitness_value: float
    gen: int | None = None
