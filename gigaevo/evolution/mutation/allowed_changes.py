"""Genome-agnostic diff contract for StructuredDiffMutationOperator.

A subclass owns one genome family and defines which changes are representable;
the operator never inspects genome internals. Parents are keyed by prompt
namespace ("A", "B", ...) to genome code. Reference implementation:
gigaevo/chains/dag_changes.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gigaevo.evolution.mutation.constants import ArchetypeName
from gigaevo.llm.agents.mutation import MutationChange, MutationStructuredOutput


class DiffInsightCitation(BaseModel):
    """One cited entry from a parent's numbered '## Program Insights' list.

    Letter-parent counterpart of InsightCitation: diff-mode parents are
    labelled by namespace letter ('=== Parent A ==='), not 1-based number.
    """

    model_config = ConfigDict(extra="forbid")

    parent: str = Field(
        default="A",
        description=(
            "Parent letter (as labelled '=== Parent A ===') whose "
            "insight list this refers to. Always 'A' in single-parent mode."
        ),
    )
    insight: int = Field(
        description=(
            "The `N.` number of the insight in that parent's "
            "'## Program Insights' list."
        )
    )


def _mirrored(field: str) -> str:
    description = MutationStructuredOutput.model_fields[field].description
    assert description is not None
    return description


class DiffStructuredOutputBase(BaseModel):
    """Evidence fields every structured diff carries.

    Mirrors MutationStructuredOutput field-for-field except: the diff payload
    replaces `code`, and parents are cited by namespace letter instead of
    1-based number (base_parent, insight_ids_used). Shared descriptions are
    read from MutationStructuredOutput so the two schemas cannot drift.
    """

    model_config = ConfigDict(extra="forbid")

    archetype: ArchetypeName = Field(description=_mirrored("archetype"))
    justification: str = Field(description=_mirrored("justification"))
    insights_used: list[str] = Field(
        default_factory=list, description=_mirrored("insights_used")
    )
    insight_ids_used: list[DiffInsightCitation] = Field(
        default_factory=list,
        description=(
            "Program Insights entries acted on, each cited as the parent letter "
            "plus the `N.` number in that parent's numbered insight list. Each "
            "parent's list numbers from 1, so the parent letter disambiguates. "
            "Empty when the lists are empty or none were used. Never invent "
            "numbers."
        ),
    )
    base_parent: str = Field(
        description=(
            "Letter of the parent (as labelled '=== Parent A ===' above) whose "
            "overall structure this child keeps. If you blend several, pick the "
            "one it most resembles. Reward and context anchor to this parent."
        ),
    )
    card_ids_used: list[str] = Field(
        default_factory=list, description=_mirrored("card_ids_used")
    )
    changes: list[MutationChange] = Field(
        default_factory=list, description=_mirrored("changes")
    )


@dataclass(frozen=True)
class DiffSchema:
    json_schema: dict[str, Any]
    validate: Callable[[Any], Any]


class AllowedChanges(ABC):
    @abstractmethod
    def build_schema(self, parents: dict[str, str]) -> DiffSchema:
        """Per-call wire schema (guided decoding) + validator for these parents."""

    @abstractmethod
    def render_parents(self, parents: dict[str, str]) -> str:
        """Render parent genomes for the mutation prompt, with stable component ids."""

    @abstractmethod
    def apply(self, diff: Any, parents: dict[str, str]) -> str:
        """Transcribe a validated diff into child genome code; MutationError on failure."""

    @abstractmethod
    def describe(self) -> str:
        """Prose description of the diff language for the system prompt."""
