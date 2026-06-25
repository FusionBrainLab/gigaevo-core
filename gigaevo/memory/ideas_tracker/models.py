"""
Data models for the IdeasTracker module.

All cross-module data-transfer types are Pydantic BaseModel, providing
validation and serialisation via model_dump(). IdeaCluster is a plain
class — it is a mutable working object internal to ClusteringAnalyzer.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gigaevo.evolution.mutation.constants import MUTATION_OUTPUT_METADATA_KEY
from gigaevo.memory.shared_memory.models import CardAlias

# ---------------------------------------------------------------------------
# Improvement normalisation  (mutation output → typed Improvement)
# ---------------------------------------------------------------------------

_DESCRIPTION_KEYS = (
    "description",
    "summary",
    "title",
    "change",
    "what_changed",
    "pattern",
    "improvement",
    "name",
)
_EXPLANATION_KEYS = (
    "explanation",
    "rationale",
    "reason",
    "why",
    "motivation",
    "expected_effect",
    "impact",
    "details",
    "justification",
)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        parts = [f"{k}: {_stringify(v)}" for k, v in value.items() if _stringify(v)]
        return "; ".join(parts)
    if isinstance(value, (list, tuple, set)):
        return "; ".join(p for p in (_stringify(i) for i in value) if p)
    return str(value).strip()


class Improvement(BaseModel):
    """A single normalised mutation change: what changed, and the stated why."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(description="What changed, in one sentence.")
    explanation: str = Field(
        default="", description="The stated motivation for the change."
    )


def normalize_improvement_item(idea: Any) -> Improvement:
    """Coerce one mutation change payload (str, dict, or anything) into an Improvement."""
    if isinstance(idea, str):
        stripped = idea.strip()
        return Improvement(description=stripped or "Unspecified change")
    if not isinstance(idea, dict):
        return Improvement(description=_stringify(idea) or "Unspecified change")

    description = next(
        (
            _stringify(idea[k])
            for k in _DESCRIPTION_KEYS
            if k in idea and _stringify(idea[k])
        ),
        "",
    )
    explanation = next(
        (
            _stringify(idea[k])
            for k in _EXPLANATION_KEYS
            if k in idea and _stringify(idea[k])
        ),
        "",
    )
    extras = [
        f"{k}: {_stringify(v)}"
        for k, v in idea.items()
        if k not in _DESCRIPTION_KEYS and k not in _EXPLANATION_KEYS and _stringify(v)
    ]
    if not description and extras:
        description, extras = extras[0], extras[1:]
    if not explanation and extras:
        explanation = "; ".join(extras)
    if not description:
        description = explanation or "Unspecified change"
    return Improvement(description=description, explanation=explanation)


def normalize_improvements(ideas: Any) -> list[Improvement]:
    """Normalise any mutation changes payload to a list of Improvement models."""
    if ideas is None:
        return []
    if isinstance(ideas, list):
        return [normalize_improvement_item(i) for i in ideas]
    return [normalize_improvement_item(ideas)]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class IdeaExplanation(BaseModel):
    """Accumulated motivations and synthesised usage summary for an Idea."""

    model_config = ConfigDict(extra="forbid")

    entries: list[str] = Field(
        default_factory=list,
        description="Accumulated motivation entries, oldest first.",
    )
    summary: str = Field(
        default="", description="LLM-condensed summary of the entries."
    )


class Idea(BaseModel):
    """
    A tracked improvement idea extracted from evolutionary programs.

    Produced by an Analyzer and stored in IdeaBank. Enriched with keywords
    and an explanation summary after initial classification.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Stable bank id of the idea.",
    )
    description: str = Field(
        description="The idea itself — the text injected into prompts."
    )
    category: str = Field(
        default="",
        description="Free-form topical category assigned by the producing analyzer.",
    )
    strategy: str = Field(
        default="", description="Mutation archetype the idea originated from."
    )
    task_description: str = Field(
        default="",
        description="Task description active when the idea was extracted.",
    )
    task_description_summary: str = Field(
        default="", description="Condensed form of the task description."
    )
    last_generation: int = Field(
        default=0, description="Latest generation at which the idea was observed."
    )
    programs: list[str] = Field(
        default_factory=list,
        description="Ids of programs that exhibited the idea.",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Search keywords assigned during enrichment.",
    )
    explanation: IdeaExplanation = Field(
        default_factory=IdeaExplanation,
        description="Accumulated motivations and synthesised usage summary.",
    )
    aliases: list[CardAlias] = Field(
        default_factory=list,
        description="Alternative phrasings merged into this idea.",
    )


class ProgramRecord(BaseModel):
    """
    Metadata extracted from a Program for idea analysis.

    Created from a raw Program object; carries only the fields that
    analysers need (no stage results, no raw execution data).
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Program id.")
    fitness: float = Field(
        description="Fitness value under the configured fitness key."
    )
    generation: int = Field(description="Generation the program was created in.")
    parents: list[str] = Field(default_factory=list, description="Parent program ids.")
    improvements: list[Improvement] = Field(
        default_factory=list,
        description="Normalised mutation changes that produced this program.",
    )
    strategy: str = Field(
        default="", description="Mutation archetype reported in the mutation output."
    )
    task_description: str = Field(
        default="", description="Task description active for this run."
    )
    task_description_summary: str = Field(
        default="", description="Condensed form of the task description."
    )
    code: str = Field(default="", description="Program source code.")
    parent_code: str = Field(
        default="",
        description="Source code of the first parent, when available.",
    )


class IdeaUpdate(BaseModel):
    """Instruction to update an existing Idea already present in IdeaBank."""

    model_config = ConfigDict(extra="forbid")

    idea_id: str = Field(description="Id of the bank idea to update.")
    programs: list[str] = Field(
        default_factory=list, description="Program ids to append to the idea."
    )
    generation: int = Field(
        default=0, description="Generation at which the update was observed."
    )
    new_description: str | None = Field(
        default=None,
        description="Replacement description; None keeps the current one.",
    )
    motivation: str | None = Field(
        default=None,
        description="New motivation entry to append to the idea's explanation.",
    )


class AnalysisResult(BaseModel):
    """
    Output of Analyzer.analyze().

    new_ideas: ideas to add to the bank.
    updates: modifications to apply to ideas already in the bank.
    """

    model_config = ConfigDict(extra="forbid")

    new_ideas: list[Idea] = Field(
        default_factory=list, description="Ideas to add to the bank."
    )
    updates: list[IdeaUpdate] = Field(
        default_factory=list,
        description="Modifications to apply to ideas already in the bank.",
    )
    failed_program_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Programs whose classification failed entirely; the tracker "
            "un-marks them as seen so a later sweep retries them."
        ),
    )


class EmbeddedIdea(BaseModel):
    """
    An improvement card with its sentence-embedding vector.

    Used internally by ClusteringAnalyzer during the embed → cluster → refine pipeline.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Ephemeral id of the improvement within the clustering pipeline.",
    )
    description: str = Field(description="Improvement text being clustered.")
    source_program_id: str = Field(
        default="", description="Id of the program the improvement came from."
    )
    cluster_id: str = Field(
        default="", description="Id of the cluster the improvement was assigned to."
    )
    change_motivation: str = Field(
        default="", description="Stated motivation accompanying the improvement."
    )
    embedding: list[float] = Field(
        default_factory=list,
        description="Sentence-embedding vector of the description.",
    )


class ShortIdEntry(BaseModel):
    """Maps an idea's short display id (UUID prefix) to its full id for one chunk."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Full idea id.")
    short_id: str = Field(
        description="UUID-prefix display id used inside the chunk text."
    )
    description: str = Field(description="Idea description shown to the LLM.")


class ClassificationChunk(BaseModel):
    """A chunk of IdeaBank ideas prepared for one LLM classification call."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="Rendered chunk text sent to the LLM.")
    short_ids: list[ShortIdEntry] = Field(
        description="Short-id to full-id mapping for the ideas in this chunk."
    )


class MutationOutput(BaseModel):
    """Validated `mutation_output` metadata blob attached to a mutated Program."""

    model_config = ConfigDict(extra="ignore")

    changes: Any = Field(
        default=None,
        description="Raw changes payload as emitted by the mutator; normalised via normalize_improvements().",
    )
    archetype: str = Field(
        default="", description="Mutation archetype label; empty when absent."
    )

    @field_validator("archetype", mode="before")
    @classmethod
    def coerce_none_archetype(cls, value: Any) -> Any:
        return value or ""


# ---------------------------------------------------------------------------
# Program → ProgramRecord conversion
# ---------------------------------------------------------------------------


def program_to_record(
    program: Any,
    task_description: str,
    task_description_summary: str,
    fitness_key: str = "fitness",
    parent_codes: dict[str, str] | None = None,
) -> ProgramRecord:
    """Convert a Program to a ProgramRecord for analyser consumption.

    The program must carry a metric under ``fitness_key`` (callers filter
    eligibility first); a missing key raises rather than minting a default.
    """
    raw_output = program.metadata.get(MUTATION_OUTPUT_METADATA_KEY)
    mutation_output = (
        MutationOutput.model_validate(raw_output)
        if isinstance(raw_output, dict)
        else MutationOutput()
    )
    parents = list(program.lineage.parents)
    parent_code = ""
    if parent_codes and parents:
        parent_code = parent_codes.get(parents[0], "")
    return ProgramRecord(
        id=program.id,
        fitness=program.metrics[fitness_key],
        generation=program.lineage.generation,
        parents=parents,
        improvements=normalize_improvements(mutation_output.changes),
        strategy=mutation_output.archetype,
        task_description=task_description,
        task_description_summary=task_description_summary,
        code=program.code,
        parent_code=parent_code,
    )


def programs_to_records(
    programs: list[Any],
    task_description: str,
    task_description_summary: str,
    fitness_key: str = "fitness",
    parent_codes: dict[str, str] | None = None,
) -> tuple[list[ProgramRecord], set[str]]:
    """Convert a list of Programs to (list[ProgramRecord], set of their ids).

    Every program must carry a metric under ``fitness_key``; filter
    eligibility before calling (a missing key raises).
    """
    records = [
        program_to_record(
            p, task_description, task_description_summary, fitness_key, parent_codes
        )
        for p in programs
    ]
    return records, {p.id for p in programs}
