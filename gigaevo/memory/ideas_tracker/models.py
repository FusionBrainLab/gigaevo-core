"""
Data models for the IdeasTracker module.

All cross-module data-transfer types are Pydantic BaseModel, providing
validation and serialisation via model_dump(). IdeaCluster is a plain
class — it is a mutable working object internal to ClusteringAnalyzer.
"""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Improvement normalisation  (mutation output → canonical dict format)
# ---------------------------------------------------------------------------

_DESCRIPTION_KEYS = (
    "description", "summary", "title", "change", "what_changed",
    "pattern", "improvement", "name",
)
_EXPLANATION_KEYS = (
    "explanation", "rationale", "reason", "why", "motivation",
    "expected_effect", "impact", "details", "justification",
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


def normalize_improvement_item(idea: Any) -> dict[str, str]:
    """Coerce a mutation change payload into {"description": ..., "explanation": ...}."""
    if isinstance(idea, str):
        stripped = idea.strip()
        return {"description": stripped or "Unspecified change", "explanation": ""}
    if not isinstance(idea, dict):
        return {"description": _stringify(idea) or "Unspecified change", "explanation": ""}

    description = next(
        (_stringify(idea[k]) for k in _DESCRIPTION_KEYS if k in idea and _stringify(idea[k])),
        "",
    )
    explanation = next(
        (_stringify(idea[k]) for k in _EXPLANATION_KEYS if k in idea and _stringify(idea[k])),
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
    return {"description": description, "explanation": explanation}


def normalize_improvements(ideas: Any) -> list[dict[str, str]]:
    """Normalise any mutation changes payload to a list of {description, explanation} dicts."""
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

    entries: list[str] = Field(default_factory=list)
    summary: str = ""


class Idea(BaseModel):
    """
    A tracked improvement idea extracted from evolutionary programs.

    Produced by an Analyzer and stored in IdeaBank. Enriched with keywords
    and an explanation summary after initial classification.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str
    category: str = ""
    strategy: str = ""
    task_description: str = ""
    task_description_summary: str = ""
    last_generation: int = 0
    programs: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    explanation: IdeaExplanation = Field(default_factory=IdeaExplanation)
    usage: dict[str, Any] = Field(default_factory=dict)
    aliases: list[dict[str, Any]] = Field(default_factory=list)


class ProgramRecord(BaseModel):
    """
    Metadata extracted from a Program for idea analysis.

    Created from a raw Program object; carries only the fields that
    analysers need (no stage results, no raw execution data).
    """

    id: str
    fitness: float
    generation: int
    parents: list[str] = Field(default_factory=list)
    insights: list[str] = Field(default_factory=list)
    improvements: list[dict[str, str]] = Field(default_factory=list)
    strategy: str = ""
    task_description: str = ""
    task_description_summary: str = ""
    code: str = ""


class IdeaUpdate(BaseModel):
    """Instruction to update an existing Idea already present in IdeaBank."""

    idea_id: str
    programs: list[str] = Field(default_factory=list)
    generation: int = 0
    new_description: str | None = None
    motivation: str | None = None


class AnalysisResult(BaseModel):
    """
    Output of Analyzer.analyze().

    new_ideas: ideas to add to the bank.
    updates: modifications to apply to ideas already in the bank.
    """

    new_ideas: list[Idea] = Field(default_factory=list)
    updates: list[IdeaUpdate] = Field(default_factory=list)


class EmbeddedIdea(BaseModel):
    """
    An improvement card with its sentence-embedding vector.

    Used internally by ClusteringAnalyzer during the embed → cluster → refine pipeline.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str
    source_program_id: str = ""
    cluster_id: str = ""
    change_motivation: str = ""
    embedding: list[float] = Field(default_factory=list)


class ClassificationChunk(BaseModel):
    """A chunk of IdeaBank ideas prepared for one LLM classification call."""

    text: str
    short_ids: list[dict[str, str]]


# ---------------------------------------------------------------------------
# Program → ProgramRecord conversion
# ---------------------------------------------------------------------------

def program_to_record(
    program: Any,
    task_description: str,
    task_description_summary: str,
    fitness_key: str = "fitness",
) -> ProgramRecord:
    """Convert a Program to a ProgramRecord for analyser consumption."""
    mutation_output = program.metadata.get("mutation_output", {})
    if not isinstance(mutation_output, dict):
        mutation_output = {}
    return ProgramRecord(
        id=program.id,
        fitness=program.metrics.get(fitness_key, 0.0),
        generation=program.lineage.generation,
        parents=list(program.lineage.parents),
        insights=mutation_output.get("insights_used") or [],
        improvements=normalize_improvements(mutation_output.get("changes")),
        strategy=mutation_output.get("archetype") or "",
        task_description=task_description,
        task_description_summary=task_description_summary,
        code=program.code,
    )


def programs_to_records(
    programs: list[Any],
    task_description: str,
    task_description_summary: str,
    fitness_key: str = "fitness",
) -> tuple[list[ProgramRecord], set[str]]:
    """Convert a list of Programs to (list[ProgramRecord], set of their ids)."""
    records = [
        program_to_record(p, task_description, task_description_summary, fitness_key)
        for p in programs
    ]
    return records, {p.id for p in programs}
