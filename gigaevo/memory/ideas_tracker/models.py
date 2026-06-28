"""
Data models for the IdeasTracker module.

All cross-module data-transfer types are Pydantic BaseModel, providing
validation and serialisation via model_dump().
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gigaevo.evolution.mutation.constants import MUTATION_OUTPUT_METADATA_KEY

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


class ProgramRecord(BaseModel):
    """Metadata extracted from a Program for the librarian write path.

    Created from a raw Program object; carries only the fields the librarian
    needs (no stage results, no raw execution data).
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
    base_parent_id: str = Field(
        default="",
        description="Id of the base parent the mutator anchored the child to.",
    )
    parent_code: str = Field(
        default="",
        description="Source code of the base parent, when available.",
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
    base_parent: int = Field(
        default=1,
        description="1-based index of the parent the mutator anchored the child to.",
    )

    @field_validator("archetype", mode="before")
    @classmethod
    def coerce_none_archetype(cls, value: Any) -> Any:
        return value or ""

    @field_validator("base_parent", mode="before")
    @classmethod
    def coerce_none_base_parent(cls, value: Any) -> Any:
        return value or 1


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
    """Convert a Program to a ProgramRecord for the librarian write path.

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
    # The librarian diffs the child against the parent the mutator anchored it to
    # (1-based ``base_parent``), not whichever parent the selector happened to list
    # first; in ≥2-parent rewrite mode the base may be a later donor's sibling. Out
    # of range falls back to the first parent, matching freeze_base_parent_snapshot.
    base_index = mutation_output.base_parent - 1
    if base_index < 0 or base_index >= len(parents):
        base_index = 0
    base_parent_id = parents[base_index] if parents else ""
    parent_code = parent_codes.get(base_parent_id, "") if parent_codes else ""
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
        base_parent_id=base_parent_id,
        parent_code=parent_code,
    )
