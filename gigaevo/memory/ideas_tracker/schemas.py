"""Structured-output response schemas for ideas-tracker LLM calls."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _strip_brackets(value: str) -> str:
    return value.strip().strip("[]").strip()


class PresentIdeaRef(BaseModel):
    """A new idea that matches an existing bank entry."""

    model_config = ConfigDict(extra="forbid")

    idea_id: str = Field(
        description="Short id of the matching bank idea, exactly as shown in the BANK section."
    )
    sequence: int = Field(
        description="1-based sequence number of the new idea being classified."
    )

    @field_validator("idea_id")
    @classmethod
    def _normalize_idea_id(cls, value: str) -> str:
        return _strip_brackets(value)


class UpdatedIdeaRef(BaseModel):
    """A new idea that matches a bank entry but states it more precisely."""

    model_config = ConfigDict(extra="forbid")

    idea_id: str = Field(
        description="Short id of the bank idea whose description should be upgraded."
    )
    sequence: int = Field(
        description="1-based sequence number of the new idea being classified."
    )
    text: str = Field(
        description="The richer description that should replace the bank idea's current text."
    )

    @field_validator("idea_id")
    @classmethod
    def _normalize_idea_id(cls, value: str) -> str:
        return _strip_brackets(value)


class ClassifyExtResponse(BaseModel):
    """Classification of newly extracted ideas against the existing idea bank."""

    model_config = ConfigDict(extra="forbid")

    new_ideas: list[str] = Field(
        description="Descriptions of ideas with no match in the bank, verbatim from the input."
    )
    present_ideas: list[PresentIdeaRef] = Field(
        description="Ideas already covered by a bank entry at equal or lesser detail."
    )
    updated_ideas: list[UpdatedIdeaRef] = Field(
        description="Ideas matching a bank entry but with strictly more identifying detail."
    )


class ClusterPartitionResponse(BaseModel):
    """Partition of a candidate cluster into a coherent core and rejected items."""

    model_config = ConfigDict(extra="forbid")

    included: list[int] = Field(
        description="1-based input line indices forming the single largest coherent group."
    )
    rejected: list[int] = Field(
        description="1-based input line indices excluded from the group; they re-cluster later."
    )


class RepresentativeChoiceResponse(BaseModel):
    """Choice of the canonical representative line for a cluster."""

    model_config = ConfigDict(extra="forbid")

    representative_index: int = Field(
        description="1-based index of the most technically specific line; in 1..n."
    )


class SynthesisedDescription(BaseModel):
    """Canonical description synthesised for a cluster."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(
        description=(
            "The synthesized description, at most 4 sentences, with no preamble or "
            "labels. The first sentence states the concrete code change itself."
        )
    )


class KeywordsResponse(BaseModel):
    """Keywords extracted from a single idea description."""

    model_config = ConfigDict(extra="forbid")

    keywords: list[str] = Field(
        description=(
            "3-7 lowercase keyword phrases of 2-4 words each: 1-3 category tags "
            "plus 2-5 mechanism tags, no duplicates, no broader/narrower overlap. "
            "Empty when the idea lacks meaningful content."
        )
    )


class SummaryResponse(BaseModel):
    """Free-text summary distilled from a longer input."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(
        description=(
            "The distilled summary, 1-2 sentences maximum, factually consistent "
            "with the input. Empty when the input has no meaningful content."
        )
    )
