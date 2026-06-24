from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Result(BaseModel):
    """Search and integration result"""

    content: str = Field(
        "",
        description=(
            "Integrated relevance summary for the request. "
            "Empty string when no useful information exists"
        ),
    )
    sources: list[str | None] = Field(
        default_factory=list,
        description="Page IDs of the snippets that supported the included facts",
    )

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]:
        schema = super().model_json_schema()
        props = list(schema.get("properties", {}).keys())  # ["content", "sources"]
        schema["required"] = props
        schema["additionalProperties"] = False
        return schema


class ResearchOutput(BaseModel):
    """Research output"""

    integrated_memory: str = Field(..., description="Integrated memory content")
    raw_memory: dict[str, Any] = Field(..., description="Raw memory data")


class TopIdea(BaseModel):
    """Final selected idea for the reflection/selection pipeline."""

    card_id: str = Field(..., description="Selected card/page id")

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]:
        schema = super().model_json_schema()
        props = list(schema.get("properties", {}).keys())
        schema["required"] = props
        schema["additionalProperties"] = False
        return schema


class Decision(BaseModel):
    """Decision output for the reflection/selection pipeline."""

    mode: Literal["final", "continue"] = Field(
        ...,
        description=(
            "final = evidence is sufficient to select the top ideas; "
            "continue = more retrieval is needed"
        ),
    )
    top_ideas: list[TopIdea] = Field(
        default_factory=list,
        description=(
            "Selected ideas when mode=final, never padded with weak ideas. "
            "Empty when mode=continue"
        ),
    )
    additional_queries: list[str] = Field(
        default_factory=list,
        description=(
            "1-5 concrete follow-up retrieval queries when mode=continue. "
            "Empty when mode=final"
        ),
    )

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]:
        schema = super().model_json_schema()
        schema["required"] = ["mode", "top_ideas", "additional_queries"]
        schema["additionalProperties"] = False
        return schema
