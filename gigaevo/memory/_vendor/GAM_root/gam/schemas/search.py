from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class SearchPlan(BaseModel):
    """Search planning structure"""

    tools: list[str] = Field(
        default_factory=list,
        description="Retrieval tools to run for this plan, chosen from the available tool names",
    )
    keyword_collection: list[str] = Field(
        default_factory=list,
        description=(
            "Short high-signal exact-match keywords, max 5. "
            "Empty when the keyword tool is not selected"
        ),
    )
    vector_queries: list[str] = Field(
        default_factory=list,
        description=(
            "Semantic search queries across all vector fields combined, max 2. "
            "Empty when the vector tool is not selected"
        ),
    )
    vector_description_queries: list[str] = Field(
        default_factory=list,
        description=(
            "Semantic queries over only the description field, max 2. "
            "Empty when the vector_description tool is not selected"
        ),
    )
    vector_task_description_queries: list[str] = Field(
        default_factory=list,
        description=(
            "Semantic queries over only the task_description field, max 2. "
            "Empty when the vector_task_description tool is not selected"
        ),
    )
    vector_explanation_summary_queries: list[str] = Field(
        default_factory=list,
        description=(
            "Semantic queries over only the explanation.summary field, max 2. "
            "Empty when the vector_explanation_summary tool is not selected"
        ),
    )
    page_index: list[int] = Field(
        default_factory=list,
        description=(
            "Known integer page indices to re-read in full, max 5. "
            "Never guessed; empty when no concrete indices are known"
        ),
    )

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]:
        schema = super().model_json_schema()
        props = list(schema.get("properties", {}).keys())
        schema["required"] = props
        schema["additionalProperties"] = False
        return schema


class Hit(BaseModel):
    """Search result hit"""

    page_id: str | None = Field(None, description="Page ID in store")
    snippet: str = Field(..., description="Text snippet from the source")
    source: str = Field(..., description="Source type (keyword/vector/page_index/tool)")
    meta: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class Retriever(Protocol):
    """Unified interface for keyword / vector / page-id retrievers."""

    name: str

    def build(self, page_store) -> None: ...
    def search(self, query_list: list[str], top_k: int = 10) -> list[list[Hit]]: ...
