from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class SearchPlan(BaseModel):
    """Search planning structure"""
    tools: list[str] = Field(default_factory=list, description="Tools to use for searching")
    keyword_collection: list[str] = Field(default_factory=list, description="Keywords to search for")
    vector_queries: list[str] = Field(default_factory=list, description="Semantic search queries across all vector fields")
    vector_description_queries: list[str] = Field(
        default_factory=list,
        description="Semantic queries for description field vector search",
    )
    vector_task_description_queries: list[str] = Field(
        default_factory=list,
        description="Semantic queries for task_description field vector search",
    )
    vector_explanation_summary_queries: list[str] = Field(
        default_factory=list,
        description="Semantic queries for explanation.summary field vector search",
    )
    page_index: list[int] = Field(default_factory=list, description="Specific page indices to retrieve")

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
    meta: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class Retriever(Protocol):
    """Unified interface for keyword / vector / page-id retrievers."""
    name: str
    def build(self, page_store) -> None: ...
    def search(self, query_list: list[str], top_k: int = 10) -> list[list[Hit]]: ...
