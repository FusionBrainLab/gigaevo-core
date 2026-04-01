from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Tool execution result"""

    tool: str = Field(..., description="Tool name")
    inputs: dict[str, Any] = Field(..., description="Input parameters")
    outputs: Any = Field(..., description="Output results")
    error: str | None = Field(None, description="Error message if any")


class Tool(Protocol):
    name: str

    def run(self, **kwargs) -> ToolResult: ...


class ToolRegistry(Protocol):
    def run_many(self, tool_inputs: dict[str, dict[str, Any]]) -> list[ToolResult]: ...
