from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol
from pydantic import BaseModel, Field

class ToolResult(BaseModel):
    """Tool execution result"""
    tool: str = Field(..., description="Tool name")
    inputs: Dict[str, Any] = Field(..., description="Input parameters")
    outputs: Any = Field(..., description="Output results")
    error: Optional[str] = Field(None, description="Error message if any")

class Tool(Protocol):
    name: str
    def run(self, **kwargs) -> ToolResult: ...

class ToolRegistry(Protocol):
    def run_many(self, tool_inputs: Dict[str, Dict[str, Any]]) -> List[ToolResult]: ...
