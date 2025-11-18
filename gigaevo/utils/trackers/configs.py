from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class TBConfig(BaseModel):
    logdir: Path
    summary_writer_kwargs: dict[str, Any] = Field(default_factory=dict)


class WBConfig(BaseModel):
    project: str | None = None
    name: str | None = None
    entity: str | None = None
    notes: str | None = None
    tags: list[str] | None = None
    config: dict[str, Any] | None = None
    resume: bool = False
