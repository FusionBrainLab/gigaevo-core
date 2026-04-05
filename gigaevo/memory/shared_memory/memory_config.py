"""Pydantic configuration models for the memory system.

Replaces the 18 scattered kwargs of AmemGamMemory.__init__ with validated,
grouped configuration objects. Follows the EngineConfig pattern used
throughout the codebase.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.shared_memory.card_update_dedup import CardUpdateDedupConfig


class GamConfig(BaseModel):
    """GAM (Generative Agentic Memory) retriever settings."""

    model_config = ConfigDict(extra="forbid")

    enable_bm25: bool = False
    allowed_tools: list[str] = Field(default_factory=list)
    top_k_by_tool: dict[str, int] = Field(default_factory=dict)
    pipeline_mode: str = "default"


class ApiConfig(BaseModel):
    """Memory API connection settings."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = "http://localhost:8000"
    namespace: str = "default"
    author: str | None = None
    channel: str = "latest"
    sync_batch_size: int = Field(default=100, gt=0)
    sync_on_init: bool = True


class MemoryConfig(BaseModel):
    """All configuration for AmemGamMemory.

    Replaces 18 scattered constructor kwargs with a single validated config.
    Use ``api=None`` for local-only mode (replaces ``use_api=False``).
    """

    model_config = ConfigDict(extra="forbid")

    checkpoint_path: Path
    search_limit: int = Field(default=5, gt=0)
    rebuild_interval: int = Field(default=10, gt=0)
    enable_llm_synthesis: bool = True
    enable_memory_evolution: bool = True
    enable_llm_card_enrichment: bool = True
    api: ApiConfig | None = None
    gam: GamConfig = Field(default_factory=GamConfig)
    dedup: CardUpdateDedupConfig = Field(default_factory=CardUpdateDedupConfig)
