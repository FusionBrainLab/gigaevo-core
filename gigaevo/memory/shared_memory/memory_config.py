"""Pydantic configuration models for the memory system.

Replaces the 18 scattered kwargs of AmemGamMemory.__init__ with validated,
grouped configuration objects. Follows the EngineConfig pattern used
throughout the codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    sync_batch_size: int = Field(default=100, ge=10)
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

    @classmethod
    def from_legacy_kwargs(
        cls,
        *,
        checkpoint_path: str,
        base_url: str = "http://localhost:8000",
        use_api: bool = True,
        namespace: str = "default",
        author: str | None = None,
        channel: str = "latest",
        search_limit: int = 5,
        enable_llm_synthesis: bool = True,
        enable_memory_evolution: bool = True,
        enable_llm_card_enrichment: bool = True,
        rebuild_interval: int = 10,
        enable_bm25: bool = False,
        sync_batch_size: int = 100,
        sync_on_init: bool = True,
        allowed_gam_tools: list[str] | None = None,
        gam_top_k_by_tool: dict[str, int] | None = None,
        gam_pipeline_mode: str = "default",
        card_update_dedup_config: dict[str, Any] | None = None,
    ) -> MemoryConfig:
        """Build a MemoryConfig from the old-style constructor kwargs.

        This bridges the transition period while callers are migrated.
        """
        api: ApiConfig | None = None
        if use_api:
            api = ApiConfig(
                base_url=base_url,
                namespace=namespace,
                author=author,
                channel=channel,
                sync_batch_size=max(10, sync_batch_size),
                sync_on_init=sync_on_init,
            )

        return cls(
            checkpoint_path=Path(checkpoint_path),
            search_limit=max(1, search_limit),
            rebuild_interval=max(1, rebuild_interval),
            enable_llm_synthesis=enable_llm_synthesis,
            enable_memory_evolution=enable_memory_evolution,
            enable_llm_card_enrichment=enable_llm_card_enrichment,
            api=api,
            gam=GamConfig(
                enable_bm25=enable_bm25,
                allowed_tools=list(allowed_gam_tools or []),
                top_k_by_tool=dict(gam_top_k_by_tool or {}),
                pipeline_mode=gam_pipeline_mode,
            ),
            dedup=CardUpdateDedupConfig.from_mapping(card_update_dedup_config or {}),
        )
