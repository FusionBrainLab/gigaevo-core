"""Memory backend factories — Hydra-composable, pure construction.

The ``config/memory/backend/`` group selects which factory the provider gets:
``local`` (canonical local card bank). Factories are plain pydantic models: no yaml parsing, no dotenv,
no env vars — every knob arrives through Hydra. ``build()`` fails fast with
:class:`MemoryStorageError` — a misconfigured backend must abort the run,
never silently degrade to a no-memory run. (Per-selection outages are
absorbed downstream by the read pipeline's fail-to-empty guard.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict

from gigaevo.exceptions import MemoryStorageError
from gigaevo.memory.shared_memory.memory_config import GamConfig, MemoryConfig


class MemoryBackendFactory(BaseModel, ABC):
    """Deferred constructor for a card-bank backend.

    Hydra instantiates the factory at config-resolution time; the provider
    calls :meth:`build` lazily on first card selection so heavy backend
    initialization never blocks config composition.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    checkpoint_dir: Path | None = None
    llm: Any | None = None
    embedding_model_name: str = "all-MiniLM-L6-v2"
    search_limit: int = 5
    rebuild_interval: int = 30
    enable_llm_synthesis: bool = False
    enable_memory_evolution: bool = False
    enable_llm_card_enrichment: bool = False

    @abstractmethod
    def backend_class(self) -> type[Any]:
        """Return the backend class to construct (test seam)."""

    @abstractmethod
    def build(
        self,
        *,
        checkpoint_dir: str | Path | None = None,
        gam: GamConfig | None = None,
        evictor: Any | None = None,
    ) -> Any:
        """Construct the backend.

        Args:
            checkpoint_dir: runtime override for the configured checkpoint dir
                (the engine pins per-run artefacts under the Hydra output dir).
            gam: retrieval knobs, normally derived from the configured
                ``GamRetriever``.
            evictor: optional write-side component.

        Raises:
            MemoryStorageError: if construction fails or no checkpoint dir
                is available.
        """

    def resolve_checkpoint_dir(self, override: str | Path | None) -> Path:
        """Return the effective checkpoint dir (runtime override wins)."""
        target = override if override is not None else self.checkpoint_dir
        if target is None:
            raise MemoryStorageError(
                "Memory backend needs a checkpoint dir: set backend.checkpoint_dir "
                "in config/memory/backend/ or pass checkpoint_dir at build time."
            )
        return Path(target)


class LocalMemoryBackendFactory(MemoryBackendFactory):
    """Canonical local card-bank backend (``memory/backend=local``)."""

    def backend_class(self) -> type[Any]:
        from gigaevo.memory.shared_memory.memory import AmemGamMemory

        return AmemGamMemory

    def build(
        self,
        *,
        checkpoint_dir: str | Path | None = None,
        gam: GamConfig | None = None,
        evictor: Any | None = None,
    ) -> Any:
        target = self.resolve_checkpoint_dir(checkpoint_dir)
        config = MemoryConfig(
            checkpoint_path=target,
            embedding_model_name=self.embedding_model_name,
            search_limit=self.search_limit,
            rebuild_interval=self.rebuild_interval,
            enable_llm_synthesis=self.enable_llm_synthesis,
            enable_memory_evolution=self.enable_memory_evolution,
            enable_llm_card_enrichment=self.enable_llm_card_enrichment,
            api=None,
            gam=gam if gam is not None else GamConfig(),
        )
        try:
            memory = self.backend_class()(
                config=config,
                **({"llm_service": self.llm} if self.llm is not None else {}),
                **({"evictor": evictor} if evictor is not None else {}),
            )
        except Exception as exc:
            logger.error("[Memory][BackendFactory] Local backend init failed: {}", exc)
            raise MemoryStorageError(
                f"Memory backend initialization failed: {exc}"
            ) from exc
        logger.info(
            "[Memory][BackendFactory] Built local backend (class={}, checkpoint={})",
            type(memory).__module__,
            target,
        )
        return memory
