"""Memory backend factories — Hydra-composable, pure construction.

The ``config/memory/backend/`` group selects which factory the provider gets:
``local`` (canonical local card bank) or ``legacy_api`` (deprecated HTTP API
backend). Factories are plain pydantic models: no yaml parsing, no dotenv,
no env vars — every knob arrives through Hydra. ``build()`` fails fast with
:class:`MemoryStorageError` — a misconfigured backend must abort the run,
never silently degrade to a no-memory run. (Per-selection outages are
absorbed downstream by the read pipeline's fail-to-empty guard.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import warnings

from loguru import logger
from pydantic import BaseModel, ConfigDict

from gigaevo.exceptions import MemoryStorageError
from gigaevo.memory.shared_memory.card_update_dedup import CardUpdateDedupConfig
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
        deduplicator: Any | None = None,
    ) -> Any:
        """Construct the backend.

        Args:
            checkpoint_dir: runtime override for the configured checkpoint dir
                (the engine pins per-run artefacts under the Hydra output dir).
            gam: retrieval knobs, normally derived from the configured
                ``GamRetriever``.
            evictor / deduplicator: optional write-side components.

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

    dedup: CardUpdateDedupConfig = CardUpdateDedupConfig()

    def backend_class(self) -> type[Any]:
        from gigaevo.memory.shared_memory.memory import AmemGamMemory

        return AmemGamMemory

    def build(
        self,
        *,
        checkpoint_dir: str | Path | None = None,
        gam: GamConfig | None = None,
        evictor: Any | None = None,
        deduplicator: Any | None = None,
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
            dedup=self.dedup,
        )
        try:
            memory = self.backend_class()(
                config=config,
                **({"llm_service": self.llm} if self.llm is not None else {}),
                **({"evictor": evictor} if evictor is not None else {}),
                **({"deduplicator": deduplicator} if deduplicator is not None else {}),
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


class LegacyApiMemoryBackendFactory(MemoryBackendFactory):
    """DEPRECATED HTTP API backend (``memory/backend=legacy_api``).

    Kept only so historical experiments remain reproducible; new runs should
    use :class:`LocalMemoryBackendFactory`. Instantiation emits a
    ``DeprecationWarning``.
    """

    base_url: str = "http://localhost:8000"
    # top-level `namespace` defaults to null, so the composed node may be None;
    # build() fails fast rather than silently reading the wrong remote bank.
    namespace: str | None = None
    channel: str = "latest"
    author: str | None = None
    sync_batch_size: int = 100
    sync_on_init: bool = True
    enable_bm25: bool = False
    allowed_gam_tools: list[str] = []
    gam_top_k_by_tool: dict[str, int] = {}
    gam_pipeline_mode: str = "default"

    def model_post_init(self, __context: Any) -> None:
        warnings.warn(
            "LegacyApiMemoryBackendFactory (memory/backend=legacy_api) is "
            "deprecated; use the local backend (memory/backend=local).",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.warning(
            "[Memory][BackendFactory] Using DEPRECATED legacy API backend ({})",
            self.base_url,
        )

    def backend_class(self) -> type[Any]:
        from gigaevo.memory_platform import AmemGamMemory

        return AmemGamMemory

    def build(
        self,
        *,
        checkpoint_dir: str | Path | None = None,
        gam: GamConfig | None = None,
        evictor: Any | None = None,
        deduplicator: Any | None = None,
    ) -> Any:
        if self.namespace is None:
            raise MemoryStorageError(
                "Legacy API backend needs a namespace: pass namespace=<ns> at "
                "the Hydra top level or set memory.backend.namespace."
            )
        if evictor is not None or deduplicator is not None:
            logger.warning(
                "[Memory][BackendFactory] Legacy API backend ignores injected "
                "evictor/deduplicator (harm gating is inline at save_card); "
                "the configured memory/evictor + memory/dedup singletons will "
                "not run"
            )
        target = self.resolve_checkpoint_dir(checkpoint_dir)
        effective_gam = (
            gam
            if gam is not None
            else GamConfig(
                enable_bm25=self.enable_bm25,
                allowed_tools=self.allowed_gam_tools,
                top_k_by_tool=self.gam_top_k_by_tool,
                pipeline_mode=self.gam_pipeline_mode,
            )
        )
        try:
            memory = self.backend_class()(
                **({"llm_service": self.llm} if self.llm is not None else {}),
                embedding_model_name=self.embedding_model_name,
                checkpoint_path=str(target),
                base_url=self.base_url,
                use_api=True,
                namespace=self.namespace,
                author=self.author,
                channel=self.channel,
                search_limit=self.search_limit,
                enable_llm_synthesis=self.enable_llm_synthesis,
                enable_memory_evolution=self.enable_memory_evolution,
                enable_llm_card_enrichment=self.enable_llm_card_enrichment,
                rebuild_interval=self.rebuild_interval,
                enable_bm25=effective_gam.enable_bm25,
                sync_batch_size=self.sync_batch_size,
                sync_on_init=self.sync_on_init,
                allowed_gam_tools=list(effective_gam.allowed_tools),
                gam_top_k_by_tool=dict(effective_gam.top_k_by_tool),
                gam_pipeline_mode=effective_gam.pipeline_mode,
            )
        except Exception as exc:
            logger.error(
                "[Memory][BackendFactory] Legacy API backend init failed: {}", exc
            )
            raise MemoryStorageError(
                f"Memory backend initialization failed: {exc}"
            ) from exc
        logger.info(
            "[Memory][BackendFactory] Built legacy API backend "
            "(base_url={}, namespace={}, checkpoint={})",
            self.base_url,
            self.namespace,
            target,
        )
        return memory
