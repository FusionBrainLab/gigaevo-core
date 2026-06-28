"""Local card-bank backend constructor.

The ``config/memory/common/backend=local`` group binds this as a Hydra ``_partial_``
over a fully-resolved :class:`MemoryConfig`; ``MemorySystem`` completes the
partial with the shared memory llm, and the read/write callers complete it with
the per-run ``checkpoint_dir`` (and read-side ``gam`` / write-side ``evictor``).
There is no factory object: every knob lives in the Hydra ``MemoryConfig`` node,
construction is one function. Fails fast with :class:`MemoryStorageError` — a
misconfigured backend must abort the run, never silently degrade to no memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from gigaevo.exceptions import MemoryStorageError
from gigaevo.memory.shared_memory.memory_config import GamConfig, MemoryConfig

if TYPE_CHECKING:
    from gigaevo.memory.shared_memory.memory import AmemGamMemory


def build_local_backend(
    *,
    config: MemoryConfig,
    llm_service: Any | None = None,
    evictor: Any | None = None,
    gam: GamConfig | None = None,
    checkpoint_dir: str | Path | None = None,
) -> AmemGamMemory:
    """Construct the local card bank from a resolved config.

    ``checkpoint_dir`` (per-run output dir) and ``gam`` (read-side retrieval
    knobs) override the config at call time; both are applied on a copy so the
    shared Hydra config node is never mutated.
    """
    # deferred so importing this module (Hydra config resolution) does not pull
    # in the embedding / agentic-runtime stack
    from gigaevo.memory.shared_memory.memory import AmemGamMemory

    overrides: dict[str, Any] = {}
    if checkpoint_dir is not None:
        overrides["checkpoint_path"] = Path(checkpoint_dir)
    if gam is not None:
        overrides["gam"] = gam
    # Always copy: the read and write builds share one Hydra config node, so
    # each backend must own a distinct config even with no overrides.
    cfg = config.model_copy(update=overrides)

    try:
        memory = AmemGamMemory(config=cfg, llm_service=llm_service, evictor=evictor)
    except Exception as exc:
        logger.error("[Memory][Backend] Local backend init failed: {}", exc)
        raise MemoryStorageError(
            f"Memory backend initialization failed: {exc}"
        ) from exc
    logger.info(
        "[Memory][Backend] Built local backend (checkpoint={})", cfg.checkpoint_path
    )
    return memory
