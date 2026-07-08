"""Writable Hugging Face cache fallback for the embedding model.

The vector index loads a sentence-transformers model, which follows
``HF_HOME`` and related env vars. Shared clusters often point these at NFS
roots that this user cannot write, breaking the embedding download. This
module clears unwritable entries and falls back to ``~/.cache/huggingface``.
It is the one place in the memory system that touches ``os.environ``; it runs
when the index is built, not at import.
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger


def _hf_cache_dir_usable(path: Path) -> bool:
    try:
        path = path.expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".gigaevo_hf_write_probe"
        probe.write_text("ok", encoding="ascii")
        probe.unlink()
        return True
    except OSError:
        return False


def ensure_writable_hf_cache() -> None:
    """Clear unwritable HF cache env entries and fall back to ``~/.cache``."""
    fallback = Path.home() / ".cache" / "huggingface"
    keys = (
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "SENTENCE_TRANSFORMERS_HOME",
    )
    for key in keys:
        raw = os.environ.get(key)
        if not raw or not str(raw).strip():
            continue
        if not _hf_cache_dir_usable(Path(raw)):
            logger.warning("[Memory][Store] Clearing unwritable {}={!r}", key, raw)
            os.environ.pop(key, None)

    hf = os.environ.get("HF_HOME")
    if hf and _hf_cache_dir_usable(Path(hf)):
        return

    fallback.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(fallback)
    hub = fallback / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub)
    os.environ["TRANSFORMERS_CACHE"] = str(hub)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(fallback)
    logger.warning("[Memory][Store] HF cache directory -> {}", fallback)
