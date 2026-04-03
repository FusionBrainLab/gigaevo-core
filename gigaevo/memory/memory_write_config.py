"""Configuration constants for the memory write pipeline.

Loaded at import time from config/memory.yaml and environment variables.
Used by memory_write_example.py and the IdeaTracker memory write pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path

from gigaevo.memory.runtime_config import (
    deep_get,
    load_settings,
    resolve_local_path,
    resolve_settings_path,
    to_bool,
    to_int,
    to_list,
    to_str,
)

THIS_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = resolve_settings_path()
SETTINGS = load_settings(SETTINGS_PATH)

_BANKS_DIR = resolve_local_path(
    THIS_DIR,
    deep_get(SETTINGS, "paths.banks_dir"),
    default_relative="../gigaevo/memory/ideas_tracker/logs/2026-02-19_19-51-02",
)

MEMORY_DIR = resolve_local_path(
    THIS_DIR,
    deep_get(SETTINGS, "paths.checkpoint_dir"),
    default_relative="memory_usage_store/api_exp1",
)
BANKS_PATH = resolve_local_path(
    THIS_DIR,
    (
        os.getenv("MEMORY_BANKS_PATH")
        or deep_get(SETTINGS, "paths.banks_path")
        or str(_BANKS_DIR / "banks.json")
    ),
    default_relative="../gigaevo/memory/ideas_tracker/logs/2026-02-19_19-51-02/banks.json",
)
BEST_IDEAS_PATH = resolve_local_path(
    THIS_DIR,
    (
        os.getenv("MEMORY_BEST_IDEAS_PATH")
        or deep_get(SETTINGS, "paths.best_ideas_path")
        or str(_BANKS_DIR / "best_ideas.json")
    ),
    default_relative="../gigaevo/memory/ideas_tracker/logs/2026-02-19_19-51-02/best_ideas.json",
)
PROGRAMS_PATH = resolve_local_path(
    THIS_DIR,
    (
        os.getenv("MEMORY_PROGRAMS_PATH")
        or deep_get(SETTINGS, "paths.programs_path")
        or str(BANKS_PATH.parent / "programs.json")
    ),
    default_relative="../gigaevo/memory/ideas_tracker/logs/2026-02-19_19-51-02/programs.json",
)
ENABLE_USAGE_TRACKING = to_bool(
    deep_get(SETTINGS, "ideas_tracker.usage_tracking.enabled"),
    default=True,
)
_USAGE_UPDATES_RAW_PATH = (
    (
        os.getenv("MEMORY_USAGE_UPDATES_PATH")
        or deep_get(SETTINGS, "paths.memory_usage_updates_path")
    )
    if ENABLE_USAGE_TRACKING
    else None
)
USAGE_UPDATES_PATH = (
    resolve_local_path(
        THIS_DIR,
        _USAGE_UPDATES_RAW_PATH,
        default_relative="../gigaevo/memory/ideas_tracker/logs/2026-02-19_19-51-02/memory_usage_updates.json",
    )
    if _USAGE_UPDATES_RAW_PATH
    else None
)

MEMORY_API_URL = os.getenv(
    "MEMORY_API_URL",
    to_str(deep_get(SETTINGS, "api.base_url"), default="http://localhost:8000"),
)
NAMESPACE = os.getenv(
    "MEMORY_NAMESPACE",
    to_str(deep_get(SETTINGS, "api.namespace"), default="exp7"),
)
USE_API = to_bool(
    os.getenv("MEMORY_USE_API"),
    default=to_bool(deep_get(SETTINGS, "api.use_api"), default=True),
)
CHANNEL = to_str(deep_get(SETTINGS, "api.channel"), default="latest")
AUTHOR = (to_str(deep_get(SETTINGS, "api.author"), default="") or "").strip() or None

ENABLE_LLM_SYNTHESIS = to_bool(
    deep_get(SETTINGS, "runtime.enable_llm_synthesis"), default=False
)
SHOULD_EVOLVE = to_bool(deep_get(SETTINGS, "runtime.should_evolve"), default=True)
FILL_MISSING_FIELDS_WITH_LLM = to_bool(
    deep_get(SETTINGS, "runtime.fill_missing_fields_with_llm"),
    default=False,
)
SEARCH_LIMIT = max(1, to_int(deep_get(SETTINGS, "runtime.search_limit"), default=5))
REBUILD_INTERVAL = max(
    1, to_int(deep_get(SETTINGS, "runtime.rebuild_interval"), default=10)
)
SYNC_BATCH_SIZE = max(
    10, to_int(deep_get(SETTINGS, "runtime.sync_batch_size"), default=100)
)
SYNC_ON_INIT = to_bool(deep_get(SETTINGS, "runtime.sync_on_init"), default=True)

ENABLE_BM25 = to_bool(deep_get(SETTINGS, "gam.enable_bm25"), default=False)
ALLOWED_GAM_TOOLS = [
    str(tool).strip() for tool in to_list(deep_get(SETTINGS, "gam.allowed_tools"))
]
GAM_PIPELINE_MODE = to_str(
    os.getenv("MEMORY_GAM_PIPELINE_MODE"),
    default=to_str(deep_get(SETTINGS, "gam.pipeline_mode"), default="default"),
)
RAW_GAM_TOP_K_BY_TOOL = deep_get(SETTINGS, "gam.top_k_by_tool", default={})
if isinstance(RAW_GAM_TOP_K_BY_TOOL, dict):
    GAM_TOP_K_BY_TOOL = {
        str(tool).strip(): max(1, to_int(value, default=5))
        for tool, value in RAW_GAM_TOP_K_BY_TOOL.items()
        if str(tool).strip()
    }
else:
    GAM_TOP_K_BY_TOOL = {}

RAW_CARD_UPDATE_DEDUP = deep_get(SETTINGS, "card_update_dedup", default={})
if isinstance(RAW_CARD_UPDATE_DEDUP, dict):
    CARD_UPDATE_DEDUP_CONFIG = RAW_CARD_UPDATE_DEDUP
else:
    CARD_UPDATE_DEDUP_CONFIG = {}
BEST_PROGRAMS_PERCENT = max(
    0.0,
    float(
        deep_get(
            SETTINGS,
            "ideas_tracker.memory_write_pipeline.best_programs_percent",
            default=5.0,
        )
        or 0.0
    ),
)


def resolve_memory_backend_class(use_api: bool):
    """Resolve the correct AmemGamMemory class based on use_api flag."""
    if use_api:
        try:
            from gigaevo.memory_platform import AmemGamMemory as platform_backend
        except Exception as exc:
            raise RuntimeError(
                "api.use_api=true selected the platform-backed memory backend, "
                "but gigaevo.memory_platform could not be imported."
            ) from exc
        return platform_backend

    from gigaevo.memory.shared_memory.memory import AmemGamMemory as legacy_backend

    return legacy_backend
