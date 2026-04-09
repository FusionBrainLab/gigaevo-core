"""Configuration loader for the memory write pipeline.

``load_config(settings_path)`` is the public API.

Config is loaded lazily via OmegaConf, which resolves ``${oc.env:VAR,default}``
interpolations declared in the YAML file.  All defaults live in the schema
dataclasses below — no manual ``to_bool``/``to_int``/``.get(key, default)``
needed, OmegaConf enforces types automatically when merging a structured
schema with the loaded YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from gigaevo.memory.runtime_config import resolve_local_path, resolve_settings_path

THIS_DIR = Path(__file__).resolve().parent

_DEFAULT_LOGS_REL = "../gigaevo/memory/ideas_tracker/logs/2026-02-19_19-51-02"

# ---------------------------------------------------------------------------
# OmegaConf section schemas — define defaults and enforce types
# ---------------------------------------------------------------------------


@dataclass
class _PathsSchema:
    checkpoint_dir: str = "memory_usage_store/api_exp1"
    banks_dir: str = _DEFAULT_LOGS_REL
    # Empty string means "derive from banks_dir"; overridden via ${oc.env:...}
    banks_path: str = ""
    best_ideas_path: str = ""
    programs_path: str = ""
    memory_usage_updates_path: str = ""


@dataclass
class _ApiSchema:
    base_url: str = "http://localhost:8000"
    namespace: str = "default"
    use_api: bool = False
    channel: str = "latest"
    author: str | None = None


@dataclass
class _RuntimeSchema:
    enable_llm_synthesis: bool = False
    should_evolve: bool = True
    fill_missing_fields_with_llm: bool = False
    search_limit: int = 5
    rebuild_interval: int = 10
    sync_batch_size: int = 100
    sync_on_init: bool = True


@dataclass
class _GamSchema:
    enable_bm25: bool = False
    pipeline_mode: str = "default"
    allowed_tools: list = field(default_factory=list)
    top_k_by_tool: dict = field(default_factory=dict)


@dataclass
class _PipelineSchema:
    enabled: bool = True
    best_programs_percent: float = 5.0


@dataclass
class _UsageTrackingSchema:
    enabled: bool = True


# ---------------------------------------------------------------------------
# Public config object
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """All configuration values for the memory write pipeline.

    Constructed by ``load_config()`` — do not instantiate directly.
    """

    settings_path: Path
    banks_path: Path
    best_ideas_path: Path
    programs_path: Path
    usage_updates_path: Path | None
    memory_dir: Path
    enable_usage_tracking: bool
    memory_api_url: str
    namespace: str
    use_api: bool
    channel: str
    author: str | None
    enable_llm_synthesis: bool
    should_evolve: bool
    fill_missing_fields_with_llm: bool
    search_limit: int
    rebuild_interval: int
    sync_batch_size: int
    sync_on_init: bool
    enable_bm25: bool
    allowed_gam_tools: list[str] = field(default_factory=list)
    gam_pipeline_mode: str = "default"
    gam_top_k_by_tool: dict[str, int] = field(default_factory=dict)
    card_update_dedup_config: dict[str, Any] = field(default_factory=dict)
    best_programs_percent: float = 5.0


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _merge_section(file_cfg: Any, key: str, schema_cls: type) -> Any:
    """Merge ``file_cfg[key]`` into a structured schema, returning a plain dict."""
    raw = file_cfg.get(key) or {}
    merged = OmegaConf.merge(OmegaConf.structured(schema_cls), raw)
    return OmegaConf.to_container(merged, resolve=True)


def _resolve_path(base: Path, raw: str, default_relative: str) -> Path:
    """Resolve raw path string (empty → use default_relative)."""
    return resolve_local_path(
        base, raw.strip() or None, default_relative=default_relative
    )


def load_config(settings_path: Path | None = None) -> PipelineConfig:
    """Load pipeline config from *settings_path* (or the env-var / default).

    OmegaConf resolves ``${oc.env:VAR,default}`` nodes declared in the YAML
    at call time.  The structured schemas above enforce Python types — no
    manual type-conversion helpers needed.
    """
    _settings_path = resolve_settings_path(settings_path)
    file_cfg = OmegaConf.load(_settings_path)

    paths: dict[str, Any] = _merge_section(file_cfg, "paths", _PathsSchema)
    api: dict[str, Any] = _merge_section(file_cfg, "api", _ApiSchema)
    runtime: dict[str, Any] = _merge_section(file_cfg, "runtime", _RuntimeSchema)
    gam: dict[str, Any] = _merge_section(file_cfg, "gam", _GamSchema)

    it_raw = OmegaConf.select(file_cfg, "ideas_tracker") or {}
    pipeline: dict[str, Any] = _merge_section(
        it_raw, "memory_write_pipeline", _PipelineSchema
    )
    usage_tracking: dict[str, Any] = _merge_section(
        it_raw, "usage_tracking", _UsageTrackingSchema
    )

    raw_dedup = OmegaConf.select(file_cfg, "card_update_dedup")
    dedup: dict[str, Any] = (
        OmegaConf.to_container(raw_dedup, resolve=True)  # type: ignore[arg-type]
        if raw_dedup is not None
        else {}
    )

    # -- Derived paths -------------------------------------------------------
    banks_dir = _resolve_path(THIS_DIR, paths["banks_dir"], _DEFAULT_LOGS_REL)
    memory_dir = _resolve_path(
        THIS_DIR, paths["checkpoint_dir"], "memory_usage_store/api_exp1"
    )

    banks_path = _resolve_path(
        THIS_DIR,
        paths["banks_path"] or str(banks_dir / "banks.json"),
        f"{_DEFAULT_LOGS_REL}/banks.json",
    )
    best_ideas_path = _resolve_path(
        THIS_DIR,
        paths["best_ideas_path"] or str(banks_dir / "best_ideas.json"),
        f"{_DEFAULT_LOGS_REL}/best_ideas.json",
    )
    programs_path = _resolve_path(
        THIS_DIR,
        paths["programs_path"] or str(banks_path.parent / "programs.json"),
        f"{_DEFAULT_LOGS_REL}/programs.json",
    )

    enable_usage_tracking: bool = usage_tracking["enabled"]
    raw_usage = paths["memory_usage_updates_path"].strip()
    usage_updates_path: Path | None = (
        _resolve_path(
            THIS_DIR, raw_usage, f"{_DEFAULT_LOGS_REL}/memory_usage_updates.json"
        )
        if (enable_usage_tracking and raw_usage)
        else None
    )

    # -- GAM top-k (int coercion) --------------------------------------------
    gam_top_k_by_tool: dict[str, int] = {
        str(k): max(1, int(v))
        for k, v in (gam["top_k_by_tool"] or {}).items()
        if str(k).strip()
    }

    return PipelineConfig(
        settings_path=_settings_path,
        banks_path=banks_path,
        best_ideas_path=best_ideas_path,
        programs_path=programs_path,
        usage_updates_path=usage_updates_path,
        memory_dir=memory_dir,
        enable_usage_tracking=enable_usage_tracking,
        memory_api_url=str(api["base_url"]),
        namespace=str(api["namespace"]),
        use_api=bool(api["use_api"]),
        channel=str(api["channel"]),
        author=(str(api["author"]).strip() or None) if api["author"] else None,
        enable_llm_synthesis=bool(runtime["enable_llm_synthesis"]),
        should_evolve=bool(runtime["should_evolve"]),
        fill_missing_fields_with_llm=bool(runtime["fill_missing_fields_with_llm"]),
        search_limit=max(1, int(runtime["search_limit"])),
        rebuild_interval=max(1, int(runtime["rebuild_interval"])),
        sync_batch_size=max(10, int(runtime["sync_batch_size"])),
        sync_on_init=bool(runtime["sync_on_init"]),
        enable_bm25=bool(gam["enable_bm25"]),
        allowed_gam_tools=[
            str(t).strip() for t in (gam["allowed_tools"] or []) if str(t).strip()
        ],
        gam_pipeline_mode=str(gam["pipeline_mode"]),
        gam_top_k_by_tool=gam_top_k_by_tool,
        card_update_dedup_config=dedup if isinstance(dedup, dict) else {},
        best_programs_percent=max(0.0, float(pipeline["best_programs_percent"])),
    )
