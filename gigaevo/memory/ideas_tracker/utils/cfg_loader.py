from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
import yaml

from gigaevo.memory.ideas_tracker.components.fabrics.fabric_redis import (
    create_redis_config,
)
from gigaevo.memory.ideas_tracker.utils.helpers import to_bool, to_float
from tools.utils import RedisRunConfig

_DEFAULT_BEST_PROGRAMS_PERCENT = 5.0


def _parse_memory_write_pipeline(raw: Any) -> tuple[bool, float]:
    if isinstance(raw, dict):
        enabled = to_bool(raw.get("enabled", False), default=False)
        pct = to_float(raw.get("best_programs_percent"))
    else:
        enabled = to_bool(raw, default=False)
        pct = None
    resolved = pct if pct is not None else _DEFAULT_BEST_PROGRAMS_PERCENT
    return enabled, max(0.0, resolved)


def _parse_usage_tracking(raw: Any) -> tuple[dict[str, Any], bool]:
    if isinstance(raw, dict):
        return raw, to_bool(raw.get("enabled", True), default=True)
    return {"enabled": raw}, to_bool(raw, default=True)


def _load_config(
    config_path: str | Path | None, idea_tracker_location: Path
) -> dict[str, Any]:
    """
    Load IdeaTracker configuration from YAML file.

    Resolves project root (3 levels up from this file) to find default config
    at config/memory.yaml when no path is provided.
    If the loaded file contains an ``ideas_tracker`` section, that section is used.
    Otherwise, the full file payload is treated as legacy IdeaTracker config.

    Args:
        config_path: Path to configuration file, or None to use default location.

    Returns:
        Dictionary containing configuration values with keys: gen_delta, model, redis.
        Returns default configuration if file is missing.
    """
    default_config: dict[str, Any] = {
        "gen_delta": 100000,
        "list_max_ideas": 5,
        "model": "deepseek/deepseek-v3.2",
        "base_url": "https://openrouter.ai/api/v1",
        "redis": {
            "redis_host": "localhost",
            "redis_port": 6379,
            "redis_db": 0,
            "redis_prefix": "heilbron",
            "label": "",
        },
        "memory_write_pipeline": {"enabled": False},
        "usage_tracking": {"enabled": True},
    }

    if config_path is None:
        project_root = idea_tracker_location.parents[3]
        path_obj = project_root / "config" / "memory.yaml"
    else:
        path_obj = Path(config_path)

    if not path_obj.is_file():
        return default_config

    with path_obj.open("r", encoding="utf-8") as f:
        try:
            payload = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            logger.error(f"Error loading config file: {e}")
            return default_config

    if not isinstance(payload, dict):
        return default_config

    # Unified config format stores tracker settings under ideas_tracker.
    ideas_tracker_cfg = payload.get("ideas_tracker")
    if isinstance(ideas_tracker_cfg, dict):
        return ideas_tracker_cfg

    # Backward-compatible fallback: treat the whole payload as tracker config.
    return payload


@dataclass
class PipelineConfig:
    list_max_ideas: int = 5
    analyzer_settings: dict[str, Any] = field(default_factory=dict)
    analyzer_pipeline_type: str = "default"
    postprocessing: dict[str, Any] = field(default_factory=dict)
    description_rewriting: bool = True

    def from_dict(self, cfg_dict) -> None:
        self.list_max_ideas = cfg_dict.get("list_max_ideas", 5)
        self.analyzer_settings = cfg_dict.get("analyzer", {})
        self.analyzer_settings["analyzer_fast_settings"] = cfg_dict.get(
            "analyzer_fast_settings", {}
        )
        self.analyzer_pipeline_type = self.analyzer_settings.get("type", "default")
        self.postprocessing = self.analyzer_settings.get(
            "postprocessing", {"type": "default"}
        )
        self.description_rewriting = cfg_dict.get("description_rewriting", True)


@dataclass
class MemoryConfig:
    memory_write_pipeline_enabled: bool = False
    best_programs_percent: float = _DEFAULT_BEST_PROGRAMS_PERCENT
    memory_usage_tracking_enabled: bool = True
    usage_tracking: dict[str, Any] = field(default_factory=dict)

    def from_dict(self, cfg_dict) -> None:
        mwp = cfg_dict.get("memory_write_pipeline", {})
        self.memory_write_pipeline_enabled, self.best_programs_percent = (
            _parse_memory_write_pipeline(mwp)
        )
        ut = cfg_dict.get("usage_tracking", {"enabled": True})
        self.usage_tracking, self.memory_usage_tracking_enabled = _parse_usage_tracking(
            ut
        )


@dataclass
class IdeaTrackerConfig:
    redis_config: RedisRunConfig
    pipeline_config: PipelineConfig
    memory_config: MemoryConfig


def load_config(
    config_path: str | Path | None, idea_tracker_location: Path
) -> IdeaTrackerConfig:
    config_dict = _load_config(config_path, idea_tracker_location)
    redis_config = create_redis_config(config_dict.get("redis", {}))
    pipeline_config = PipelineConfig()
    pipeline_config.from_dict(config_dict)
    memory_config = MemoryConfig()
    memory_config.from_dict(config_dict)
    config = IdeaTrackerConfig(
        redis_config=redis_config,
        pipeline_config=pipeline_config,
        memory_config=memory_config,
    )
    return config
