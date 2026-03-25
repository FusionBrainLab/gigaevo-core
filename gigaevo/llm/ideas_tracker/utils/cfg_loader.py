from pathlib import Path
from typing import Any, Optional

import yaml


def _load_config(
    config_path: Optional[str | Path], idea_tracker_location: Path
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
        payload = yaml.safe_load(f) or {}

    if not isinstance(payload, dict):
        return default_config

    # Unified config format stores tracker settings under ideas_tracker.
    ideas_tracker_cfg = payload.get("ideas_tracker")
    if isinstance(ideas_tracker_cfg, dict):
        return ideas_tracker_cfg

    # Backward-compatible fallback: treat the whole payload as tracker config.
    return payload
