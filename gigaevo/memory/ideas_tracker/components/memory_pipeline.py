import importlib
import json
import os
from pathlib import Path
from typing import Any

from loguru import logger as _logger

from gigaevo.memory.ideas_tracker.components.data_components import RecordBank
from gigaevo.memory.ideas_tracker.utils.helpers import merge_usage_payloads
from gigaevo.memory.ideas_tracker.utils.it_logger import IdeasTrackerLogger


def _has_best_ideas_snapshot(best_ideas_path: Path) -> bool:
    """
    Check that best_ideas.json contains at least one snapshot with 'best_ideas'.
    """
    try:
        with best_ideas_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    if isinstance(payload, dict):
        return "best_ideas" in payload
    if isinstance(payload, list):
        return any(isinstance(item, dict) and "best_ideas" in item for item in payload)
    return False


def run_memory_write_pipeline(
    memory_write_pipeline_enabled: bool,
    memory_usage_tracking_enabled: bool,
    logger: IdeasTrackerLogger,
) -> None:
    """
    Optionally run memory_write_example.py using current run's banks/best ideas logs.
    """
    if not memory_write_pipeline_enabled:
        return

    banks_path = logger.banks_file
    best_ideas_path = logger.best_ideas_file

    if banks_path is None or best_ideas_path is None:
        _logger.warning(
            "Memory write pipeline skipped: logger output paths are unavailable."
        )
        return
    if not banks_path.exists():
        _logger.warning(
            f"Memory write pipeline skipped: missing banks file at {banks_path}."
        )
        return
    if not best_ideas_path.exists() or not _has_best_ideas_snapshot(best_ideas_path):
        _logger.warning(
            "Memory write pipeline skipped: best_ideas snapshot was not generated for this run."
        )
        return

    env_overrides = {
        "MEMORY_BANKS_PATH": str(banks_path),
        "MEMORY_BEST_IDEAS_PATH": str(best_ideas_path),
    }
    programs_path = logger.programs_file
    if programs_path is not None and programs_path.exists():
        env_overrides["MEMORY_PROGRAMS_PATH"] = str(programs_path)
    usage_updates_path = logger.memory_usage_updates_file
    if (
        memory_usage_tracking_enabled
        and usage_updates_path is not None
        and usage_updates_path.exists()
    ):
        env_overrides["MEMORY_USAGE_UPDATES_PATH"] = str(usage_updates_path)
    previous_env = {key: os.environ.get(key) for key in env_overrides}

    try:
        os.environ.update(env_overrides)
        memory_write_module = importlib.import_module(
            "gigaevo.memory.memory_write_example"
        )
        memory_write_module = importlib.reload(memory_write_module)
        snapshot = memory_write_module.main()
        if isinstance(snapshot, dict):
            stats = snapshot.get("stats", {})
            stats_by_card_type = snapshot.get("stats_by_card_type", {})
            ideas_stats = (
                stats_by_card_type.get("ideas", {})
                if isinstance(stats_by_card_type, dict)
                else {}
            )
            programs_stats = (
                stats_by_card_type.get("programs", {})
                if isinstance(stats_by_card_type, dict)
                else {}
            )
            if isinstance(stats, dict):
                _logger.info(
                    "Memory write pipeline stats: "
                    f"processed={stats.get('processed', 0)}, "
                    f"ideas_processed={ideas_stats.get('processed', 0)}, "
                    f"programs_processed={programs_stats.get('processed', 0)}, "
                    f"added={stats.get('added', 0)}, "
                    f"ideas_added={ideas_stats.get('added', 0)}, "
                    f"programs_added={programs_stats.get('added', 0)}, "
                    f"updated={stats.get('updated', 0)}, "
                    f"ideas_updated={ideas_stats.get('updated', 0)}, "
                    f"programs_updated={programs_stats.get('updated', 0)}, "
                    f"rejected={stats.get('rejected', 0)}"
                )
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def apply_memory_usage_updates_to_idea_banks(
    bank: RecordBank, memory_usage_updates_by_card: dict[str, dict[str, Any]]
) -> None:
    for idea in bank.all_ideas_cards():
        usage_update = memory_usage_updates_by_card.get(str(idea.id or ""))
        if not usage_update:
            continue
        merged_usage = merge_usage_payloads(
            getattr(idea, "usage", {}),
            usage_update,
        )
        idea.update_metadata(usage=merged_usage)
