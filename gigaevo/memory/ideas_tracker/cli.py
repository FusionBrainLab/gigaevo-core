from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml

from gigaevo.memory.ideas_tracker.csv_loader import load_programs_from_csv
from gigaevo.memory.ideas_tracker.redis_loader import load_programs_from_redis


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_config_path() -> Path:
    return _project_root() / "config" / "memory.yaml"


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file_obj:
        payload = yaml.safe_load(file_obj) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML mapping in {path}")

    return payload


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _ensure_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        value = {}
        payload[key] = value
    return value


def _resolve_project_relative_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return _project_root() / path


def _build_runtime_memory_payload(config_path: Path | None) -> dict[str, Any]:
    default_payload = _load_yaml_mapping(_default_config_path())
    if config_path is None:
        return default_payload

    custom_payload = _load_yaml_mapping(config_path)
    if isinstance(custom_payload.get("ideas_tracker"), dict):
        return _merge_dicts(default_payload, custom_payload)

    return _merge_dicts(default_payload, {"ideas_tracker": custom_payload})


def _write_runtime_memory_config(payload: dict[str, Any]) -> Path:
    runtime_dir = Path(tempfile.mkdtemp(prefix="ideas-tracker-", dir="/tmp"))
    runtime_path = runtime_dir / "memory.runtime.yaml"
    with runtime_path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(payload, file_obj, sort_keys=False)
    return runtime_path


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ideas tracker independently from run.py using an existing Redis run database."
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help=(
            "Optional YAML config path. May be a full unified memory config or "
            "a tracker-only config. Defaults to config/memory.yaml."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Override paths.checkpoint_dir for the final memory write pipeline. "
            "Useful when running the tracker after run.py."
        ),
    )
    parser.add_argument(
        "--logs-dir",
        default=None,
        help=(
            "Write ideas_tracker logs into this existing directory. "
            "A timestamped subfolder will be created per run."
        ),
    )
    parser.add_argument(
        "--memory-write",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override ideas_tracker.memory_write_pipeline.enabled.",
    )
    parser.add_argument("--redis-host", default=None, help="Redis host override.")
    parser.add_argument(
        "--redis-port", type=int, default=None, help="Redis port override."
    )
    parser.add_argument("--redis-db", type=int, default=None, help="Redis DB override.")
    parser.add_argument(
        "--redis-prefix",
        default=None,
        help="Redis key prefix override. This usually matches the problem name.",
    )
    parser.add_argument(
        "--redis-label",
        default=None,
        help="Optional Redis label override for logging/debugging.",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help=(
            "Path to evolution_data.csv exported by tools/redis2pd.py. "
            "When provided, programs are loaded from the CSV instead of Redis."
        ),
    )
    return parser


def _apply_cli_overrides(
    payload: dict[str, Any],
    *,
    checkpoint_dir: str | None,
    memory_write: bool | None,
    redis_host: str | None,
    redis_port: int | None,
    redis_db: int | None,
    redis_prefix: str | None,
    redis_label: str | None,
) -> dict[str, Any]:
    result = dict(payload)

    if checkpoint_dir is not None:
        paths_cfg = _ensure_mapping(result, "paths")
        paths_cfg["checkpoint_dir"] = str(
            _resolve_project_relative_path(checkpoint_dir)
        )

    ideas_tracker_cfg = _ensure_mapping(result, "ideas_tracker")
    redis_cfg = _ensure_mapping(ideas_tracker_cfg, "redis")

    if redis_host is not None:
        redis_cfg["redis_host"] = redis_host
    if redis_port is not None:
        redis_cfg["redis_port"] = int(redis_port)
    if redis_db is not None:
        redis_cfg["redis_db"] = int(redis_db)
    if redis_prefix is not None:
        redis_cfg["redis_prefix"] = redis_prefix
    if redis_label is not None:
        redis_cfg["label"] = redis_label

    if memory_write is not None:
        memory_write_cfg = ideas_tracker_cfg.get("memory_write_pipeline")
        if not isinstance(memory_write_cfg, dict):
            memory_write_cfg = {}
            ideas_tracker_cfg["memory_write_pipeline"] = memory_write_cfg
        memory_write_cfg["enabled"] = bool(memory_write)

    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    from gigaevo.memory.ideas_tracker.ideas_tracker import IdeaTracker

    config_path = Path(args.config_path) if args.config_path else None
    runtime_payload = _build_runtime_memory_payload(config_path)
    runtime_payload = _apply_cli_overrides(
        runtime_payload,
        checkpoint_dir=args.checkpoint_dir,
        memory_write=args.memory_write,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
        redis_prefix=args.redis_prefix,
        redis_label=args.redis_label,
    )
    runtime_config_path = _write_runtime_memory_config(runtime_payload)

    previous_config_path = os.environ.get("EVO_MEMORY_CONFIG_PATH")
    os.environ["EVO_MEMORY_CONFIG_PATH"] = str(runtime_config_path)
    try:
        tracker = IdeaTracker(
            logs_dir=args.logs_dir, redis_prefix=args.redis_prefix or ""
        )
        if args.csv_path is not None:
            programs = load_programs_from_csv(args.csv_path)
        else:
            programs = load_programs_from_redis(
                host=args.redis_host or "localhost",
                port=args.redis_port or 6379,
                db=args.redis_db or 0,
                prefix=args.redis_prefix or "",
            )
        tracker.run(programs)
    finally:
        if previous_config_path is None:
            os.environ.pop("EVO_MEMORY_CONFIG_PATH", None)
        else:
            os.environ["EVO_MEMORY_CONFIG_PATH"] = previous_config_path

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
