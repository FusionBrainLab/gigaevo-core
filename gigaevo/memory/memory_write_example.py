from __future__ import annotations

from datetime import UTC, datetime
import json
import math
from math import ceil
import os
from pathlib import Path
import statistics
from typing import Any, Protocol

from dotenv import load_dotenv

try:
    from .runtime_config import (
        deep_get,
        load_settings,
        resolve_local_path,
        resolve_settings_path,
        to_bool,
        to_int,
        to_list,
        to_str,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from runtime_config import (
        deep_get,
        load_settings,
        resolve_local_path,
        resolve_settings_path,
        to_bool,
        to_int,
        to_list,
        to_str,
    )

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)


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
AUTHOR = to_str(deep_get(SETTINGS, "api.author"), default="").strip() or None

ENABLE_LLM_SYNTHESIS = to_bool(deep_get(SETTINGS, "runtime.enable_llm_synthesis"), default=False)
SHOULD_EVOLVE = to_bool(deep_get(SETTINGS, "runtime.should_evolve"), default=True)
FILL_MISSING_FIELDS_WITH_LLM = to_bool(
    deep_get(SETTINGS, "runtime.fill_missing_fields_with_llm"),
    default=False,
)
SEARCH_LIMIT = max(1, to_int(deep_get(SETTINGS, "runtime.search_limit"), default=5))
REBUILD_INTERVAL = max(1, to_int(deep_get(SETTINGS, "runtime.rebuild_interval"), default=10))
SYNC_BATCH_SIZE = max(10, to_int(deep_get(SETTINGS, "runtime.sync_batch_size"), default=100))
SYNC_ON_INIT = to_bool(deep_get(SETTINGS, "runtime.sync_on_init"), default=True)

ENABLE_BM25 = to_bool(deep_get(SETTINGS, "gam.enable_bm25"), default=False)
ALLOWED_GAM_TOOLS = [str(tool).strip() for tool in to_list(deep_get(SETTINGS, "gam.allowed_tools"))]
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


def _resolve_memory_backend_class(use_api: bool):
    if use_api:
        try:
            from gigaevo.memory_platform import AmemGamMemory as platform_backend
        except Exception as exc:  # pragma: no cover - runtime dependency issue
            raise RuntimeError(
                "api.use_api=true selected the platform-backed memory backend, "
                "but gigaevo.memory_platform could not be imported."
            ) from exc
        return platform_backend

    try:
        from .shared_memory.memory import AmemGamMemory as legacy_backend
    except ImportError:  # pragma: no cover - direct script execution fallback
        from shared_memory.memory import AmemGamMemory as legacy_backend
    return legacy_backend


class CardMemory(Protocol):
    def get_card(self, card_id: str) -> dict[str, Any] | None: ...


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Cards file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _latest_snapshot(payload: Any, required_key: str) -> dict[str, Any]:
    if isinstance(payload, dict):
        if required_key in payload:
            return payload
        raise ValueError(f"Missing key '{required_key}' in snapshot payload")

    if isinstance(payload, list):
        snapshots = [item for item in payload if isinstance(item, dict) and required_key in item]
        if snapshots:
            return snapshots[-1]
        raise ValueError(f"No snapshot with key '{required_key}' found in payload list")

    raise ValueError("Invalid snapshot JSON format. Expected a dict or list of dict snapshots")


def _to_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _top_percent_count(total: int, percent: float) -> int:
    if total <= 0 or percent <= 0:
        return 0
    return max(1, ceil(total * (percent / 100.0)))


def _card_type(card: dict[str, Any]) -> str:
    if str(card.get("category") or "").strip().lower() == "program":
        return "programs"
    if str(card.get("program_id") or "").strip():
        return "programs"
    return "ideas"


def _zero_write_stats() -> dict[str, int]:
    return {
        "processed": 0,
        "added": 0,
        "updated": 0,
        "rejected": 0,
        "updated_target_cards": 0,
    }


def _diff_write_stats(
    before: dict[str, int],
    after: dict[str, int],
) -> dict[str, int]:
    keys = set(before) | set(after)
    return {
        key: max(0, int(after.get(key, 0)) - int(before.get(key, 0)))
        for key in keys
    }


def _extract_usage_task_deltas(usage: Any) -> dict[str, list[float]]:
    if not isinstance(usage, dict):
        return {}
    used = usage.get("used")
    if not isinstance(used, dict):
        return {}
    entries = used.get("entries")
    if not isinstance(entries, list):
        return {}

    task_to_deltas: dict[str, list[float]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        task_summary = str(entry.get("task_description_summary") or "").strip()
        if not task_summary:
            continue
        raw_deltas = entry.get("fitness_delta_per_use")
        if raw_deltas is None:
            raw_deltas = entry.get("fitness_deltas")
        if not isinstance(raw_deltas, list):
            continue

        deltas: list[float] = []
        for raw_delta in raw_deltas:
            parsed = _to_float(raw_delta)
            if parsed is not None:
                deltas.append(parsed)
        if not deltas:
            continue
        task_to_deltas.setdefault(task_summary, []).extend(deltas)
    return task_to_deltas


def _build_usage_payload_from_task_deltas(task_to_deltas: dict[str, list[float]]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    total_deltas: list[float] = []

    for task_summary in sorted(task_to_deltas):
        deltas = [
            parsed
            for raw in task_to_deltas.get(task_summary, [])
            if (parsed := _to_float(raw)) is not None
        ]
        if not deltas:
            continue
        entries.append(
            {
                "task_description_summary": task_summary,
                "used_count": len(deltas),
                "fitness_delta_per_use": deltas,
                "median_delta_fitness": _median_or_none(deltas),
            }
        )
        total_deltas.extend(deltas)

    return {
        "used": {
            "entries": entries,
            "total": {
                "total_used": len(total_deltas),
                "median_delta_fitness": _median_or_none(total_deltas),
            },
        }
    }


def _merge_usage_payloads(existing_usage: Any, incoming_usage: Any) -> dict[str, Any]:
    existing_task_deltas = _extract_usage_task_deltas(existing_usage)
    incoming_task_deltas = _extract_usage_task_deltas(incoming_usage)
    if not existing_task_deltas and not incoming_task_deltas:
        if isinstance(existing_usage, dict):
            return dict(existing_usage)
        if isinstance(incoming_usage, dict):
            return dict(incoming_usage)
        return {}

    merged_task_deltas: dict[str, list[float]] = {
        task: list(deltas) for task, deltas in existing_task_deltas.items()
    }
    for task_summary, deltas in incoming_task_deltas.items():
        merged_task_deltas.setdefault(task_summary, []).extend(deltas)

    merged_usage: dict[str, Any] = (
        dict(existing_usage) if isinstance(existing_usage, dict) else {}
    )
    if isinstance(incoming_usage, dict):
        for key, value in incoming_usage.items():
            if key != "used":
                merged_usage[key] = value
    merged_usage["used"] = _build_usage_payload_from_task_deltas(merged_task_deltas)["used"]
    return merged_usage


def _load_usage_updates(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}

    payload = _load_json(path)
    if isinstance(payload, list):
        snapshots = [
            item
            for item in payload
            if isinstance(item, dict) and isinstance(item.get("usage_updates"), dict)
        ]
        if snapshots:
            payload = snapshots[-1]["usage_updates"]
        else:
            return {}
    elif isinstance(payload, dict) and "usage_updates" in payload:
        payload = payload.get("usage_updates")

    if not isinstance(payload, dict):
        return {}

    updates: dict[str, dict[str, Any]] = {}
    for raw_card_id, usage_update in payload.items():
        card_id = str(raw_card_id or "").strip()
        if not card_id or not isinstance(usage_update, dict):
            continue
        updates[card_id] = usage_update
    return updates


def _parse_best_ideas(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    payload = _load_json(path)
    snapshot = _latest_snapshot(payload, "best_ideas")
    best_ideas = snapshot.get("best_ideas")
    if not isinstance(best_ideas, list):
        raise ValueError(f"Invalid best ideas format in {path}: expected list under 'best_ideas'")

    idea_ids: list[str] = []
    best_by_id: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    for item in best_ideas:
        if not isinstance(item, dict):
            continue
        idea_id = str(item.get("idea_id") or item.get("id") or "").strip()
        if not idea_id or idea_id in seen_ids:
            continue
        seen_ids.add(idea_id)
        idea_ids.append(idea_id)
        best_by_id[idea_id] = item

    return idea_ids, best_by_id


def _parse_programs(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    snapshot = _latest_snapshot(payload, "programs")
    programs = snapshot.get("programs")
    if not isinstance(programs, list):
        raise ValueError(f"Invalid programs format in {path}: expected list under 'programs'")
    return [program for program in programs if isinstance(program, dict)]


def _merge_best_idea_metrics(card: dict[str, Any], best_entry: dict[str, Any]) -> dict[str, Any]:
    merged = dict(card)
    if not merged.get("description"):
        merged["description"] = str(best_entry.get("description") or "")

    best_metrics = {
        key: value
        for key, value in best_entry.items()
        if key not in {"idea_id", "id", "description"}
    }
    if best_metrics:
        evolution_stats = merged.get("evolution_statistics")
        if not isinstance(evolution_stats, dict):
            evolution_stats = {}
        evolution_stats["best_ideas_snapshot"] = best_metrics
        merged["evolution_statistics"] = evolution_stats

    return merged


def _load_banks_cards(path: Path, best_ideas_path: Path) -> list[dict]:
    if not best_ideas_path.exists():
        raise FileNotFoundError(f"Best ideas file not found: {best_ideas_path}")

    payload = _load_json(path)
    snapshot = _latest_snapshot(payload, "active_bank")
    active_bank = snapshot.get("active_bank")
    if not isinstance(active_bank, list):
        raise ValueError(
            f"Invalid banks format in {path}: expected 'active_bank' list"
        )

    all_cards = [card for card in active_bank if isinstance(card, dict)]
    cards_by_id = {
        str(card.get("id")).strip(): card
        for card in all_cards
        if str(card.get("id") or "").strip()
    }
    best_idea_ids, best_by_id = _parse_best_ideas(best_ideas_path)

    selected_cards: list[dict] = []
    missing_cards: list[str] = []
    for idea_id in best_idea_ids:
        bank_card = cards_by_id.get(idea_id)
        best_entry = best_by_id.get(idea_id, {})
        if bank_card is None:
            missing_cards.append(idea_id)
            bank_card = {"id": idea_id}
        selected_cards.append(_merge_best_idea_metrics(bank_card, best_entry))

    if missing_cards:
        print(
            f"Warning: {len(missing_cards)} best_ideas IDs were missing in banks and "
            f"were written as minimal cards."
        )

    return selected_cards


def _load_latest_bank_cards(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    snapshot = _latest_snapshot(payload, "active_bank")
    active_bank = snapshot.get("active_bank")
    if not isinstance(active_bank, list):
        raise ValueError(
            f"Invalid banks format in {path}: expected 'active_bank' list"
        )
    return [card for card in active_bank if isinstance(card, dict)]


def _build_program_cards(
    *,
    programs_path: Path | None,
    banks_path: Path,
    best_programs_percent: float,
) -> list[dict[str, Any]]:
    if programs_path is None or not programs_path.exists() or best_programs_percent <= 0:
        return []

    programs = _parse_programs(programs_path)
    if not programs:
        return []

    eligible_programs: list[dict[str, Any]] = []
    for program in programs:
        program_id = str(program.get("id") or program.get("program_id") or "").strip()
        fitness = _to_float(program.get("fitness"))
        if not program_id or fitness is None:
            continue
        enriched = dict(program)
        enriched["program_id"] = program_id
        enriched["fitness"] = fitness
        eligible_programs.append(enriched)

    if not eligible_programs:
        return []

    eligible_programs.sort(
        key=lambda program: (float(program.get("fitness", 0.0)), str(program["program_id"])),
        reverse=True,
    )
    selected_count = _top_percent_count(len(eligible_programs), best_programs_percent)
    selected_programs = eligible_programs[:selected_count]

    connected_ideas_by_program: dict[str, list[dict[str, str]]] = {}
    for idea_card in _load_latest_bank_cards(banks_path):
        idea_id = str(idea_card.get("id") or "").strip()
        idea_description = str(idea_card.get("description") or "").strip()
        if not idea_id:
            continue
        programs_for_idea = idea_card.get("programs")
        if not isinstance(programs_for_idea, list):
            continue
        linked_idea = {
            "idea_id": idea_id,
            "description": idea_description,
        }
        for raw_program_id in programs_for_idea:
            linked_program_id = str(raw_program_id or "").strip()
            if not linked_program_id:
                continue
            connected_ideas_by_program.setdefault(linked_program_id, []).append(linked_idea)

    cards: list[dict[str, Any]] = []
    total_programs = len(eligible_programs)
    for rank, program in enumerate(selected_programs, start=1):
        program_id = str(program["program_id"])
        task_description = str(program.get("task_description") or "").strip()
        task_description_summary = str(program.get("task_description_summary") or "").strip()
        connected_ideas = connected_ideas_by_program.get(program_id, [])
        connected_descriptions = [
            str(item.get("description") or "").strip()
            for item in connected_ideas
            if isinstance(item, dict)
        ]
        connected_descriptions = [text for text in connected_descriptions if text]
        connected_summary = "; ".join(connected_descriptions[:5])
        description_seed = task_description_summary or task_description or "task summary unavailable"
        description = (
            f"Top evolved program for {description_seed} "
            f"(fitness={float(program['fitness']):.6g}, rank={rank}/{total_programs})."
        )
        if connected_summary:
            description += f" Connected ideas: {connected_summary}"

        cards.append(
            {
                "id": f"program-{program_id}",
                "category": "program",
                "program_id": program_id,
                "task_description": task_description,
                "task_description_summary": task_description_summary,
                "description": description,
                "fitness": float(program["fitness"]),
                "code": str(program.get("code") or ""),
                "connected_ideas": connected_ideas,
            }
        )

    return cards


def _apply_usage_updates_to_cards(
    cards: list[dict[str, Any]],
    *,
    usage_updates: dict[str, dict[str, Any]],
    memory: CardMemory,
) -> list[dict[str, Any]]:
    if not usage_updates:
        return cards

    cards_by_id: dict[str, dict[str, Any]] = {}
    for card in cards:
        card_id = str(card.get("id") or "").strip()
        if card_id:
            cards_by_id[card_id] = dict(card)

    missing_card_ids: list[str] = []
    for card_id, usage_update in usage_updates.items():
        current_card = cards_by_id.get(card_id)
        if current_card is None:
            existing = memory.get_card(card_id)
            if not isinstance(existing, dict):
                missing_card_ids.append(card_id)
                continue
            current_card = dict(existing)

        current_card["usage"] = _merge_usage_payloads(
            current_card.get("usage"),
            usage_update,
        )
        cards_by_id[card_id] = current_card

    if missing_card_ids:
        print(
            "Warning: skipped usage updates for "
            f"{len(missing_card_ids)} card(s) because they were not found in memory store."
        )

    return list(cards_by_id.values())


def load_memory_cards(
    path: Path,
    best_ideas_path: Path,
    *,
    programs_path: Path | None = None,
    best_programs_percent: float = 0.0,
    usage_updates_path: Path | None = None,
    memory: CardMemory | None = None,
) -> list[dict]:
    payload = _load_json(path)
    usage_updates = _load_usage_updates(usage_updates_path)

    cards: list[dict]
    if isinstance(payload, dict) and "active_bank" in payload:
        cards = _load_banks_cards(path, best_ideas_path)
    elif (
        isinstance(payload, list)
        and payload
        and isinstance(payload[0], dict)
        and "active_bank" in payload[0]
    ):
        cards = _load_banks_cards(path, best_ideas_path)
    else:
        raise ValueError(
            "Invalid banks JSON format. Expected payload with 'active_bank'"
        )

    if usage_updates and memory is not None:
        cards = _apply_usage_updates_to_cards(
            cards,
            usage_updates=usage_updates,
            memory=memory,
        )

    return cards + _build_program_cards(
        programs_path=programs_path,
        banks_path=path,
        best_programs_percent=best_programs_percent,
    )


def _write_memory_write_stats(
    *,
    stats_path: Path,
    input_cards_count: int,
    input_card_type_counts: dict[str, int],
    write_stats: dict[str, int],
    write_stats_by_card_type: dict[str, dict[str, int]],
) -> dict[str, Any]:
    snapshot = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "input_cards_count": int(input_cards_count),
        "input_card_type_counts": input_card_type_counts,
        "stats": write_stats,
        "stats_by_card_type": write_stats_by_card_type,
    }

    existing: list[dict[str, Any]] = []
    if stats_path.exists():
        try:
            raw = _load_json(stats_path)
            if isinstance(raw, list):
                existing = [item for item in raw if isinstance(item, dict)]
            elif isinstance(raw, dict):
                existing = [raw]
        except Exception:
            existing = []

    existing.append(snapshot)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=True, indent=2)
    return snapshot


def main() -> dict[str, Any] | None:
    memory_backend_cls = _resolve_memory_backend_class(USE_API)
    memory = memory_backend_cls(
        checkpoint_path=str(MEMORY_DIR),
        base_url=MEMORY_API_URL,
        use_api=USE_API,
        namespace=NAMESPACE,
        channel=CHANNEL,
        author=AUTHOR,
        search_limit=SEARCH_LIMIT,
        enable_llm_synthesis=ENABLE_LLM_SYNTHESIS,
        enable_memory_evolution=SHOULD_EVOLVE,
        enable_llm_card_enrichment=FILL_MISSING_FIELDS_WITH_LLM,
        rebuild_interval=REBUILD_INTERVAL,
        enable_bm25=ENABLE_BM25,
        sync_batch_size=SYNC_BATCH_SIZE,
        sync_on_init=SYNC_ON_INIT,
        allowed_gam_tools=ALLOWED_GAM_TOOLS,
        gam_top_k_by_tool=GAM_TOP_K_BY_TOOL,
        gam_pipeline_mode=GAM_PIPELINE_MODE,
        card_update_dedup_config=CARD_UPDATE_DEDUP_CONFIG,
    )

    print("\n==============================")
    print("API Memory Demo: Card Write")
    print("==============================\n")
    print(f"Config file: {SETTINGS_PATH}")
    print(f"Memory evolution enabled: {SHOULD_EVOLVE}")
    print(f"LLM field fill enabled: {FILL_MISSING_FIELDS_WITH_LLM}")
    print(f"Memory usage tracking enabled: {ENABLE_USAGE_TRACKING}")
    print(
        "Card update/dedup enabled: "
        f"{to_bool(CARD_UPDATE_DEDUP_CONFIG.get('enabled'), default=False)}"
    )

    if not BANKS_PATH.exists():
        raise FileNotFoundError(f"Banks file not found: {BANKS_PATH}")
    memory_cards = load_memory_cards(
        BANKS_PATH,
        best_ideas_path=BEST_IDEAS_PATH,
        programs_path=PROGRAMS_PATH,
        best_programs_percent=BEST_PROGRAMS_PERCENT,
        usage_updates_path=USAGE_UPDATES_PATH,
        memory=memory,
    )
    print(
        f"Loaded {len(memory_cards)} cards from banks: {BANKS_PATH} "
        f"(filtered by: {BEST_IDEAS_PATH})"
    )
    if USE_API:
        print(f"Writing to API: {MEMORY_API_URL} (namespace={NAMESPACE})\n")
    else:
        print(f"Writing in local-only mode (checkpoint={MEMORY_DIR})\n")

    try:
        write_stats_by_card_type = {
            "ideas": _zero_write_stats(),
            "programs": _zero_write_stats(),
        }
        for idx, card in enumerate(memory_cards, start=1):
            card_type = _card_type(card)
            before_stats = memory.get_card_write_stats()
            memory_id = memory.save_card(card)
            after_stats = memory.get_card_write_stats()
            stat_diff = _diff_write_stats(before_stats, after_stats)
            for stat_name, stat_value in stat_diff.items():
                write_stats_by_card_type[card_type][stat_name] = (
                    int(write_stats_by_card_type[card_type].get(stat_name, 0)) + int(stat_value)
                )
            stored = memory.get_card(memory_id) or {}
            print(f"[{idx:03d}] saved {memory_id}: {stored.get('description', '')[:110]}")
    except RuntimeError as exc:
        print(f"\nWrite failed: {exc}\n")
        return None

    memory.rebuild()
    print(f"\nLocal API index saved in: {MEMORY_DIR / 'api_index.json'}")

    write_stats = memory.get_card_write_stats()
    input_card_type_counts = {
        "ideas": sum(1 for card in memory_cards if _card_type(card) == "ideas"),
        "programs": sum(1 for card in memory_cards if _card_type(card) == "programs"),
    }
    print(
        "Write stats: "
        f"processed={write_stats.get('processed', 0)}, "
        f"ideas_processed={write_stats_by_card_type['ideas'].get('processed', 0)}, "
        f"programs_processed={write_stats_by_card_type['programs'].get('processed', 0)}, "
        f"added={write_stats.get('added', 0)}, "
        f"ideas_added={write_stats_by_card_type['ideas'].get('added', 0)}, "
        f"programs_added={write_stats_by_card_type['programs'].get('added', 0)}, "
        f"updated={write_stats.get('updated', 0)}, "
        f"ideas_updated={write_stats_by_card_type['ideas'].get('updated', 0)}, "
        f"programs_updated={write_stats_by_card_type['programs'].get('updated', 0)}, "
        f"rejected={write_stats.get('rejected', 0)}, "
        f"ideas_rejected={write_stats_by_card_type['ideas'].get('rejected', 0)}, "
        f"programs_rejected={write_stats_by_card_type['programs'].get('rejected', 0)}, "
        f"updated_target_cards={write_stats.get('updated_target_cards', 0)}"
    )

    stats_path = BANKS_PATH.parent / "memory_write_stats.json"
    snapshot = _write_memory_write_stats(
        stats_path=stats_path,
        input_cards_count=len(memory_cards),
        input_card_type_counts=input_card_type_counts,
        write_stats=write_stats,
        write_stats_by_card_type=write_stats_by_card_type,
    )
    print(f"Memory write stats saved to: {stats_path}")
    return snapshot


if __name__ == "__main__":
    main()
