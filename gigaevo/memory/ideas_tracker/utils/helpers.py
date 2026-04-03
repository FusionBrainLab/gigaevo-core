from __future__ import annotations

import ast
import json
import math
import statistics
from typing import Any

import pandas as pd


def to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def to_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def as_string_list(value: Any) -> list[str]:
    parsed: Any
    if isinstance(value, list):
        parsed = value
    elif isinstance(value, tuple):
        parsed = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        parsed = text
        if text[0] in "[{(":
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                try:
                    parsed = ast.literal_eval(text)
                except Exception:
                    parsed = text
        if not isinstance(parsed, list):
            return [str(parsed).strip()] if str(parsed).strip() else []
    else:
        return []

    out: list[str] = []
    for item in parsed:
        text = str(item or "").strip()
        if text:
            out.append(text)
    return out


def parse_json_like(value: Any) -> Any:
    """Parse JSON-ish strings from CSV back to Python objects when possible."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except Exception:
            return value


def coerce_bool_series(series: pd.Series) -> pd.Series:
    """Coerce CSV-backed truthy/falsy values into a bool series."""
    if series.dtype == bool:
        return series
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map({"true": True, "false": False, "1": True, "0": False}).fillna(
        False
    )


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Redis/CSV data into the shape expected by IdeaTracker."""
    result = df.copy()

    if "is_root" in result.columns:
        result["is_root"] = coerce_bool_series(result["is_root"])

    for col in ("parent_ids", "children_ids", "metadata_mutation_output"):
        if col in result.columns:
            result[col] = result[col].apply(parse_json_like)

    return result


def median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def extract_usage_task_deltas(usage: Any) -> dict[str, list[float]]:
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

        parsed_deltas: list[float] = []
        for raw_delta in raw_deltas:
            delta = to_float(raw_delta)
            if delta is not None:
                parsed_deltas.append(delta)
        if not parsed_deltas:
            continue
        task_to_deltas.setdefault(task_summary, []).extend(parsed_deltas)
    return task_to_deltas


def build_usage_payload_from_task_deltas(
    task_to_deltas: dict[str, list[float]],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    total_deltas: list[float] = []

    for task_summary in sorted(task_to_deltas):
        deltas = [
            parsed
            for raw in task_to_deltas.get(task_summary, [])
            if (parsed := to_float(raw)) is not None
        ]
        if not deltas:
            continue
        entries.append(
            {
                "task_description_summary": task_summary,
                "used_count": len(deltas),
                "fitness_delta_per_use": deltas,
                "median_delta_fitness": median_or_none(deltas),
            }
        )
        total_deltas.extend(deltas)

    return {
        "used": {
            "entries": entries,
            "total": {
                "total_used": len(total_deltas),
                "median_delta_fitness": median_or_none(total_deltas),
            },
        }
    }


def merge_usage_payloads(existing_usage: Any, incoming_usage: Any) -> dict[str, Any]:
    existing_task_deltas = extract_usage_task_deltas(existing_usage)
    incoming_task_deltas = extract_usage_task_deltas(incoming_usage)
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
    merged_usage["used"] = build_usage_payload_from_task_deltas(merged_task_deltas)[
        "used"
    ]
    return merged_usage


def build_memory_usage_updates(
    programs_df: pd.DataFrame, task_summary: str = "Task summary unavailable"
) -> dict[str, dict[str, Any]]:
    required_columns = {
        "program_id",
        "metric_fitness",
        "parent_ids",
        "metadata_memory_selected_idea_ids",
    }
    if not required_columns.issubset(programs_df.columns):
        return {}

    fitness_by_program_id: dict[str, float] = {}
    for _, row in programs_df.iterrows():
        program_id = str(row.get("program_id") or "").strip()
        if not program_id:
            continue
        fitness = to_float(row.get("metric_fitness"))
        if fitness is not None:
            fitness_by_program_id[program_id] = fitness

    usage_by_card: dict[str, dict[str, list[float]]] = {}
    for _, row in programs_df.iterrows():
        selected_ids = as_string_list(row.get("metadata_memory_selected_idea_ids"))
        if not selected_ids:
            continue

        child_fitness = to_float(row.get("metric_fitness"))
        if child_fitness is None:
            continue

        parent_ids = as_string_list(row.get("parent_ids"))
        parent_fitnesses = [
            fitness_by_program_id[parent_id]
            for parent_id in parent_ids
            if parent_id in fitness_by_program_id
        ]
        if not parent_fitnesses:
            continue

        delta_fitness = child_fitness - max(parent_fitnesses)
        unique_selected_ids = list(dict.fromkeys(selected_ids))
        for card_id in unique_selected_ids:
            per_task = usage_by_card.setdefault(card_id, {})
            per_task.setdefault(task_summary, []).append(delta_fitness)

    return {
        card_id: build_usage_payload_from_task_deltas(task_deltas)
        for card_id, task_deltas in usage_by_card.items()
    }


def sort_ideas(ideas: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    sorted_ideas: dict[str, list[dict[str, Any]]] = {
        "new": [],
        "update": [],
        "rewrite": [],
    }
    for idea in ideas:
        if idea["rewrite"] and idea["classified"]:
            sorted_ideas["rewrite"].append(idea)
        elif idea["classified"] and not idea["rewrite"]:
            sorted_ideas["update"].append(idea)
        elif not idea["classified"]:
            sorted_ideas["new"].append(idea)
    return sorted_ideas
