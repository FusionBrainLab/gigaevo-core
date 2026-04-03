from __future__ import annotations

import ast
import json
import math
import statistics
from typing import Any

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
)
from gigaevo.programs.program import Program


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


def build_memory_usage_updates_from_programs(
    programs: list[Program],
    task_summary: str = "Task summary unavailable",
    fitness_key: str = "fitness",
) -> dict[str, dict[str, Any]]:
    """Build per-card usage updates from Program objects.

    For each program that has memory_selected_idea_ids, compute
    fitness_delta = child_fitness - max(parent_fitnesses) and attribute
    that delta to each selected card.
    """
    fitness_by_id: dict[str, float] = {}
    for prog in programs:
        fitness = to_float(prog.metrics.get(fitness_key))
        if fitness is not None:
            fitness_by_id[prog.id] = fitness

    usage_by_card: dict[str, dict[str, list[float]]] = {}
    for prog in programs:
        selected_ids = as_string_list(
            prog.metadata.get(MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY)
        )
        if not selected_ids:
            continue

        child_fitness = to_float(prog.metrics.get(fitness_key))
        if child_fitness is None:
            continue

        parent_fitnesses = [
            fitness_by_id[pid] for pid in prog.lineage.parents if pid in fitness_by_id
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
