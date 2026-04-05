from __future__ import annotations

import json
import math
import re
import statistics
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

QUERY_DESCRIPTION = "description"
QUERY_EXPLANATION_SUMMARY = "explanation_summary"
QUERY_DESCRIPTION_EXPLANATION_SUMMARY = "description_explanation_summary"
QUERY_DESCRIPTION_TASK_DESCRIPTION_SUMMARY = "description_task_description_summary"

QUERY_KEYS = (
    QUERY_DESCRIPTION,
    QUERY_EXPLANATION_SUMMARY,
    QUERY_DESCRIPTION_EXPLANATION_SUMMARY,
    QUERY_DESCRIPTION_TASK_DESCRIPTION_SUMMARY,
)


class RetrievalWeights(BaseModel):
    model_config = ConfigDict(frozen=True)

    description: float = 0.35
    explanation_summary: float = 0.2
    description_explanation_summary: float = 0.3
    description_task_description_summary: float = 0.15

    @classmethod
    def from_mapping(cls, payload: Any) -> RetrievalWeights:
        if not isinstance(payload, dict):
            return cls()

        def _to_float(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        return cls(
            description=_to_float(payload.get("description"), cls().description),
            explanation_summary=_to_float(
                payload.get("explanation_summary"),
                cls().explanation_summary,
            ),
            description_explanation_summary=_to_float(
                payload.get("description_explanation_summary"),
                cls().description_explanation_summary,
            ),
            description_task_description_summary=_to_float(
                payload.get("description_task_description_summary"),
                cls().description_task_description_summary,
            ),
        )

    def as_score_multipliers(self) -> dict[str, float]:
        return {
            QUERY_DESCRIPTION: self.description,
            QUERY_EXPLANATION_SUMMARY: self.explanation_summary,
            QUERY_DESCRIPTION_EXPLANATION_SUMMARY: self.description_explanation_summary,
            QUERY_DESCRIPTION_TASK_DESCRIPTION_SUMMARY: self.description_task_description_summary,
        }


class CardUpdateDedupConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    top_k_per_query: int = 5
    final_top_n: int = 5
    min_final_score: float = 0.0
    llm_max_retries: int = 2
    weights: RetrievalWeights = Field(default_factory=RetrievalWeights)

    @classmethod
    def from_mapping(cls, payload: Any) -> CardUpdateDedupConfig:
        if not isinstance(payload, dict):
            return cls()

        retrieval = payload.get("retrieval")
        if not isinstance(retrieval, dict):
            retrieval = {}

        llm = payload.get("llm")
        if not isinstance(llm, dict):
            llm = {}

        def _to_int(value: Any, default: int, min_value: int = 1) -> int:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                parsed = default
            return max(min_value, parsed)

        def _to_float(value: Any, default: float, min_value: float = 0.0) -> float:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                parsed = default
            return max(min_value, parsed)

        enabled_raw = payload.get("enabled")
        enabled = False
        if isinstance(enabled_raw, bool):
            enabled = enabled_raw
        elif enabled_raw is not None:
            enabled = str(enabled_raw).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }

        return cls(
            enabled=enabled,
            top_k_per_query=_to_int(
                retrieval.get("top_k_per_query"),
                default=cls().top_k_per_query,
                min_value=1,
            ),
            final_top_n=_to_int(
                retrieval.get("final_top_n"),
                default=cls().final_top_n,
                min_value=1,
            ),
            min_final_score=_to_float(
                retrieval.get("min_final_score"),
                default=cls().min_final_score,
                min_value=0.0,
            ),
            llm_max_retries=_to_int(
                llm.get("max_retries"),
                default=cls().llm_max_retries,
                min_value=1,
            ),
            weights=RetrievalWeights.from_mapping(retrieval.get("weights")),
        )


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _safe_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []


def get_explanation_summary(card: dict[str, Any]) -> str:
    explanation = card.get("explanation")
    if isinstance(explanation, dict):
        summary = str(explanation.get("summary") or "").strip()
        if summary:
            return summary
        explanations = _safe_string_list(explanation.get("explanations"))
        if explanations:
            return explanations[-1]
    elif isinstance(explanation, str):
        text = explanation.strip()
        if text:
            return text
    return ""


def get_full_explanations(card: dict[str, Any]) -> list[str]:
    explanation = card.get("explanation")
    if isinstance(explanation, dict):
        explanations = _safe_string_list(explanation.get("explanations"))
        if explanations:
            return explanations
    summary = get_explanation_summary(card)
    return [summary] if summary else []


def _merge_labeled_text(
    first: str,
    second: str,
    *,
    first_label: str,
    second_label: str,
) -> str:
    first_text = _normalize_text(first)
    second_text = _normalize_text(second)
    if first_text and second_text:
        return f"{first_label}: {first_text}\n{second_label}: {second_text}"
    if first_text:
        return f"{first_label}: {first_text}"
    if second_text:
        return f"{second_label}: {second_text}"
    return ""


def build_dedup_queries(card: dict[str, Any]) -> dict[str, str]:
    description = _normalize_text(card.get("description"))
    explanation_summary = _normalize_text(get_explanation_summary(card))
    task_summary = _normalize_text(card.get("task_description_summary"))
    return {
        QUERY_DESCRIPTION: description,
        QUERY_EXPLANATION_SUMMARY: explanation_summary,
        QUERY_DESCRIPTION_EXPLANATION_SUMMARY: _merge_labeled_text(
            description,
            explanation_summary,
            first_label="IDEA_DESCRIPTION",
            second_label="EXPLANATION_SUMMARY",
        ),
        QUERY_DESCRIPTION_TASK_DESCRIPTION_SUMMARY: _merge_labeled_text(
            description,
            task_summary,
            first_label="IDEA_DESCRIPTION",
            second_label="TASK_DESCRIPTION_SUMMARY",
        ),
    }


def compute_weighted_candidates(
    scores_by_query: dict[str, dict[str, float]],
    *,
    weights: RetrievalWeights,
    final_top_n: int,
    min_final_score: float = 0.0,
) -> list[dict[str, Any]]:
    candidate_ids: set[str] = set()
    for scores in scores_by_query.values():
        candidate_ids.update(scores.keys())

    if not candidate_ids:
        return []

    multipliers = weights.as_score_multipliers()
    ranked: list[dict[str, Any]] = []

    for card_id in candidate_ids:
        component_scores = {
            key: float(scores_by_query.get(key, {}).get(card_id, 0.0))
            for key in QUERY_KEYS
        }
        final_score = sum(
            multipliers[key] * component_scores.get(key, 0.0) for key in QUERY_KEYS
        )
        if final_score < float(min_final_score):
            continue
        ranked.append(
            {
                "card_id": card_id,
                "final_score": float(final_score),
                "scores": component_scores,
            }
        )

    ranked.sort(key=lambda item: (item["final_score"], item["card_id"]), reverse=True)
    return ranked[: max(1, int(final_top_n))]


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    text = str(raw_text or "").strip()
    if not text:
        return None

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    for candidate in (text,):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def parse_llm_card_decision(
    raw_text: str,
    *,
    candidate_ids: set[str],
) -> dict[str, Any] | None:
    payload = _extract_json_object(raw_text)
    if payload is None:
        return None
    action = str(payload.get("action") or "add").strip().lower()
    if action not in {"add", "discard", "update"}:
        action = "add"

    duplicate_of = str(payload.get("duplicate_of") or "").strip()
    if duplicate_of and duplicate_of not in candidate_ids:
        duplicate_of = ""

    updates_raw = payload.get("updates")
    updates: list[dict[str, Any]] = []
    if isinstance(updates_raw, list):
        for item in updates_raw:
            if not isinstance(item, dict):
                continue
            card_id = str(item.get("card_id") or "").strip()
            if not card_id or card_id not in candidate_ids:
                continue
            update_task_description = bool(item.get("update_task_description"))
            update_explanation = bool(item.get("update_explanation"))
            task_description_append = str(
                item.get("task_description_append") or ""
            ).strip()
            task_description_summary = str(
                item.get("task_description_summary") or ""
            ).strip()
            explanation_append = str(item.get("explanation_append") or "").strip()
            explanation_summary = str(item.get("explanation_summary") or "").strip()
            if not (
                update_task_description
                or update_explanation
                or task_description_append
                or explanation_append
                or task_description_summary
                or explanation_summary
            ):
                continue
            updates.append(
                {
                    "card_id": card_id,
                    "update_task_description": update_task_description
                    or bool(task_description_append)
                    or bool(task_description_summary),
                    "task_description_append": task_description_append,
                    "task_description_summary": task_description_summary,
                    "update_explanation": update_explanation
                    or bool(explanation_append)
                    or bool(explanation_summary),
                    "explanation_append": explanation_append,
                    "explanation_summary": explanation_summary,
                }
            )

    reason = str(payload.get("reason") or "").strip()

    if action == "update" and not updates:
        if duplicate_of:
            action = "discard"
        else:
            action = "add"

    if action == "discard" and not duplicate_of:
        if updates:
            action = "update"
        else:
            action = "add"

    return {
        "action": action,
        "reason": reason,
        "duplicate_of": duplicate_of,
        "updates": updates,
    }


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def append_unique_text(
    original_text: str,
    added_text: str,
    *,
    separator: str = "\n\n---\n\n",
) -> str:
    left = str(original_text or "").strip()
    right = str(added_text or "").strip()
    if not right:
        return left
    if not left:
        return right
    if _normalize_text(right).lower() in _normalize_text(left).lower():
        return left
    return f"{left}{separator}{right}"


def _safe_float(value: Any) -> float | None:
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
        deltas = [
            parsed for raw in raw_deltas if (parsed := _safe_float(raw)) is not None
        ]
        if not deltas:
            continue
        task_to_deltas.setdefault(task_summary, []).extend(deltas)
    return task_to_deltas


def _build_usage_payload(task_to_deltas: dict[str, list[float]]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    total_deltas: list[float] = []
    for task_summary in sorted(task_to_deltas):
        deltas = [
            parsed
            for raw in task_to_deltas.get(task_summary, [])
            if (parsed := _safe_float(raw)) is not None
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


def merge_usage_payloads(existing_usage: Any, incoming_usage: Any) -> dict[str, Any]:
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
    merged_usage["used"] = _build_usage_payload(merged_task_deltas)["used"]
    return merged_usage


def merge_updated_card(
    existing_card: dict[str, Any],
    incoming_card: dict[str, Any],
    update: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(existing_card)
    merged["id"] = str(existing_card.get("id") or merged.get("id") or "").strip()

    existing_programs = _safe_string_list(existing_card.get("programs"))
    incoming_programs = _safe_string_list(incoming_card.get("programs"))
    merged["programs"] = dedupe_keep_order(existing_programs + incoming_programs)

    try:
        existing_gen = int(existing_card.get("last_generation") or 0)
    except (TypeError, ValueError):
        existing_gen = 0
    try:
        incoming_gen = int(incoming_card.get("last_generation") or 0)
    except (TypeError, ValueError):
        incoming_gen = 0
    merged["last_generation"] = max(existing_gen, incoming_gen)

    if bool(update.get("update_task_description")):
        incoming_task = str(
            update.get("task_description_append")
            or incoming_card.get("task_description")
            or incoming_card.get("task_description_summary")
            or ""
        ).strip()
        merged["task_description"] = append_unique_text(
            str(existing_card.get("task_description") or ""),
            incoming_task,
        )
        explicit_summary = str(update.get("task_description_summary") or "").strip()
        if explicit_summary:
            merged["task_description_summary"] = explicit_summary
        elif not str(existing_card.get("task_description_summary") or "").strip():
            merged["task_description_summary"] = str(
                incoming_card.get("task_description_summary")
                or incoming_card.get("task_description")
                or ""
            ).strip()

    existing_explanation = existing_card.get("explanation")
    if not isinstance(existing_explanation, dict):
        existing_explanation = {}
    explanation_items = _safe_string_list(existing_explanation.get("explanations"))

    if bool(update.get("update_explanation")):
        append_text = str(update.get("explanation_append") or "").strip()
        if append_text:
            explanation_items.append(append_text)
        else:
            explanation_items.extend(get_full_explanations(incoming_card))

    explanation_summary = str(existing_explanation.get("summary") or "").strip()
    explicit_explanation_summary = str(update.get("explanation_summary") or "").strip()
    if explicit_explanation_summary:
        explanation_summary = explicit_explanation_summary
    elif not explanation_summary:
        explanation_summary = get_explanation_summary(incoming_card)

    merged["explanation"] = {
        "explanations": dedupe_keep_order(explanation_items),
        "summary": explanation_summary,
    }
    merged["usage"] = merge_usage_payloads(
        existing_card.get("usage"),
        incoming_card.get("usage"),
    )
    return merged
