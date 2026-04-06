from __future__ import annotations

import json
import math
import re
import statistics
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gigaevo.memory.shared_memory.utils import dedupe_keep_order

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
    """Per-query weights for multi-signal dedup retrieval scoring.

    Each weight corresponds to a query type used during candidate retrieval.
    Weights should sum to approximately 1.0.  Non-dict input or non-numeric
    values are silently dropped so Pydantic uses field defaults.
    """

    model_config = ConfigDict(frozen=True)

    description: float = 0.35
    explanation_summary: float = 0.2
    description_explanation_summary: float = 0.3
    description_task_description_summary: float = 0.15

    @model_validator(mode="before")
    @classmethod
    def _normalize_input(cls, data: Any) -> dict[str, Any]:
        """Accept non-dict input gracefully; coerce values to float."""
        if not isinstance(data, dict):
            return {}
        result: dict[str, Any] = {}
        for key, val in data.items():
            try:
                result[key] = float(val)
            except (TypeError, ValueError):
                pass  # omit → Pydantic uses field default
        return result

    def as_score_multipliers(self) -> dict[str, float]:
        """Return weights keyed by query-type constant for scoring."""
        return {
            QUERY_DESCRIPTION: self.description,
            QUERY_EXPLANATION_SUMMARY: self.explanation_summary,
            QUERY_DESCRIPTION_EXPLANATION_SUMMARY: self.description_explanation_summary,
            QUERY_DESCRIPTION_TASK_DESCRIPTION_SUMMARY: self.description_task_description_summary,
        }


_TRUTHY_STRINGS = frozenset({"1", "true", "yes", "on"})


class CardUpdateDedupConfig(BaseModel):
    """Configuration for the card-level deduplication pipeline.

    Accepts either a nested dict (as stored in YAML/env configs)::

        {"enabled": True, "retrieval": {"top_k_per_query": 10}, "llm": {"max_retries": 3}}

    or the flat format matching field names::

        {"enabled": True, "top_k_per_query": 10, "llm_max_retries": 3}

    Non-dict input returns all defaults.  Integer fields are clamped to >= 1,
    float fields to >= 0.0.  The ``enabled`` field accepts truthy strings
    (``"true"``, ``"1"``, ``"yes"``, ``"on"``).
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    top_k_per_query: int = 5
    final_top_n: int = 5
    min_final_score: float = 0.0
    llm_max_retries: int = 2
    weights: RetrievalWeights = Field(default_factory=RetrievalWeights)

    @model_validator(mode="before")
    @classmethod
    def _normalize_input(cls, data: Any) -> dict[str, Any]:
        """Flatten nested retrieval/llm sub-dicts and coerce types."""
        if not isinstance(data, dict):
            return {}

        result: dict[str, Any] = {}

        # enabled: bool or truthy string
        enabled_raw = data.get("enabled")
        if isinstance(enabled_raw, bool):
            result["enabled"] = enabled_raw
        elif enabled_raw is not None:
            result["enabled"] = str(enabled_raw).strip().lower() in _TRUTHY_STRINGS

        retrieval = data.get("retrieval", {})
        if not isinstance(retrieval, dict):
            retrieval = {}
        llm = data.get("llm", {})
        if not isinstance(llm, dict):
            llm = {}

        # Int fields: prefer nested source, fall back to flat key
        _int_fields = {
            "top_k_per_query": (retrieval, "top_k_per_query", 1),
            "final_top_n": (retrieval, "final_top_n", 1),
            "llm_max_retries": (llm, "max_retries", 1),
        }
        for target, (source, key, min_val) in _int_fields.items():
            raw = source.get(key)
            if raw is None:
                raw = data.get(target)
            if raw is not None:
                try:
                    result[target] = max(min_val, int(raw))
                except (TypeError, ValueError):
                    pass  # omit → Pydantic uses field default

        # Float field with min-0 clamping
        raw_score = retrieval.get("min_final_score")
        if raw_score is None:
            raw_score = data.get("min_final_score")
        if raw_score is not None:
            try:
                result["min_final_score"] = max(0.0, float(raw_score))
            except (TypeError, ValueError):
                pass

        # Nested weights (Pydantic + RetrievalWeights validator handles coercion)
        raw_weights = retrieval.get("weights")
        if raw_weights is None:
            raw_weights = data.get("weights")
        if raw_weights is not None:
            result["weights"] = raw_weights

        return result


def _normalize_text(value: Any) -> str:
    """Collapse whitespace and strip a value coerced to string."""
    return " ".join(str(value or "").split()).strip()


def _safe_string_list(value: Any) -> list[str]:
    """Coerce *value* to a list of non-empty stripped strings."""
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
    """Extract the best single-line explanation summary from a card dict.

    Checks ``explanation.summary``, then the last entry in
    ``explanation.explanations``, then ``explanation`` as a bare string.
    """
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
    """Return all explanation strings from a card dict.

    Prefers the ``explanations`` list; falls back to the single summary.
    """
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
    """Combine two text fragments with labels, skipping empty parts."""
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
    """Build the four query strings used for multi-signal dedup retrieval."""
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
    """Rank candidate card IDs by weighted multi-query score.

    Returns up to *final_top_n* candidates sorted by descending
    weighted score, each above *min_final_score*.
    """
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
    """Extract the first JSON object from *raw_text*, stripping markdown fences."""
    text = str(raw_text or "").strip()
    if not text:
        return None

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        parsed = json.loads(text)
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
    """Parse an LLM-generated JSON decision into a validated action dict.

    Returns ``None`` if the text contains no parseable JSON.  Otherwise
    normalises the ``action`` to one of ``add`` / ``discard`` / ``update``
    and validates ``duplicate_of`` and ``updates`` against *candidate_ids*.
    """
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


def append_unique_text(
    original_text: str,
    added_text: str,
    *,
    separator: str = "\n\n---\n\n",
) -> str:
    """Append *added_text* to *original_text* unless it is already contained."""
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
    """Convert *value* to float, returning ``None`` for NaN/Inf/unparseable."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _median_or_none(values: list[float]) -> float | None:
    """Return the median of *values*, or ``None`` if empty."""
    if not values:
        return None
    return float(statistics.median(values))


def _extract_usage_task_deltas(usage: Any) -> dict[str, list[float]]:
    """Extract per-task fitness deltas from a usage payload dict."""
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
    """Build a canonical ``{"used": {...}}`` payload from per-task deltas."""
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
    """Merge two usage payloads, combining per-task fitness deltas."""
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
    """Merge *incoming_card* into *existing_card* using an LLM *update* spec.

    Handles programs dedup, generation bumping, task description / explanation
    appending, and usage payload merging.
    """
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
