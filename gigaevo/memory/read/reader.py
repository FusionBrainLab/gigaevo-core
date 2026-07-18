"""Read-selection value type + policy-version digest helper.

``MemorySelection`` is the read side's result object — rendered cards plus the
durable assignment ledger payload — consumed by ``MemoryContextStage`` and the
v2 provider. ``extend_policy_version`` folds the exclusion policy and downstream
delivery into a read policy-version digest.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gigaevo.memory.cards import AssignmentRecord


class MemorySelection(BaseModel):
    """Result of memory card selection for mutation guidance."""

    model_config = ConfigDict(frozen=True)

    cards: tuple[str, ...] = Field(
        default=(),
        description="Rendered mutator-facing text blocks, one per selected card.",
    )
    card_ids: tuple[str, ...] = Field(
        default=(),
        description="Bank ids of the selected cards, aligned with ``cards``.",
    )
    decision_id: str = Field(
        default="", description="Correlation id of the read decision."
    )
    assignment: AssignmentRecord | None = Field(
        default=None, description="Durable assignment ledger payload."
    )
    preformatted: bool = Field(
        default=False,
        description="Cards already contain their complete mutator-facing wrapper.",
    )


_POLICY_VOLATILE_NAMES = frozenset(
    {
        "bank",
        "cache",
        "events",
        "index",
        "lock",
        "rng",
        "semaphore",
        "state",
        "tracker",
    }
)
_POLICY_NESTED_NAMES = frozenset(
    {
        "agent",
        "config",
        "context_model",
        "inner",
        "llm",
        "policy_identifiers",
        "reputation",
        "store",
    }
)


def _policy_class(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _canonical_policy_value(value: Any, seen: set[int] | None = None) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.name
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_policy_value(item, seen)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_canonical_policy_value(item, seen) for item in value]
        return (
            sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
            if isinstance(value, (set, frozenset))
            else items
        )

    seen = set() if seen is None else seen
    if id(value) in seen:
        return {"class": _policy_class(value)}
    seen.add(id(value))
    try:
        if isinstance(value, BaseModel):
            fields = type(value).model_fields
            model_config = {
                name: _canonical_policy_value(getattr(value, name), seen)
                for name in sorted(fields)
                if name not in {"path", "cache_dir"}
            }
            return {"class": _policy_class(value), "config": model_config}

        config: dict[str, Any] = {}
        for raw_name, item in sorted(vars(value).items()):
            name = raw_name.lstrip("_")
            if any(token in name for token in _POLICY_VOLATILE_NAMES):
                continue
            if name in _POLICY_NESTED_NAMES or isinstance(
                item, (bool, int, float, str, tuple, list, dict, BaseModel)
            ):
                config[name] = _canonical_policy_value(item, seen)
        return {"class": _policy_class(value), "config": config}
    finally:
        seen.remove(id(value))


def _policy_component(component: Any) -> dict[str, Any]:
    canonical = _canonical_policy_value(component)
    return canonical if isinstance(canonical, dict) else {"value": canonical}


def extend_policy_version(
    policy_version: str,
    *,
    exclusion_policy: Any = None,
    downstream_delivery: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "read_policy_version": policy_version,
        "exclusion_policy": _policy_component(exclusion_policy),
        "downstream_delivery": _canonical_policy_value(downstream_delivery or {}),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"MemoryPolicy:{sha256(encoded).hexdigest()[:16]}"
