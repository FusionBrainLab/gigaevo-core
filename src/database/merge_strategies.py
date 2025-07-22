"""Merge strategies for :class:`Program` objects.

The public entry-point is :func:`resolve_merge_strategy` which returns a
callable that merges two :class:`src.programs.program.Program` instances
(*current* and *incoming*) according to the requested strategy.
"""

from __future__ import annotations

from typing import Callable, Optional, Union, get_origin

from loguru import logger

from src.programs.program import Program
from src.programs.program_state import merge_states

__all__ = ["resolve_merge_strategy"]


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------


def _additive_merge(current: Optional[Program], incoming: Program) -> Program:
    """Schema-aware additive merge used by the evolution engine.

    Args:
        current: Existing program (may be None for new programs)
        incoming: New program data to merge

    Returns:
        Merged program with combined field values
    """

    if current is None:
        return Program.from_dict(incoming.to_dict())

    result_dict = current.to_dict()

    for field_name, field_info in Program.model_fields.items():
        strategy = _field_strategy(field_name, field_info)
        current_val = getattr(current, field_name)
        incoming_val = getattr(incoming, field_name)

        if strategy == "merge_dict":
            merged = dict(current_val)
            merged.update(incoming_val)
            result_dict[field_name] = merged

        elif strategy == "logical_or":
            result_dict[field_name] = current_val or incoming_val

        elif strategy == "state_hierarchy":
            result_dict[field_name] = merge_states(current_val, incoming_val)

        elif strategy == "timestamp":
            result_dict[field_name] = max(current_val, incoming_val)

        elif strategy == "keep_current":
            result_dict[field_name] = current_val

        elif strategy == "warn_if_different":
            if incoming_val != current_val:
                logger.warning(
                    "[additive merge] Field '%s' differs; keeping current value. Current=%r Incoming=%r",
                    field_name,
                    current_val,
                    incoming_val,
                )
            result_dict[field_name] = current_val

    # Always bump updated_at to the newer timestamp
    result_dict["updated_at"] = max(current.updated_at, incoming.updated_at)
    return Program.from_dict(result_dict)


def _overwrite_merge(_current: Optional[Program], incoming: Program) -> Program:
    """Trivial overwrite strategy (incoming wins)."""
    return Program.from_dict(incoming.to_dict())


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _field_strategy(field_name: str, field_info) -> str:
    """Derive merge behaviour for *field_name* based on heuristic rules."""

    annotation = field_info.annotation
    origin = get_origin(annotation)

    # Name-based overrides --------------------------------------------------
    if field_name == "created_at":
        return "keep_current"
    if field_name == "updated_at":
        return "timestamp"
    if field_name == "state":
        return "state_hierarchy"
    if field_name == "id":
        return "keep_current"
    if field_name in {"code", "lineage", "name"}:
        return "warn_if_different"

    # Type-based fallbacks --------------------------------------------------
    if origin is dict or annotation is dict:
        return "merge_dict"
    if annotation is bool:
        return "logical_or"
    if origin is list or annotation is list:
        return "warn_if_different"

    return "warn_if_different"


# ---------------------------------------------------------------------------
# Public resolver
# ---------------------------------------------------------------------------


def resolve_merge_strategy(
    strategy: Union[str, Callable[[Optional[Program], Program], Program]],
) -> Callable[[Optional[Program], Program], Program]:
    """Return a merge function according to *strategy* specification."""

    if callable(strategy):
        return strategy

    if strategy == "additive":
        return _additive_merge
    if strategy == "overwrite":
        return _overwrite_merge

    raise ValueError(
        "Unknown merge_strategy. Provide 'additive', 'overwrite', or a callable."
    )
