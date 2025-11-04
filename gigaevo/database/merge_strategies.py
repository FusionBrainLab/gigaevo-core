from typing import Any

from loguru import logger

from gigaevo.programs.program import Lineage, Program
from gigaevo.programs.program_state import merge_states


def _merge_dict_by_prog_ts(
    current_d: dict[str, Any],
    incoming_d: dict[str, Any],
    *,
    current_prog: Program,
    incoming_prog: Program,
) -> dict[str, Any]:
    """
    Shallow-merge dicts with tie-breaker ONLY by Program.atomic_counter:
      - New keys -> take incoming
      - Same key, different values -> take side with larger Program.atomic_counter
    """
    merged: dict[str, Any] = dict(current_d or {})
    if not (inc := incoming_d):
        return merged

    prefer_incoming = incoming_prog.atomic_counter > current_prog.atomic_counter
    if prefer_incoming:
        merged.update(inc)
    else:
        for k, v in inc.items():
            if k not in merged:
                merged[k] = v
    return merged


def _merge_lineage(curr: Lineage, inc: Lineage) -> Lineage:
    """
    Keep lineage identity (parents/mutation/generation) from current.
    Children are a stable union (dedup, preserve current order).
    """
    if inc and (
        (inc.parents and inc.parents != curr.parents)
        or (inc.mutation is not None and inc.mutation != curr.mutation)
        or (inc.generation is not None and inc.generation != curr.generation)
    ):
        logger.warning(
            "[merge] lineage immutables differ; keeping current immutables. "
            f"current={curr} incoming={inc}"
        )

    cur_children = list(curr.children or [])
    inc_children = list(inc.children or []) if inc else []
    seen = set(cur_children)
    merged_children = cur_children + [c for c in inc_children if c not in seen]

    return Lineage(
        parents=list(curr.parents or []),
        children=merged_children,
        mutation=curr.mutation,
        generation=curr.generation,
    )


def merge_programs(current: Program | None, incoming: Program) -> Program:
    """
    Minimal additive merge (dict tie-breaks ONLY by Program.atomic_counter):
      - id / created_at / atomic_counter / code / name are immutable -> keep from CURRENT (storage bumps atomic_counter later)
      - state                         -> merge_states(current, incoming)
      - metadata, metrics, stage_results -> shallow dict merge (program-level ts tie-break)
      - lineage                       -> keep immutables; union children
      - code/name                     -> keep current; fill name if current is None
    """
    if current is None:
        return incoming.model_copy(deep=True)

    if current.id != incoming.id:
        raise ValueError(f"id mismatch: current={current.id} incoming={incoming.id}")
    if current.code != incoming.code:
        raise ValueError(
            f"code mismatch: current={current.code} incoming={incoming.code}"
        )
    if (
        current.name is not None
        and incoming.name is not None
        and current.name != incoming.name
    ):
        raise ValueError(
            f"name mismatch: current={current.name} incoming={incoming.name}"
        )

    updates: dict[str, Any] = {}

    # Lifecycle state lattice
    updates["state"] = merge_states(current.state, incoming.state)
    # Dict merges (same-key conflict -> side with larger Program.atomic_counter)
    updates["metadata"] = _merge_dict_by_prog_ts(
        current.metadata,
        incoming.metadata,
        current_prog=current,
        incoming_prog=incoming,
    )
    updates["metrics"] = _merge_dict_by_prog_ts(
        current.metrics, incoming.metrics, current_prog=current, incoming_prog=incoming
    )
    updates["stage_results"] = _merge_dict_by_prog_ts(
        current.stage_results,
        incoming.stage_results,
        current_prog=current,
        incoming_prog=incoming,
    )
    # Lineage
    updates["lineage"] = _merge_lineage(current.lineage, incoming.lineage)

    # Identity & timestamping (storage owns counter)
    updates["id"] = current.id
    updates["created_at"] = current.created_at
    updates["code"] = current.code
    updates["name"] = current.name

    # Will be updated by storage
    updates["atomic_counter"] = current.atomic_counter

    return current.model_copy(update=updates, deep=True)


def resolve_merge_strategy(strategy: str):
    if strategy == "additive":
        return merge_programs
    raise ValueError("Unknown merge_strategy. Only 'additive' is supported.")
