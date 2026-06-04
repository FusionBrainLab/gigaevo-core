"""Freeze the parent stage outputs that produced a child, for debug reconstruction.

Outputs are written once to the storage stage-output store, content-addressed by a
cache id derived from the serialized bytes (so byte-identical outputs dedup across
sibling children). The child carries only the lean ``{parent_id: {stage: cache_id}}``
map. This survives the parent's NO_CACHE context/suggestion stages being overwritten
when the parent is later re-selected or re-evaluated.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from gigaevo.evolution.mutation.constants import (
    MUTATION_PARENT_STAGE_OUTPUTS_METADATA_KEY,
)
from gigaevo.programs.utils import pickle_b64_deserialize, pickle_b64_serialize

if TYPE_CHECKING:
    from gigaevo.database.program_storage import ProgramStorage
    from gigaevo.programs.program import Program


def stage_output_cache_id(blob: str) -> str:
    """Content-derived id for a serialized stage output (matches content_hash width)."""
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


async def snapshot_parent_stage_outputs(
    parents: list[Program], storage: ProgramStorage
) -> dict[str, dict[str, str]]:
    """Write every parent's stage outputs to the store; return the id-map.

    Returns ``{parent_id: {stage_name: cache_id}}``, skipping parents/stages with
    no output. Empty when nothing is captured (caller omits the metadata key).
    """
    id_map: dict[str, dict[str, str]] = {}
    for parent in parents:
        # The steady-state refresher hands mutation lean parents (stage_results
        # excluded for speed); the outputs live only on the persisted record.
        # Re-load the full record so they are present — read now, at child birth,
        # before the parent's NO_CACHE context stages are overwritten by the next
        # mutation. Parents that already carry stage_results (tests, direct calls)
        # short-circuit without a storage round-trip.
        stage_results = parent.stage_results
        if not stage_results:
            full = await storage.mget([parent.id])
            if not full or full[0] is None:
                continue
            stage_results = full[0].stage_results
        stage_ids: dict[str, str] = {}
        for stage_name, result in stage_results.items():
            if result.output is None:
                continue
            blob = pickle_b64_serialize(result.output)
            cache_id = stage_output_cache_id(blob)
            await storage.put_stage_output(cache_id, blob)
            stage_ids[stage_name] = cache_id
        if stage_ids:
            id_map[parent.id] = stage_ids
    return id_map


async def resolve_parent_stage_outputs(
    child: Program, storage: ProgramStorage
) -> dict[str, dict[str, Any]]:
    """Reconstruct the parent stage outputs that produced *child*.

    Reads the child's id-map and fetches each output from the store, skipping ids
    that are no longer present.
    """
    id_map = child.get_metadata(MUTATION_PARENT_STAGE_OUTPUTS_METADATA_KEY) or {}
    resolved: dict[str, dict[str, Any]] = {}
    for parent_id, stage_ids in id_map.items():
        stages: dict[str, Any] = {}
        for stage_name, cache_id in stage_ids.items():
            blob = await storage.get_stage_output(cache_id)
            if blob is not None:
                stages[stage_name] = pickle_b64_deserialize(blob)
        resolved[parent_id] = stages
    return resolved
