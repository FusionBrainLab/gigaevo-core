"""Pre-retrieval card excluders (filter-first lineage gate).

The provider asks an excluder "which ids must not be retrieved for this program?"
and threads the answer into the GAM research pass, so the selector-LLM ranks only
over lineage-fresh candidates. ``NullExcluder`` is the default (byte-identical to
the un-gated read path); ``LineageExcluder`` reads the birth-frozen closure.
"""

from __future__ import annotations

from typing import Any

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_LINEAGE_APPLIED_IDS_METADATA_KEY,
)


class NullExcluder:
    """Excludes nothing — the control arm."""

    def exclude_for(self, program: Any) -> frozenset[str]:
        return frozenset()

    def dose_for(self, program: Any) -> int:
        return 0


class LineageExcluder:
    """Excludes every card applied to this program or any ancestor."""

    def exclude_for(self, program: Any) -> frozenset[str]:
        applied = program.get_metadata(MUTATION_MEMORY_LINEAGE_APPLIED_IDS_METADATA_KEY)
        return frozenset(applied or ())

    def dose_for(self, program: Any) -> int:
        return 0
