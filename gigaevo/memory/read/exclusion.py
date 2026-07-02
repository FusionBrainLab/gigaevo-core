"""Pre-retrieval card excluders (filter-first lineage gate).

The provider asks an excluder "which ids must not be retrieved for this
program?" and threads the answer into the research pass, so the reflector
ranks only over lineage-fresh candidates. ``NullExcluder`` is the default
(byte-identical to the un-gated read path); ``LineageExcluder`` reads the
birth-frozen closure.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_LINEAGE_APPLIED_IDS_METADATA_KEY,
)


@runtime_checkable
class CardExcluder(Protocol):
    """Decides which card ids must be pruned from the candidate pool BEFORE
    retrieval ranks them (filter-first lineage gate)."""

    def exclude_for(self, program: Any) -> frozenset[str]: ...


class NullExcluder:
    """Excludes nothing — the control arm."""

    def exclude_for(self, program: Any) -> frozenset[str]:
        return frozenset()


class LineageExcluder:
    """Excludes every card applied to this program or any ancestor."""

    def exclude_for(self, program: Any) -> frozenset[str]:
        applied = program.get_metadata(MUTATION_MEMORY_LINEAGE_APPLIED_IDS_METADATA_KEY)
        return frozenset(applied or ())
