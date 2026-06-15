"""Dose-matched random-drop control excluder (lineage isolation).

``LineageExcluder`` entangles two mechanisms in its +best-CV gain: AWARENESS
(dropping the *ancestrally-applied* cards specifically) and DOSE (just removing
``k`` cards from the slate). This excluder reproduces the DOSE alone — it excludes
nothing by id but reports a per-program drop dose equal to the lineage-closure
size, so the read path drops that many *random* pool hits. The residual gap
between the lineage arm and this control is the awareness-attributable effect.
"""

from __future__ import annotations

from typing import Any

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_LINEAGE_APPLIED_IDS_METADATA_KEY,
)


class RandomDropExcluder:
    """Excludes nothing by id; doses the random drop to the lineage-closure size."""

    def exclude_for(self, program: Any) -> frozenset[str]:
        return frozenset()

    def dose_for(self, program: Any) -> int:
        applied = program.get_metadata(MUTATION_MEMORY_LINEAGE_APPLIED_IDS_METADATA_KEY)
        return len(applied or ())
