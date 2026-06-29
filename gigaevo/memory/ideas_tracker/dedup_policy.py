"""DedupPolicy: the near-duplicate thresholds for the librarian write path.

One frozen value object groups every dedup knob — the online pre-gate
thresholds applied per mutation diff and the batch consolidation thresholds
applied across the whole bank — so they live in one Hydra-instantiable node
instead of scattered scalar defaults on the Librarian, the consolidation pass,
and the scheduler.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DedupPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    online_eps: float = Field(
        default=0.05,
        description="Cosine distance below which the closest existing card is a "
        "near-duplicate; the online pre-gate bumps its provenance instead of "
        "authoring a new card.",
    )
    online_top_k: int = Field(
        default=5,
        description="Neighbours fetched per diff to seed the pre-gate check and "
        "the reconcile agent's context.",
    )
    max_cards_per_diff: int = Field(
        default=3,
        description="Upper bound on cards authored from a single mutation diff.",
    )
    consolidation_eps: float = Field(
        default=0.2,
        description="Cosine distance below which two bank cards are surfaced as a "
        "merge CANDIDATE during a background consolidation pass. This is a "
        "generous candidate-recall gate, not the merge decision: the consolidate "
        "agent reviews each candidate pair and folds only the ones it rules name "
        "the same lever (abstaining otherwise), so the gate can stay loose — "
        "~0.2 admits cards with strong semantic overlap (cosine similarity ~0.8) "
        "without auto-merging anything. Precision is delegated to the agent, "
        "unlike the tight silent-auto-merge ``online_eps``.",
    )
    consolidation_k: int = Field(
        default=5,
        description="Neighbours fetched per card during a consolidation pass.",
    )
