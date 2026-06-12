"""Canonical metadata key constants for mutation context and memory.

All mutation/memory metadata keys live here. Import from this module — never
hardcode the string values.
"""

from typing import Literal

MUTATION_CONTEXT_METADATA_KEY = "mutation_context"
MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY = "memory_selected_idea_ids"

#: Structured mutation output (archetype, changes) stamped on a child program's
#: metadata by the mutation operator (``MutationSpec.META_OUTPUT``).
MUTATION_OUTPUT_METADATA_KEY = "mutation_output"

#: Per-mutation Thompson-auction record: one entry per candidate card offered to
#: the auction (card_id, posterior a/b, draws, baseline arm, selected). Written
#: even when zero cards win, so the offline lifecycle analysis can see which
#: candidates were rejected (the "no-card" outcome).
MUTATION_MEMORY_CANDIDATE_SLATE_METADATA_KEY = "memory_candidate_slate"

#: Frozen onto a child at birth: per parent, the cache id of every parent stage
#: output that produced the child (outputs live content-addressed in the storage
#: stage-output store). Survives the parent's NO_CACHE stages being overwritten
#: on re-selection/re-eval. Maps parent_id -> {stage_name: cache_id}.
MUTATION_PARENT_STAGE_OUTPUTS_METADATA_KEY = "parent_stage_outputs"


ARCHETYPE_NAMES: tuple[str, ...] = (
    "Precision Optimization",
    "Proven Pattern Extension",
    "Harmful Pattern Removal",
    "Computational Reinvention",
    "Solution Space Exploration",
    "Approach Synthesis",
    "Guided Innovation",
    "Component Substitution",
)

ArchetypeName = Literal[
    "Precision Optimization",
    "Proven Pattern Extension",
    "Harmful Pattern Removal",
    "Computational Reinvention",
    "Solution Space Exploration",
    "Approach Synthesis",
    "Guided Innovation",
    "Component Substitution",
]
