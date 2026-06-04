"""Canonical metadata key constants for mutation context and memory.

All mutation/memory metadata keys live here. Import from this module — never
hardcode the string values.
"""

from typing import Literal

MUTATION_CONTEXT_METADATA_KEY = "mutation_context"
MUTATION_MEMORY_METADATA_KEY = "mutation_memory"
MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY = "memory_selected_idea_ids"

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
