"""Canonical metadata key constants for mutation context and memory.

All mutation/memory metadata keys live here. Import from this module — never
hardcode the string values.
"""

from typing import Literal

MUTATION_CONTEXT_METADATA_KEY = "mutation_context"
MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY = "memory_selected_idea_ids"
MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY = "memory_no_card_control"

#: Frozen onto a child at birth: sorted union of the parents' prompt-time
#: selected card ids. Posterior attribution reads this stamp — the parents'
#: own selected-ids metadata is overwritten on every NO_CACHE requeue, so it
#: cannot be trusted at sweep time. Empty list means "born without cards";
#: absent means legacy program (fall back to parents' current slates).
MUTATION_MEMORY_INJECTED_IDS_METADATA_KEY = "memory_injected_idea_ids"

#: Bool stamped on a child at birth: whether any card was in its mutation prompt.
MUTATION_MEMORY_USED_METADATA_KEY = "memory_used"

#: Frozen onto a child at birth: sorted transitive closure of every card id
#: applied to this program or any ancestor — ``injected_ids(child)`` unioned with
#: each parent's own lineage-applied closure. Read by the lineage card-excluder so
#: a mutation never re-serves a card already used up the ancestry. Sourced only
#: from frozen keys (a program's own selected-ids are overwritten on requeue).
MUTATION_MEMORY_LINEAGE_APPLIED_IDS_METADATA_KEY = "memory_lineage_applied_ids"

#: Structured mutation output (archetype, changes) stamped on a child program's
#: metadata by the mutation operator (``MutationSpec.META_OUTPUT``).
MUTATION_OUTPUT_METADATA_KEY = "mutation_output"

#: Per-mutation Thompson-auction record: one entry per candidate card offered to
#: the auction (card_id, posterior a/b, draws, baseline arm, selected). Written
#: even when zero cards win, so the offline lifecycle analysis can see which
#: candidates were rejected (the "no-card" outcome).
MUTATION_MEMORY_CANDIDATE_SLATE_METADATA_KEY = "memory_candidate_slate"

#: Legacy/base-parent selected ids frozen at birth. Current attribution prefers
#: MUTATION_MEMORY_INJECTED_IDS_METADATA_KEY (the full prompt slate across all
#: parents) so crossover donor cards can be credited; this key still supplies a
#: base-parent fallback for older children.
MUTATION_MEMORY_BASE_SELECTED_IDS_METADATA_KEY = "memory_base_selected_idea_ids"

#: Frozen onto a child at birth: the base parent's metric dict — the decision context
#: and the reward baseline (base_fitness = base_metrics[fitness_key], derived at the
#: write seam where fitness_key is known).
MUTATION_MEMORY_BASE_METRICS_METADATA_KEY = "memory_base_metrics"

#: Frozen onto a child at birth: the base parent's program id — the decision-context
#: parent identity that accompanies base_metrics through use-attribution.
MUTATION_MEMORY_BASE_ID_METADATA_KEY = "memory_base_id"

#: Frozen onto a child at birth: the base parent's per-sample score vector
#: (``per_sample_scores`` metadata), when the eval emits one. Coherent with the
#: frozen base_metrics — the parent's own vector is overwritten on re-eval, so
#: paired crediting must read the child's stamp. Absent when the eval is
#: scalar-only; crediting degrades to the exact point delta.
MUTATION_MEMORY_BASE_SCORES_METADATA_KEY = "memory_base_scores"

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
