"""GigaEvo Memory — card-based memory system for evolution-guided mutation.

Four strict layers (see README.md):
    cards.py    — domain models: Card, ContextualGain, DecisionContext
    storage/    — MemoryStore ABC + LocalMemoryStore (bank ∘ index ∘ research)
    read/       — read-side selection surface: MemorySelection value type +
                  shortlist/exclusion protocols (memory_v2 assembles the policy)
    write/      — MemoryWriter: extraction → librarian → admission → merge
    provider.py — engine-facing adapters (the stable import surface)

Only the light domain layer is re-exported here so ``import gigaevo`` stays
light for leaf tools. Import the heavy nodes from their submodules directly:
    LocalMemoryStore → ``gigaevo.memory.storage.local``
    MemoryWriter     → ``gigaevo.memory.write.writer``
"""

from __future__ import annotations

from gigaevo.memory.cards import (
    AssignmentRecord,
    Card,
    CardKind,
    CausalStrength,
    ContextualGain,
    DecisionContext,
    EvidenceAttribution,
    EvidenceSource,
    card_brief,
)

__all__ = [
    "AssignmentRecord",
    "Card",
    "CardKind",
    "CausalStrength",
    "ContextualGain",
    "DecisionContext",
    "EvidenceAttribution",
    "EvidenceSource",
    "card_brief",
]
