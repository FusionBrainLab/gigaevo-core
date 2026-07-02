"""Candidate shortlisting: mutation context in, researched cards out.

Builds the research request that mirrors the mutation agent's context (task,
metrics, mutation mode, parent code + live snapshots) and runs the store's
agentic retrieval. Shortlist width (recall) is the store's
``ResearchConfig.max_cards``; the reader's ``max_cards`` is the downstream
injection budget the budgeter caps to — fusing the two collapses the pool
before any ranker runs.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from gigaevo.evolution.mutation.constants import MUTATION_CONTEXT_METADATA_KEY
from gigaevo.memory.storage.base import MemoryStore, ResearchRequest, ResearchResult


def _parent_blocks(parents: list[Any], parent_contexts: list[str] | None = None) -> str:
    blocks: list[str] = []
    for i, parent in enumerate(parents):
        if parent_contexts is not None:
            formatted_context = parent_contexts[i] if i < len(parent_contexts) else ""
        else:
            formatted_context = parent.metadata.get(MUTATION_CONTEXT_METADATA_KEY) or ""
        block = f"""=== Parent {i + 1} ===
```python
{parent.code}
```

{formatted_context}
"""
        blocks.append(block)
    return "\n\n".join(blocks)


def build_research_query(
    *,
    parents: list[Any],
    mutation_mode: str,
    task_description: str,
    metrics_description: str,
    parent_contexts: list[str] | None = None,
) -> str:
    """The mutation-grounded retrieval request both the planner and the
    reflector judge against."""
    return (
        "MUTATION INPUTS\n\n"
        "TASK DESCRIPTION:\n"
        f"{task_description.strip() or '<empty>'}\n\n"
        "AVAILABLE METRICS:\n"
        f"{metrics_description.strip() or '<empty>'}\n\n"
        "MUTATION MODE:\n"
        f"{mutation_mode.strip() or 'rewrite'}\n\n"
        "PARENTS (parent code + this-pass lineage card + live evolutionary snapshot):\n"
        f"{_parent_blocks(parents, parent_contexts)}\n\n"
        "Find the stored cards whose mechanism overlaps a plausible next "
        "mutation of these parents; select none if no card overlaps."
    )


class ResearchShortlister:
    """Runs the store's research pass over the mutation-grounded query.

    Fail-to-empty: any store failure degrades to an empty result so a memory
    outage can never sink a mutation.
    """

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def shortlist(
        self,
        *,
        parents: list[Any],
        mutation_mode: str,
        task_description: str,
        metrics_description: str,
        exclude_ids: frozenset[str] = frozenset(),
        parent_contexts: list[str] | None = None,
    ) -> ResearchResult:
        query = build_research_query(
            parents=parents,
            mutation_mode=mutation_mode,
            task_description=task_description,
            metrics_description=metrics_description,
            parent_contexts=parent_contexts,
        )
        try:
            return await self._store.research(
                ResearchRequest(query=query, exclude_ids=exclude_ids)
            )
        except Exception:
            logger.opt(exception=True).warning(
                "[Memory][Shortlist] research failed; empty shortlist"
            )
            return ResearchResult()
