"""Candidate shortlisting: mutation context in, researched cards out.

Builds the research request that mirrors the mutation agent's context (task,
metrics, mutation mode, parent code + live snapshots) and runs the store's
agentic retrieval. Shortlist width (recall) is the store's
``ResearchConfig.max_cards``; the reader's ``max_cards`` is the downstream
injection budget the budgeter caps to — fusing the two collapses the pool
before any ranker runs.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from typing import Any

from loguru import logger

from gigaevo.evolution.mutation.constants import MUTATION_CONTEXT_METADATA_KEY
from gigaevo.memory.cards import Card
from gigaevo.memory.read.exclusion import expand_exclude_ids, is_card_excluded
from gigaevo.memory.storage.base import MemoryStore, ResearchRequest, ResearchResult

_DIGEST_MAX_CARDS = 50
_DIGEST_LINE_CHARS = 200


def _latest_event_time(card: Card) -> datetime | None:
    stamps = [
        ts if ts.tzinfo else ts.replace(tzinfo=UTC)
        for e in card.gain_events
        if (ts := e.context.timestamp) is not None
    ]
    return max(stamps) if stamps else None


def _digest_order(card: Card) -> tuple[bool, float, str]:
    latest = _latest_event_time(card)
    return (latest is None, -latest.timestamp() if latest else 0.0, card.id)


def _bank_digest(
    cards: tuple[Card, ...],
    *,
    max_cards: int,
    exclude_ids: frozenset[str] = frozenset(),
) -> str:
    """One description line per banked card, so the planner writes queries
    against what the bank actually holds instead of guessing at its contents.
    Newest first, by the latest gain-event timestamp (unstamped cards last,
    id-stable), truncated to the ``max_cards`` freshest: the digest frames
    every research call, so an unbounded bank must not crowd the planner's
    prompt, and under the cap it is the freshest levers that must survive.
    Excluded ids (lineage + upstream-benched) never enter — they cannot be
    selected, so they must not spend digest slots or steer the planner.
    Overflow is summarized as a drop count (visible to the planner, logged
    here)."""
    cards = tuple(card for card in cards if not is_card_excluded(card, exclude_ids))
    if not cards:
        return ""
    lines: list[str] = []
    for card in sorted(cards, key=_digest_order)[:max_cards]:
        desc = " ".join(card.description.split())
        if len(desc) > _DIGEST_LINE_CHARS:
            desc = desc[: _DIGEST_LINE_CHARS - 1] + "…"
        lines.append(f"- {desc}")
    dropped = len(cards) - len(lines)
    if dropped:
        lines.append(
            "(+{} more card{} not shown)".format(dropped, "" if dropped == 1 else "s")
        )
        logger.debug(
            "[Memory][Shortlist] bank digest capped: {} of {} cards shown",
            len(lines) - 1,
            len(cards),
        )
    plural = "" if len(cards) == 1 else "s"
    return "BANK CONTENTS ({} card{}):\n{}".format(len(cards), plural, "\n".join(lines))


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

    def __init__(
        self, store: MemoryStore, digest_max_cards: int = _DIGEST_MAX_CARDS
    ) -> None:
        if digest_max_cards <= 0:
            raise ValueError(
                f"digest_max_cards must be positive, got {digest_max_cards}"
            )
        self._store = store
        self._digest_max_cards = digest_max_cards

    @property
    def policy_digest(self) -> str:
        """Fingerprint request construction plus the store's retrieval policy."""

        store_digest = getattr(self._store, "retrieval_policy_digest", None)
        if not isinstance(store_digest, str) or len(store_digest) != 64:
            store_digest = (
                f"{type(self._store).__module__}.{type(self._store).__qualname__}"
            )
        payload = {
            "shortlister": f"{type(self).__module__}.{type(self).__qualname__}",
            "digest_max_cards": self._digest_max_cards,
            "store_retrieval_policy": store_digest,
        }
        return hashlib.sha256(
            json.dumps(
                payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
        ).hexdigest()

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
            bank = self._store.snapshot()
            effective_exclude_ids = expand_exclude_ids(bank, exclude_ids)
            digest = _bank_digest(
                bank,
                max_cards=self._digest_max_cards,
                exclude_ids=effective_exclude_ids,
            )
        except Exception:
            logger.opt(exception=True).warning(
                "[Memory][Shortlist] bank digest failed; planning without it"
            )
            digest = ""
            effective_exclude_ids = exclude_ids
        try:
            return await self._store.research(
                ResearchRequest(
                    query=query,
                    planning_context=digest,
                    exclude_ids=effective_exclude_ids,
                )
            )
        except Exception:
            logger.opt(exception=True).warning(
                "[Memory][Shortlist] research failed; empty shortlist"
            )
            return ResearchResult()
