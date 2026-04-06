"""Card search, ranking, and LLM synthesis.

Pure functions that score cards against queries and optionally synthesize
results via an LLM service.
"""

from __future__ import annotations

import re
from typing import Any

from gigaevo.memory.shared_memory.models import AnyCard, MemoryCard


def format_search_results(query: str, cards: list[AnyCard]) -> str:
    """Format search results as numbered card list for MemorySelectorAgent parsing."""
    lines = [f"Query: {query}", "", "Top relevant memory cards:"]
    for idx, card in enumerate(cards, start=1):
        lines.append(f"{idx}. {card.id} [{card.category}] {card.description.strip()}")
    return "\n".join(lines)


def search_cards_by_keyword(
    cards_dict: dict[str, AnyCard],
    query: str,
    memory_state: str | None,
    search_limit: int,
) -> list[AnyCard]:
    """Score and rank cards by keyword match. Pure function.

    Args:
        cards_dict: Card ID -> Card mapping
        query: Search query
        memory_state: Optional memory state context (added to query)
        search_limit: Max cards to return

    Returns:
        Top-ranked cards, sorted by match score (highest first)
    """
    if not cards_dict:
        return []

    query_text = f"{query} {memory_state or ''}".strip().lower()
    tokens = [tok for tok in re.split(r"\W+", query_text) if tok]
    if not tokens:
        tokens = [query.strip().lower()] if query.strip() else []

    scored: list[tuple[int, AnyCard]] = []
    for card in cards_dict.values():
        haystack_text = " ".join(
            [
                str(card.description or ""),
                str(card.task_description_summary or ""),
                str(card.task_description or ""),
                " ".join([str(x) for x in (card.keywords or [])]),
                str(card.category or ""),
            ]
        ).lower()
        haystack_tokens = set(re.split(r"\W+", haystack_text))
        score = sum(1 for tok in tokens if tok and tok in haystack_tokens)
        if score > 0:
            scored.append((score, card))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [card for _, card in scored[:search_limit]]


def synthesize_search_results(
    query: str,
    memory_state: str | None,
    cards: list[AnyCard],
    llm_service: Any | None,
) -> str:
    """Use LLM to synthesize search results, or fall back to plain format.

    Args:
        query: User search query
        memory_state: Optional memory state context
        cards: Retrieved memory cards
        llm_service: LLM service (protocol: has generate(str) method).
            If None, uses plain format.

    Returns:
        Synthesized result text
    """
    if llm_service is None:
        return format_search_results(query, cards)

    cards_blob = []
    for card in cards:
        expl_text = card.explanation.summary if isinstance(card, MemoryCard) else ""
        cards_blob.append(
            "\n".join(
                [
                    f"id: {card.id}",
                    f"category: {card.category}",
                    f"task_description_summary: {card.task_description_summary}",
                    f"task_description: {card.task_description}",
                    f"description: {card.description}",
                    f"keywords: {card.keywords}",
                    f"explanation: {expl_text}",
                ]
            )
        )

    prompt = (
        "You are a memory retrieval assistant.\n"
        "Use only the provided memory cards to answer the user query.\n"
        "Always cite card ids explicitly (example: mem-029).\n"
        "If evidence is insufficient, say so clearly.\n\n"
        f"Memory state:\n{memory_state or '(empty)'}\n\n"
        f"User query:\n{query}\n\n"
        "Retrieved cards:\n" + "\n\n".join(cards_blob) + "\n\nAnswer:"
    )

    try:
        text, _, _, _ = llm_service.generate(prompt)
        text = str(text or "").strip()
        if text:
            return text
    except Exception:
        pass

    return format_search_results(query, cards)
