"""Card search, ranking, and LLM synthesis.

Pure functions that score cards against queries and optionally synthesize
results via an LLM service.
"""

from __future__ import annotations

import re

from loguru import logger

from gigaevo.memory.shared_memory.models import AnyCard, MemoryCard, ProgramCard
from gigaevo.memory.shared_memory.protocols import LLMServiceProtocol


def _overlap_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    short, long_s = (a, b) if len(a) <= len(b) else (b, a)
    matches = sum(
        1 for i in range(len(short)) if i < len(long_s) and short[i] == long_s[i]
    )
    return matches / max(len(short), 1)


def apply_render_filters(cards: list[AnyCard]) -> list[AnyCard]:
    mem_cards: list[MemoryCard] = []
    prog_cards: list[ProgramCard] = []
    for c in cards:
        if isinstance(c, ProgramCard):
            prog_cards.append(c)
        elif isinstance(c, MemoryCard):
            mem_cards.append(c)

    def _stats(card: MemoryCard) -> dict[str, float]:
        es = card.evolution_statistics or {}
        return {
            "support": int(es.get("support", 0)),
            "delta_best": float(es.get("delta_best", 0.0)),
        }

    def _has_keyword(card: AnyCard, key: str) -> bool:
        return any(kw == key for kw in (card.keywords or []))

    def _canonical(card: MemoryCard) -> str | None:
        for kw in card.keywords or []:
            if isinstance(kw, str) and kw.startswith("canonical:"):
                return kw
        return None

    filtered_mem: list[MemoryCard] = []
    for c in mem_cards:
        s = _stats(c)
        if _has_keyword(c, "verified:false") and s["support"] < 3:
            continue
        filtered_mem.append(c)

    seen: dict[str, MemoryCard] = {}
    for c in filtered_mem:
        ck = _canonical(c)
        if ck is None:
            seen[c.id] = c
            continue
        prev = seen.get(ck)
        if prev is None or _stats(c)["delta_best"] > _stats(prev)["delta_best"]:
            seen[ck] = c
    filtered_mem = list(seen.values())

    def _first60(card: MemoryCard) -> str:
        return (card.description or "").strip()[:60].lower()

    with_canon: list[MemoryCard] = [c for c in filtered_mem if _canonical(c)]
    without_canon: list[MemoryCard] = [c for c in filtered_mem if not _canonical(c)]

    dedup_no_canon: list[MemoryCard] = []
    for c in without_canon:
        prefix = _first60(c)
        if any(_overlap_ratio(prefix, _first60(d)) > 0.7 for d in dedup_no_canon):
            continue
        dedup_no_canon.append(c)

    dedup_mem = with_canon + dedup_no_canon

    dedup_mem.sort(
        key=lambda c: _stats(c)["support"] * max(_stats(c)["delta_best"], 0.0),
        reverse=True,
    )

    def _filter_prog(c: ProgramCard) -> bool:
        if _has_keyword(c, "pending_analysis:true"):
            return False
        if not (c.description or "").strip() and not c.connected_ideas:
            return False
        return True

    filtered_prog = [c for c in prog_cards if _filter_prog(c)]
    filtered_prog.sort(
        key=lambda c: c.fitness if c.fitness is not None else 0.0,
        reverse=True,
    )

    return dedup_mem + filtered_prog


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
    llm_service: LLMServiceProtocol | None,
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
    except Exception as exc:
        logger.warning(
            "[Memory][CardSearch]LLM synthesis failed, falling back to keyword results: {}",
            exc,
        )

    return format_search_results(query, cards)
