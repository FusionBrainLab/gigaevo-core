"""Card search, ranking, and LLM synthesis.

Pure functions that score cards against queries and optionally synthesize
results via an LLM service.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from loguru import logger

from gigaevo.memory.shared_memory.models import (
    AnyCard,
    CardStatsBlock,
    MemoryCard,
    ProgramCard,
)
from gigaevo.memory.shared_memory.protocols import LLMServiceProtocol

if TYPE_CHECKING:
    from gigaevo.memory.core.protocols import ReputationModel


def topical_keywords(keywords: list[str] | None) -> list[str]:
    """Card keywords used as search/corpus text."""
    return list(keywords or [])


def format_block_efficacy(
    card: AnyCard | None, block: CardStatsBlock | None
) -> str | None:
    """One legible per-card endorsement line for the mutator, or None.

    ``block`` is the reputation's ``card_stats`` for the decision context, so the
    rendered line reflects the same locality the auction bid on — cell-local under
    BD proximity, global otherwise. MemoryCard: rendered only when the
    Beta-Binomial downside posterior is confident — non-confident and no-signal
    cards stay silent (description only). ProgramCard: exemplar fitness
    (block-independent).
    """
    if card is None:
        return None
    if isinstance(card, ProgramCard):
        if card.fitness is None:
            return None
        return f"efficacy: exemplar fitness {card.fitness:.4f}"

    if block is None:
        return None
    intros = block.intro_events
    median = block.IntroGain_best_median
    if intros <= 0 or median is None:
        return None
    if not block.efficacy_confident:
        return None
    # Gains are stored in "positive = improvement" space regardless of metric
    # direction (analyse() negates for minimize metrics), so the wording must be
    # direction-neutral — "fitness change +x" would read inverted on minimize.
    line = (
        f"efficacy: introduced in {intros} children; "
        f"median improvement {float(median):+.4f}"
    )
    # A noise-band-confident posterior with a losing median must never read as
    # an endorsement.
    if float(median) <= 0:
        return line + " (caution: non-positive median)"
    return line + " (confident)"


def format_card_efficacy(
    card: AnyCard | None, reputation: ReputationModel | None = None
) -> str | None:
    """Context-free efficacy line: the card's own global stats, computed by
    reputation from its stored gain events (no decision context).

    For callers without a decision context (dedup, GAM corpus build). The
    decision render path resolves a context-aware block via ``card_stats`` and
    calls ``format_block_efficacy`` directly. Defaults to the global
    Beta-Binomial reputation when no model is supplied.
    """
    if card is None:
        return None
    if reputation is None:
        from gigaevo.memory.core.reputation import BetaBinomialReputation

        reputation = BetaBinomialReputation()
    block = reputation.card_stats(card) if isinstance(card, MemoryCard) else None
    return format_block_efficacy(card, block)


def format_card_brief(card: AnyCard) -> str:
    """Compact card projection for the librarian judging prompts (reconcile /
    consolidate): description + why-text + keywords on one line, empty fields
    omitted. The reconcile caller prepends the id (it needs it as the
    DUPLICATE/MERGE target); consolidate uses the body alone.

    Carries ``explanation_summary`` and ``keywords`` so the judge can tell apart
    overlapping descriptions with different causal rationales and preserve a
    partner's rationale in an authored union — the dedup INDEX stays mechanism-
    keyed, only the post-recall judging prompt is enriched.
    """
    parts = [card.description]
    why = (card.explanation_summary or "").strip()
    if why:
        parts.append(f"why: {why}")
    kws = topical_keywords(card.keywords)
    if kws:
        parts.append(f"keywords: {', '.join(kws)}")
    return " | ".join(parts)


def format_search_results(query: str, cards: list[AnyCard]) -> str:
    """Format search results as numbered card list for LLM card-selector parsing."""
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
                str(card.explanation_summary or ""),
                str(card.task_description_summary or ""),
                str(card.task_description or ""),
                " ".join(topical_keywords(card.keywords)),
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
        cards_blob.append(
            "\n".join(
                [
                    f"id: {card.id}",
                    f"category: {card.category}",
                    f"task_description_summary: {card.task_description_summary}",
                    f"task_description: {card.task_description}",
                    f"description: {card.description}",
                    f"explanation_summary: {card.explanation_summary}",
                    f"keywords: {topical_keywords(card.keywords)}",
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
            "[Memory][CardSearch] LLM synthesis failed, falling back to keyword "
            "results: {}",
            exc,
        )

    return format_search_results(query, cards)
