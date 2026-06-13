"""Machine keyword tokens must not leak into GAM corpus text or search index.

``verified:`` / ``mechanism_unverified:`` / ``canonical:`` tokens carry
verification-gate semantics for the selector prompt and canonical dedup; they
stay on the stored card but must not surface as topical content in
``render_card_text`` (GAM page corpus) or match queries in
``search_cards_by_keyword``.
"""

from __future__ import annotations

from gigaevo.memory.shared_memory.amem_gam_retriever import render_card_text
from gigaevo.memory.shared_memory.card_search import search_cards_by_keyword
from gigaevo.memory.shared_memory.models import MemoryCard, MemoryCardExplanation

_MACHINE_KEYWORDS = [
    "canonical:CLIP:y_pred:none:0.97",
    "verified:true",
    "mechanism_unverified:monotone damping",
]


def _card(id: str = "m1", keywords: list[str] | None = None) -> MemoryCard:
    return MemoryCard(
        id=id,
        description="spatial target-encoding for capped target",
        keywords=keywords or [],
        explanation=MemoryCardExplanation(summary=""),
    )


class TestRenderCardText:
    def test_machine_tokens_stripped_topical_kept(self) -> None:
        card = _card(keywords=["target-encoding", *_MACHINE_KEYWORDS])
        text = render_card_text(card)
        keywords_line = next(
            line for line in text.splitlines() if line.startswith("keywords:")
        )
        assert keywords_line == "keywords: target-encoding"
        assert "canonical:" not in text
        assert "verified:" not in text
        assert "mechanism_unverified:" not in text

    def test_only_machine_tokens_renders_empty_keywords(self) -> None:
        card = _card(keywords=list(_MACHINE_KEYWORDS))
        text = render_card_text(card)
        keywords_line = next(
            line for line in text.splitlines() if line.startswith("keywords:")
        )
        assert keywords_line.removeprefix("keywords:").strip() == ""

    def test_stored_card_keeps_machine_tokens(self) -> None:
        card = _card(keywords=["target-encoding", *_MACHINE_KEYWORDS])
        render_card_text(card)
        assert set(_MACHINE_KEYWORDS) <= set(card.keywords)


class TestSearchCardsByKeyword:
    def test_machine_token_does_not_match_query(self) -> None:
        card = _card(keywords=list(_MACHINE_KEYWORDS))
        hits = search_cards_by_keyword(
            {card.id: card}, query="clip", memory_state=None, search_limit=5
        )
        assert hits == []

    def test_topical_keyword_still_matches(self) -> None:
        card = _card(keywords=["quantile-clipping", *_MACHINE_KEYWORDS])
        hits = search_cards_by_keyword(
            {card.id: card}, query="quantile", memory_state=None, search_limit=5
        )
        assert [c.id for c in hits] == [card.id]
