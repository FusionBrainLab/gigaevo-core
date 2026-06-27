from __future__ import annotations

import inspect

from gigaevo.memory.shared_memory import card_search
from gigaevo.memory.shared_memory.card_search import search_cards_by_keyword
from gigaevo.memory.shared_memory.models import MemoryCard
from gigaevo.prompts import load_prompt


class TestDeadPromptCaveatsRemoved:
    def test_mutation_prompt_drops_verified_caveat(self) -> None:
        text = load_prompt("mutation", "system").lower()
        assert "verified:" not in text
        assert "unverified:" not in text

    def test_memory_selector_prompt_drops_verified_caveat(self) -> None:
        text = load_prompt("memory_selector", "system").lower()
        assert "verified:" not in text
        assert "unverified:" not in text

    def test_mutation_prompt_retains_quality_rules(self) -> None:
        text = load_prompt("mutation", "system")
        assert "Tautology test" in text
        assert "Cite, never invent" in text


def _card(cid: str, keywords: list[str]) -> MemoryCard:
    return MemoryCard(id=cid, description="d", keywords=keywords)


class TestKeywordFilterNoMachinePrefix:
    def test_machine_prefixed_keyword_becomes_searchable(self) -> None:
        cards = {"c1": _card("c1", ["verified:zzqq"])}
        hits = search_cards_by_keyword(cards, "zzqq", None, 10)
        assert "c1" in {c.id for c in hits}


class TestMachinePrefixSymbolGone:
    def test_card_search_drops_machine_prefix_symbol(self) -> None:
        assert "MACHINE_KEYWORD_PREFIXES" not in inspect.getsource(card_search)
