"""Contract tests for the slimmed memory-selector system prompt (C2)."""

from __future__ import annotations

from gigaevo.prompts import MemorySelectorPrompts


def test_prompt_loads_and_formats_without_keyerror():
    rendered = MemorySelectorPrompts.system().format()
    assert len(rendered) > 0


def test_prompt_keeps_mandatory_empty_hand():
    rendered = MemorySelectorPrompts.system().format()
    assert "empty selection" in rendered
    assert "padding wastes the mutator's context budget" in rendered.lower()


def test_prompt_drops_numeric_ranking_and_grammar_tax():
    rendered = MemorySelectorPrompts.system().format()
    assert "Δbest" not in rendered
    assert "2b" not in rendered
    assert "70%" not in rendered
    assert "support=N" not in rendered
    assert "packed-grammar" not in rendered.lower()


def test_prompt_keeps_semantic_caveats():
    rendered = MemorySelectorPrompts.system().format()
    assert "verified:false" in rendered
    assert "mechanism_unverified:true" in rendered


def test_prompt_within_size_budget():
    raw = MemorySelectorPrompts.system()
    assert len(raw) <= 4000
