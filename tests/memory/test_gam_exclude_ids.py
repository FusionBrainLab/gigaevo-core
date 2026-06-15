from __future__ import annotations

from gigaevo.memory._vendor.GAM_root.gam.agents.research_agent import ResearchAgent
from gigaevo.memory._vendor.GAM_root.gam.schemas.search import Hit


def _bare_agent(card_map):
    agent = object.__new__(ResearchAgent)
    agent._card_map_by_id = lambda: card_map  # type: ignore[method-assign]
    return agent


def _ideas(agent, hits, **kw):
    return [i["card_id"] for i in agent._build_retrieved_ideas(hits, **kw)]


CARD_MAP = {
    "card_a": {"id": "card_a", "description": "alpha"},
    "card_b": {"id": "card_b", "description": "beta"},
}
HITS = [
    Hit(page_id="card_a", snippet="a", source="vector"),
    Hit(page_id="card_b", snippet="b", source="page_index"),
]


def test_excluded_card_id_is_dropped_from_both_tools():
    agent = _bare_agent(CARD_MAP)
    assert _ideas(agent, HITS, exclude_ids=frozenset({"card_a"})) == ["card_b"]


def test_empty_exclude_keeps_all():
    agent = _bare_agent(CARD_MAP)
    assert _ideas(agent, HITS) == ["card_a", "card_b"]


def test_all_excluded_yields_empty_pool():
    agent = _bare_agent(CARD_MAP)
    assert _ideas(agent, HITS, exclude_ids=frozenset({"card_a", "card_b"})) == []
