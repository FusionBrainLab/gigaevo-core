"""GamSearch must pass max_iters/max_cards through to the vendor ResearchAgent
(the vendor defaults to max_cards=3 silently when unplumbed)."""

from __future__ import annotations

from pathlib import Path

import gigaevo.memory.shared_memory.amem_gam_retriever as gam_helpers
from gigaevo.memory.shared_memory.card_store import CardStore
from gigaevo.memory.shared_memory.gam_search import GamSearch


class _CaptureAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _make_gam_search(tmp_path, monkeypatch, **overrides) -> GamSearch:
    monkeypatch.setattr(
        gam_helpers, "build_gam_store", lambda records, store_dir: ({}, {}, 0)
    )
    monkeypatch.setattr(
        gam_helpers,
        "build_retrievers",
        lambda *a, **k: {"vector": object()},
    )
    return GamSearch(
        research_agent_cls=_CaptureAgent,
        generator=object(),
        card_store=CardStore(index_file=Path(tmp_path) / "index.json"),
        checkpoint_dir=Path(tmp_path),
        gam_store_dir=Path(tmp_path) / "store",
        export_file=Path(tmp_path) / "missing.jsonl",
        allowed_gam_tools={"vector"},
        gam_top_k_by_tool={"vector": 3},
        **overrides,
    )


def test_max_iters_and_max_cards_reach_research_agent(tmp_path, monkeypatch):
    search = _make_gam_search(tmp_path, monkeypatch, max_iters=5, max_cards=1)
    search.build_research_agent()
    assert search.agent.kwargs["max_iters"] == 5
    assert search.agent.kwargs["max_cards"] == 1


def test_defaults_match_vendor(tmp_path, monkeypatch):
    search = _make_gam_search(tmp_path, monkeypatch)
    search.build_research_agent()
    assert search.agent.kwargs["max_iters"] == 3
    assert search.agent.kwargs["max_cards"] == 3
