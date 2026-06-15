from __future__ import annotations

import pytest

from gigaevo.memory._vendor.GAM_root.gam.schemas import ResearchOutput
from gigaevo.memory.core import (
    BetaBinomialReputation,
    EfficacyCardRenderer,
    MemoryReadPipeline,
    ThompsonAuctioneer,
    TopThetaBudgeter,
)


class _RecordingRetriever:
    def __init__(self):
        self.seen_exclude = "UNSET"

    def research(
        self,
        query,
        *,
        planning_request=None,
        exclude_ids=frozenset(),
        random_drop_dose=0,
    ):
        self.seen_exclude = exclude_ids
        return ResearchOutput(
            integrated_memory="", raw_memory={"final_decision": {"top_ideas": []}}
        )

    def get_card(self, card_id):
        return None


class _Selector:
    def build_core_request(self, **k):
        return "req"

    def build_query(self, **k):
        return "q"

    def shortlist(self, raw_memory):
        return []


def _pipeline(retriever):
    return MemoryReadPipeline(
        retriever=retriever,
        selector=_Selector(),
        auctioneer=ThompsonAuctioneer(),
        budgeter=TopThetaBudgeter(),
        renderer=EfficacyCardRenderer(),
        reputation=BetaBinomialReputation(),
    )


@pytest.mark.asyncio
async def test_exclude_ids_reach_the_retriever():
    retr = _RecordingRetriever()
    await _pipeline(retr).select(
        parents=[object()],
        mutation_mode="rewrite",
        task_description="",
        metrics_description="",
        max_cards=1,
        exclude_ids=frozenset({"stale"}),
    )
    assert retr.seen_exclude == frozenset({"stale"})


@pytest.mark.asyncio
async def test_default_select_passes_empty_exclude():
    retr = _RecordingRetriever()
    await _pipeline(retr).select(
        parents=[object()],
        mutation_mode="rewrite",
        task_description="",
        metrics_description="",
        max_cards=1,
    )
    assert retr.seen_exclude == frozenset()
