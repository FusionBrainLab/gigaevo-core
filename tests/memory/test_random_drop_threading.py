from __future__ import annotations

from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_LINEAGE_APPLIED_IDS_METADATA_KEY,
)
from gigaevo.memory.core import NullExcluder
from gigaevo.memory.core.random_drop import RandomDropExcluder
from gigaevo.memory.core.read_pipeline import MemoryReadPipeline
from gigaevo.memory.core.retriever import GamRetriever
from gigaevo.memory.core.selection import MemorySelection
from gigaevo.memory.provider import SelectorMemoryProvider

_CONFIG = Path(__file__).resolve().parents[2] / "config" / "memory"


class _Prog:
    def __init__(self, applied):
        self._m = {MUTATION_MEMORY_LINEAGE_APPLIED_IDS_METADATA_KEY: applied}

    def get_metadata(self, key):
        return self._m.get(key)


class _RecordingPipeline:
    def __init__(self):
        self.seen_dose = "UNSET"

    async def select(self, *, random_drop_dose=0, **kw):
        self.seen_dose = random_drop_dose
        return MemorySelection(cards=[], card_ids=[])


def _provider(excluder):
    p = SelectorMemoryProvider(backend=object(), excluder=excluder)
    p._pipeline = _RecordingPipeline()
    return p


@pytest.mark.asyncio
async def test_provider_feeds_dose_to_pipeline():
    p = _provider(RandomDropExcluder())
    await p.select_cards(
        _Prog(["c1", "c2"]), task_description="", metrics_description=""
    )
    assert p._pipeline.seen_dose == 2


@pytest.mark.asyncio
async def test_provider_default_dose_is_zero():
    p = _provider(NullExcluder())
    await p.select_cards(_Prog(["c1"]), task_description="", metrics_description="")
    assert p._pipeline.seen_dose == 0


class _Result:
    raw_memory = None


class _RecordingRetriever:
    def __init__(self):
        self.seen_dose = "UNSET"

    def research(
        self,
        query,
        *,
        planning_request=None,
        exclude_ids=frozenset(),
        random_drop_dose=0,
    ):
        self.seen_dose = random_drop_dose
        return _Result()

    def get_card(self, card_id):
        return None


class _StubSelector:
    def build_core_request(self, **kw):
        return ""

    def build_query(self, **kw):
        return ""

    def shortlist(self, raw_memory):
        return []


class _StubAuctioneer:
    def run(self, candidates, rng):
        return [], []


class _StubBudgeter:
    def cap(self, card_ids, slate, max_cards):
        return []


@pytest.mark.asyncio
async def test_read_pipeline_threads_dose_to_retriever():
    retriever = _RecordingRetriever()
    pipeline = MemoryReadPipeline(
        retriever=retriever,
        selector=_StubSelector(),
        auctioneer=_StubAuctioneer(),
        budgeter=_StubBudgeter(),
        renderer=object(),
        reputation=object(),
    )
    await pipeline.select(
        parents=[_Prog([])],
        mutation_mode="rewrite",
        task_description="",
        metrics_description="",
        random_drop_dose=4,
    )
    assert retriever.seen_dose == 4


class _RecordingBackend:
    def __init__(self):
        self.seen_dose = "UNSET"

    def research(
        self,
        query,
        *,
        planning_request=None,
        exclude_ids=frozenset(),
        random_drop_dose=0,
    ):
        self.seen_dose = random_drop_dose
        return _Result()


def test_gam_retriever_threads_dose_to_backend():
    backend = _RecordingBackend()
    retriever = GamRetriever().bind(backend)
    retriever.research("q", random_drop_dose=3)
    assert backend.seen_dose == 3


def test_hydra_random_drop_config_builds_the_excluder():
    cfg = OmegaConf.load(_CONFIG / "reader" / "excluder" / "random_drop.yaml")
    assert isinstance(instantiate(cfg), RandomDropExcluder)


def test_hydra_default_excluder_is_null():
    cfg = OmegaConf.load(_CONFIG / "reader" / "excluder" / "none.yaml")
    assert isinstance(instantiate(cfg), NullExcluder)
