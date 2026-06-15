from __future__ import annotations

import pytest

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_LINEAGE_APPLIED_IDS_METADATA_KEY,
)
from gigaevo.memory.core import LineageExcluder, NullExcluder
from gigaevo.memory.core.selection import MemorySelection
from gigaevo.memory.provider import SelectorMemoryProvider


class _Prog:
    def __init__(self, applied):
        self._m = {MUTATION_MEMORY_LINEAGE_APPLIED_IDS_METADATA_KEY: applied}

    def get_metadata(self, key):
        return self._m.get(key)


class _RecordingPipeline:
    def __init__(self):
        self.seen_exclude = "UNSET"

    async def select(self, *, exclude_ids=frozenset(), **kw):
        self.seen_exclude = exclude_ids
        return MemorySelection(cards=[], card_ids=[])


def _provider(excluder):
    p = SelectorMemoryProvider(backend=object(), excluder=excluder)
    p._pipeline = _RecordingPipeline()  # skip the heavy lazy build
    return p


@pytest.mark.asyncio
async def test_lineage_excluder_feeds_closure_to_pipeline():
    p = _provider(LineageExcluder())
    await p.select_cards(
        _Prog(["c1", "c2"]), task_description="", metrics_description=""
    )
    assert p._pipeline.seen_exclude == frozenset({"c1", "c2"})


@pytest.mark.asyncio
async def test_default_provider_excludes_nothing():
    p = _provider(None)  # None -> NullExcluder
    await p.select_cards(_Prog(["c1"]), task_description="", metrics_description="")
    assert p._pipeline.seen_exclude == frozenset()


def test_provider_defaults_to_null_excluder():
    assert isinstance(SelectorMemoryProvider(backend=object())._excluder, NullExcluder)
