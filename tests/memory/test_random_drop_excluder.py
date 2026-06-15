from __future__ import annotations

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_LINEAGE_APPLIED_IDS_METADATA_KEY,
)
from gigaevo.memory.core import LineageExcluder, NullExcluder
from gigaevo.memory.core.random_drop import RandomDropExcluder


class _Prog:
    def __init__(self, applied):
        self._m = {MUTATION_MEMORY_LINEAGE_APPLIED_IDS_METADATA_KEY: applied}

    def get_metadata(self, key):
        return self._m.get(key)


class _Legacy:
    def get_metadata(self, key):
        return None


def test_random_drop_excludes_nothing_by_id():
    assert RandomDropExcluder().exclude_for(_Prog(["a", "b", "c"])) == frozenset()


def test_random_drop_dose_is_the_closure_size():
    assert RandomDropExcluder().dose_for(_Prog(["a", "b", "c"])) == 3


def test_random_drop_dose_matches_lineage_closure_exactly():
    prog = _Prog(["x", "y"])
    assert RandomDropExcluder().dose_for(prog) == len(
        LineageExcluder().exclude_for(prog)
    )


def test_random_drop_dose_zero_when_no_lineage_metadata():
    assert RandomDropExcluder().dose_for(_Legacy()) == 0


def test_null_excluder_dose_is_zero():
    assert NullExcluder().dose_for(_Prog(["a", "b"])) == 0


def test_lineage_excluder_dose_is_zero():
    assert LineageExcluder().dose_for(_Prog(["a", "b"])) == 0
