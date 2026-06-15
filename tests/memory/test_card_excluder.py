from __future__ import annotations

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_LINEAGE_APPLIED_IDS_METADATA_KEY,
)
from gigaevo.memory.core import LineageExcluder, NullExcluder


class _Prog:
    def __init__(self, lineage_applied):
        self._m = {MUTATION_MEMORY_LINEAGE_APPLIED_IDS_METADATA_KEY: lineage_applied}

    def get_metadata(self, key):
        return self._m.get(key)


def test_null_excluder_excludes_nothing():
    assert NullExcluder().exclude_for(_Prog(["a", "b"])) == frozenset()


def test_lineage_excluder_returns_the_closure():
    assert LineageExcluder().exclude_for(_Prog(["a", "b"])) == frozenset({"a", "b"})


def test_lineage_excluder_legacy_program_without_key_is_empty():
    class _Legacy:
        def get_metadata(self, key):
            return None

    assert LineageExcluder().exclude_for(_Legacy()) == frozenset()
