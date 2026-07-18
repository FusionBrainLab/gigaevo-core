"""Pre-retrieval excluders: null control and the lineage gate."""

from __future__ import annotations

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_LINEAGE_BLOCKED_IDS_METADATA_KEY,
)
from gigaevo.memory.read.exclusion import CardExcluder, LineageExcluder, NullExcluder


class _Program:
    def __init__(self, metadata: dict | None = None) -> None:
        self._metadata = metadata or {}

    def get_metadata(self, key: str):
        return self._metadata.get(key)


def test_null_excluder_excludes_nothing():
    assert NullExcluder().exclude_for(_Program()) == frozenset()


def test_lineage_excluder_reads_blocked_closure():
    program = _Program(
        {MUTATION_MEMORY_LINEAGE_BLOCKED_IDS_METADATA_KEY: ["m1", "m2", "m1"]}
    )
    assert LineageExcluder().exclude_for(program) == frozenset({"m1", "m2"})


def test_lineage_excluder_empty_without_stamp():
    assert LineageExcluder().exclude_for(_Program()) == frozenset()


def test_excluders_satisfy_protocol():
    assert isinstance(NullExcluder(), CardExcluder)
    assert isinstance(LineageExcluder(), CardExcluder)
