from __future__ import annotations

from gigaevo.evolution.engine.mutation import (
    applied_memory_ids,
    lineage_applied_closure,
)
from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_LINEAGE_APPLIED_IDS_METADATA_KEY,
)


class _FakeParent:
    def __init__(self, lineage_applied: list[str]) -> None:
        self._m = {MUTATION_MEMORY_LINEAGE_APPLIED_IDS_METADATA_KEY: lineage_applied}

    def get_metadata(self, key: str):
        return self._m.get(key)


def test_closure_unions_parent_lineage_with_child_applied_cards():
    parents = [_FakeParent(["a", "b"]), _FakeParent(["b", "c"])]
    assert lineage_applied_closure(applied_ids=["d"], parents=parents) == [
        "a",
        "b",
        "c",
        "d",
    ]


def test_root_no_parents_no_injection_is_empty():
    assert lineage_applied_closure(applied_ids=[], parents=[]) == []


def test_grandparent_card_survives_two_hops():
    parent = _FakeParent(["gp_card"])
    assert lineage_applied_closure(applied_ids=[], parents=[parent]) == ["gp_card"]


def test_missing_parent_metadata_is_treated_as_empty():
    class _Legacy:
        def get_metadata(self, key):
            return None

    assert lineage_applied_closure(applied_ids=["x"], parents=[_Legacy()]) == ["x"]


def test_applied_memory_ids_uses_only_structured_used_ids_from_injected_slate():
    assert applied_memory_ids(
        ["shown-a", "shown-b"],
        {"card_ids_used": ["shown-b", "hallucinated"]},
    ) == ["shown-b"]


def test_applied_memory_ids_does_not_treat_structured_missing_used_as_used():
    assert applied_memory_ids(["shown-a"], {"changes": []}) == []


def test_applied_memory_ids_legacy_without_structured_output_falls_back_to_injected():
    assert applied_memory_ids(["shown-a", "shown-b"], None) == ["shown-a", "shown-b"]
