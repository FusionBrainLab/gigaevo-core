from __future__ import annotations

from gigaevo.evolution.engine.mutation import (
    applied_memory_ids,
    lineage_blocked_closure,
    proposed_probe_ids,
)
from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_ASSIGNMENT_METADATA_KEY,
    MUTATION_MEMORY_LINEAGE_BLOCKED_IDS_METADATA_KEY,
)
from gigaevo.memory.cards import AssignmentRecord, DecisionContext


class _FakeParent:
    def __init__(
        self,
        lineage_blocked: list[str] | None = None,
        assignment: AssignmentRecord | None = None,
    ) -> None:
        self._m: dict[str, object] = {
            MUTATION_MEMORY_LINEAGE_BLOCKED_IDS_METADATA_KEY: lineage_blocked or []
        }
        if assignment is not None:
            self._m[MUTATION_MEMORY_ASSIGNMENT_METADATA_KEY] = assignment.model_dump(
                mode="json"
            )

    def get_metadata(self, key: str):
        return self._m.get(key)


def test_closure_unions_parent_lineage_with_child_blocked_cards():
    parents = [_FakeParent(["a", "b"]), _FakeParent(["b", "c"])]
    assert lineage_blocked_closure(blocked_ids=["d"], parents=parents) == [
        "a",
        "b",
        "c",
        "d",
    ]


def test_root_no_parents_no_injection_is_empty():
    assert lineage_blocked_closure(blocked_ids=[], parents=[]) == []


def test_grandparent_card_survives_two_hops():
    parent = _FakeParent(["gp_card"])
    assert lineage_blocked_closure(blocked_ids=[], parents=[parent]) == ["gp_card"]


def test_missing_parent_metadata_is_treated_as_empty():
    class _WithoutMetadata:
        def get_metadata(self, key):
            return None

    assert lineage_blocked_closure(blocked_ids=["x"], parents=[_WithoutMetadata()]) == [
        "x"
    ]


def test_both_randomized_arms_block_the_proposed_card() -> None:
    parents = [
        _FakeParent(
            assignment=AssignmentRecord(
                decision_id=f"decision-{arm}",
                policy_version="test",
                task_key="task",
                arm="injected" if arm == "treated" else "none",
                probe_arm=arm,
                randomized=True,
                propensity_kind="probe_bernoulli",
                propensities={f"card-{arm}": 0.5},
                ope_eligible=True,
                context=DecisionContext(),
            )
        )
        for arm in ("treated", "control")
    ]

    assert proposed_probe_ids(parents) == ["card-control", "card-treated"]


def test_applied_memory_ids_uses_only_structured_used_ids_from_injected_slate():
    assert applied_memory_ids(
        ["shown-a", "shown-b"],
        {"card_ids_used": ["shown-b", "hallucinated"]},
    ) == ["shown-b"]


def test_applied_memory_ids_does_not_treat_structured_missing_used_as_used():
    assert applied_memory_ids(["shown-a"], {"changes": []}) == []


def test_applied_memory_ids_without_grounded_output_credits_nothing():
    assert applied_memory_ids(["shown-a", "shown-b"], None) == []
