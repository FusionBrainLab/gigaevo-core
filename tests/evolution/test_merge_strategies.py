"""Tests for gigaevo/database/merge_strategies.py"""

import pytest

from gigaevo.database.merge_strategies import (
    _merge_dict_by_prog_ts,
    _merge_lineage,
    merge_programs,
    resolve_merge_strategy,
)
from gigaevo.programs.program import Lineage, Program
from gigaevo.programs.program_state import ProgramState


def _make_prog(
    code="def solve(): return 1",
    state=ProgramState.QUEUED,
    atomic_counter=0,
    **kwargs,
):
    return Program(code=code, state=state, atomic_counter=atomic_counter, **kwargs)


class TestMergeDictByProgTs:
    def test_higher_counter_overwrites(self):
        curr = _make_prog(atomic_counter=1)
        inc = _make_prog(atomic_counter=2)
        result = _merge_dict_by_prog_ts(
            {"a": 1, "b": 2}, {"a": 10, "c": 3}, current_prog=curr, incoming_prog=inc
        )
        assert result == {"a": 10, "b": 2, "c": 3}

    def test_lower_counter_only_adds_new(self):
        curr = _make_prog(atomic_counter=5)
        inc = _make_prog(atomic_counter=1)
        result = _merge_dict_by_prog_ts(
            {"a": 1}, {"a": 99, "b": 2}, current_prog=curr, incoming_prog=inc
        )
        assert result == {"a": 1, "b": 2}

    def test_empty_incoming(self):
        curr = _make_prog(atomic_counter=1)
        inc = _make_prog(atomic_counter=2)
        result = _merge_dict_by_prog_ts(
            {"a": 1}, {}, current_prog=curr, incoming_prog=inc
        )
        assert result == {"a": 1}

    def test_empty_current(self):
        curr = _make_prog(atomic_counter=1)
        inc = _make_prog(atomic_counter=2)
        result = _merge_dict_by_prog_ts(
            {}, {"a": 1}, current_prog=curr, incoming_prog=inc
        )
        assert result == {"a": 1}

    def test_both_empty(self):
        curr = _make_prog(atomic_counter=1)
        inc = _make_prog(atomic_counter=2)
        result = _merge_dict_by_prog_ts({}, {}, current_prog=curr, incoming_prog=inc)
        assert result == {}

    def test_equal_counter_incoming_loses(self):
        """When atomic_counter is equal, incoming does NOT win (not strictly greater)."""
        curr = _make_prog(atomic_counter=5)
        inc = _make_prog(atomic_counter=5)
        result = _merge_dict_by_prog_ts(
            {"a": 1, "b": 2},
            {"a": 99, "c": 3},
            current_prog=curr,
            incoming_prog=inc,
        )
        # Equal counter -> prefer_incoming is False -> current "a" kept, new "c" added
        assert result == {"a": 1, "b": 2, "c": 3}


class TestMergeLineage:
    def test_children_union_deduped(self):
        curr = Lineage(parents=["p1"], children=["c1", "c2"])
        inc = Lineage(parents=["p1"], children=["c2", "c3"])
        result = _merge_lineage(curr, inc)
        assert result.children == ["c1", "c2", "c3"]

    def test_current_order_preserved(self):
        curr = Lineage(parents=["p1"], children=["c2", "c1"])
        inc = Lineage(parents=["p1"], children=["c3"])
        result = _merge_lineage(curr, inc)
        assert result.children == ["c2", "c1", "c3"]

    def test_immutables_from_current(self):
        curr = Lineage(parents=["p1"], mutation="mutA", generation=3)
        inc = Lineage(parents=["p2"], mutation="mutB", generation=5)
        result = _merge_lineage(curr, inc)
        assert result.parents == ["p1"]
        assert result.mutation == "mutA"
        assert result.generation == 3

    def test_empty_children(self):
        curr = Lineage(parents=["p1"], children=[])
        inc = Lineage(parents=["p1"], children=["c1"])
        result = _merge_lineage(curr, inc)
        assert result.children == ["c1"]


class TestMergePrograms:
    def test_none_current_returns_incoming_copy(self):
        inc = _make_prog(code="def solve(): return 42")
        result = merge_programs(None, inc)
        assert result.code == inc.code
        assert result.id == inc.id

    def test_id_mismatch_raises(self):
        curr = _make_prog()
        inc = _make_prog()
        assert curr.id != inc.id
        with pytest.raises(ValueError, match="id mismatch"):
            merge_programs(curr, inc)

    def test_state_merged(self):
        p_id = _make_prog().id
        curr = _make_prog(id=p_id, state=ProgramState.QUEUED, atomic_counter=1)
        inc = _make_prog(id=p_id, state=ProgramState.RUNNING, atomic_counter=2)
        result = merge_programs(curr, inc)
        assert result.state == ProgramState.RUNNING

    def test_code_from_higher_counter(self):
        p_id = _make_prog().id
        curr = _make_prog(id=p_id, code="old_code = 1", atomic_counter=1)
        inc = _make_prog(id=p_id, code="new_code = 2", atomic_counter=5)
        result = merge_programs(curr, inc)
        assert result.code == "new_code = 2"

    def test_code_from_lower_counter_keeps_current(self):
        p_id = _make_prog().id
        curr = _make_prog(id=p_id, code="current_code = 1", atomic_counter=10)
        inc = _make_prog(id=p_id, code="old_code = 2", atomic_counter=1)
        result = merge_programs(curr, inc)
        assert result.code == "current_code = 1"

    def test_metrics_merged(self):
        p_id = _make_prog().id
        curr = _make_prog(id=p_id, atomic_counter=1)
        curr.add_metrics({"a": 1.0})
        inc = _make_prog(id=p_id, atomic_counter=2)
        inc.add_metrics({"b": 2.0})
        result = merge_programs(curr, inc)
        assert result.metrics["a"] == 1.0
        assert result.metrics["b"] == 2.0

    def test_metadata_merged(self):
        p_id = _make_prog().id
        curr = _make_prog(id=p_id, atomic_counter=1, metadata={"x": 1})
        inc = _make_prog(id=p_id, atomic_counter=2, metadata={"y": 2})
        result = merge_programs(curr, inc)
        assert result.metadata["x"] == 1
        assert result.metadata["y"] == 2

    def test_lineage_merged(self):
        p_id = _make_prog().id
        curr = _make_prog(id=p_id, atomic_counter=1)
        curr.lineage.children.append("c1")
        inc = _make_prog(id=p_id, atomic_counter=2)
        inc.lineage.children.append("c2")
        result = merge_programs(curr, inc)
        assert "c1" in result.lineage.children
        assert "c2" in result.lineage.children

    def test_name_mismatch_raises(self):
        p_id = _make_prog().id
        curr = _make_prog(id=p_id, name="alpha", atomic_counter=1)
        inc = _make_prog(id=p_id, name="beta", atomic_counter=2)
        with pytest.raises(ValueError, match="name mismatch"):
            merge_programs(curr, inc)

    def test_name_filled_from_incoming_when_current_is_none(self):
        """When current.name is None and incoming has a name, incoming name is kept.

        Bug (Finding 18): `updates["name"] = current.name` silently dropped the
        incoming name when current.name was None. The docstring said "fill name if
        current is None" but the code didn't do it.
        """
        p_id = _make_prog().id
        curr = _make_prog(id=p_id, name=None, atomic_counter=1)
        inc = _make_prog(id=p_id, name="discovered_name", atomic_counter=2)
        result = merge_programs(curr, inc)
        assert result.name == "discovered_name", (
            "incoming name should be preserved when current.name is None"
        )

    def test_name_kept_from_current_when_both_have_same_name(self):
        """When both programs have the same name, it is kept."""
        p_id = _make_prog().id
        curr = _make_prog(id=p_id, name="shared", atomic_counter=1)
        inc = _make_prog(id=p_id, name="shared", atomic_counter=2)
        result = merge_programs(curr, inc)
        assert result.name == "shared"

    def test_name_kept_from_current_when_incoming_has_no_name(self):
        """When incoming.name is None, current name is preserved."""
        p_id = _make_prog().id
        curr = _make_prog(id=p_id, name="keep_me", atomic_counter=1)
        inc = _make_prog(id=p_id, name=None, atomic_counter=2)
        result = merge_programs(curr, inc)
        assert result.name == "keep_me"

    def test_stage_results_merged(self):
        """Verify stage_results dict is merged like metrics."""
        from gigaevo.programs.core_types import ProgramStageResult, StageState

        p_id = _make_prog().id
        sr_a = ProgramStageResult(status=StageState.COMPLETED)
        sr_b = ProgramStageResult(status=StageState.COMPLETED)

        curr = _make_prog(id=p_id, atomic_counter=1, stage_results={"stage_a": sr_a})
        inc = _make_prog(id=p_id, atomic_counter=2, stage_results={"stage_b": sr_b})
        result = merge_programs(curr, inc)
        assert "stage_a" in result.stage_results
        assert "stage_b" in result.stage_results

    def test_equal_counter_keeps_current_code(self):
        """When counters are equal, current code is kept."""
        p_id = _make_prog().id
        curr = _make_prog(id=p_id, code="current_code = 1", atomic_counter=5)
        inc = _make_prog(id=p_id, code="incoming_code = 2", atomic_counter=5)
        result = merge_programs(curr, inc)
        assert result.code == "current_code = 1"

    def test_merge_returns_independent_copy(self):
        """Modifying merge result should not affect originals."""
        p_id = _make_prog().id
        curr = _make_prog(id=p_id, atomic_counter=1)
        curr.add_metrics({"a": 1.0})
        inc = _make_prog(id=p_id, atomic_counter=2)
        inc.add_metrics({"b": 2.0})
        result = merge_programs(curr, inc)
        # Modify result
        result.metrics["a"] = 999.0
        result.metrics["new_key"] = 42.0
        # Originals unchanged
        assert curr.metrics["a"] == 1.0
        assert "new_key" not in curr.metrics
        assert "new_key" not in inc.metrics


class TestResolveMergeStrategy:
    def test_additive(self):
        fn = resolve_merge_strategy("additive")
        assert fn is merge_programs

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown merge_strategy"):
            resolve_merge_strategy("unknown")
