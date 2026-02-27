"""Tests for gigaevo/programs/program.py

Covers:
- Lineage: parent_count, child_count, add_child (dedup), is_root
- Program: UUID coercion, invalid UUID, from_dict roundtrip, from_dict with
  stage_results, create_child, create_child with no parents, from_mutation_spec,
  format_stage_error, format_errors, __hash__ and __eq__, is_root property,
  failed_stages property, is_failed property, is_complete property
"""

from __future__ import annotations

from unittest.mock import MagicMock
import uuid

import pytest

from gigaevo.programs.core_types import ProgramStageResult, StageError, StageState
from gigaevo.programs.program import (
    NO_STAGE_ERRORS_MSG,
    Lineage,
    Program,
)
from gigaevo.programs.program_state import ProgramState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_program(
    code: str = "def solve(): return 1",
    state: ProgramState = ProgramState.QUEUED,
) -> Program:
    return Program(code=code, state=state)


def _failed_result(stage: str = "TestStage") -> ProgramStageResult:
    return ProgramStageResult.failure(
        error=StageError(type="RuntimeError", message="boom", stage=stage)
    )


def _success_result() -> ProgramStageResult:
    return ProgramStageResult.success()


# ---------------------------------------------------------------------------
# Lineage tests
# ---------------------------------------------------------------------------


class TestLineage:
    def test_parent_count_empty(self) -> None:
        lin = Lineage()
        assert lin.parent_count == 0

    def test_parent_count_with_parents(self) -> None:
        lin = Lineage(parents=["a", "b", "c"])
        assert lin.parent_count == 3

    def test_child_count_empty(self) -> None:
        lin = Lineage()
        assert lin.child_count == 0

    def test_child_count_with_children(self) -> None:
        lin = Lineage(children=["x", "y"])
        assert lin.child_count == 2

    def test_add_child_appends_new_id(self) -> None:
        lin = Lineage()
        lin.add_child("child-1")
        assert "child-1" in lin.children
        assert lin.child_count == 1

    def test_add_child_does_not_duplicate(self) -> None:
        """Calling add_child with the same ID twice must not create a duplicate."""
        lin = Lineage()
        lin.add_child("child-1")
        lin.add_child("child-1")
        assert lin.children.count("child-1") == 1

    def test_add_child_multiple_distinct(self) -> None:
        lin = Lineage()
        lin.add_child("a")
        lin.add_child("b")
        assert lin.children == ["a", "b"]

    def test_is_root_true_when_no_parents(self) -> None:
        lin = Lineage()
        assert lin.is_root() is True

    def test_is_root_false_when_has_parents(self) -> None:
        lin = Lineage(parents=["parent-id"])
        assert lin.is_root() is False


# ---------------------------------------------------------------------------
# Program UUID coercion and validation
# ---------------------------------------------------------------------------


class TestProgramUUIDCoercion:
    def test_uuid_object_is_coerced_to_string(self) -> None:
        """Passing a uuid.UUID instance as id must be stored as its string form."""
        uid = uuid.uuid4()
        prog = Program(code="def f(): pass", id=uid)  # type: ignore[arg-type]
        assert prog.id == str(uid)
        assert isinstance(prog.id, str)

    def test_valid_uuid_string_accepted(self) -> None:
        uid_str = str(uuid.uuid4())
        prog = Program(code="def f(): pass", id=uid_str)
        assert prog.id == uid_str

    def test_invalid_uuid_string_raises_value_error(self) -> None:
        """A malformed UUID string must raise a ValueError (from Pydantic validation)."""
        with pytest.raises((ValueError, Exception)):
            Program(code="def f(): pass", id="not-a-valid-uuid")

    def test_auto_generated_id_is_valid_uuid(self) -> None:
        prog = _make_program()
        # Must parse without error
        uuid.UUID(prog.id)


# ---------------------------------------------------------------------------
# Program.from_dict roundtrip
# ---------------------------------------------------------------------------


class TestProgramFromDict:
    def test_roundtrip_basic(self) -> None:
        """to_dict followed by from_dict produces an equivalent Program."""
        prog = _make_program(state=ProgramState.DONE)
        prog.add_metrics({"score": 3.14})
        d = prog.to_dict()
        restored = Program.from_dict(d)
        assert restored.id == prog.id
        assert restored.code == prog.code
        assert restored.state == prog.state
        assert restored.metrics == prog.metrics

    def test_roundtrip_with_metadata(self) -> None:
        """Metadata serialized as pickle-base64 must survive the roundtrip."""
        prog = _make_program()
        prog.set_metadata("key", {"nested": [1, 2, 3]})
        d = prog.to_dict()
        restored = Program.from_dict(d)
        assert restored.get_metadata("key") == {"nested": [1, 2, 3]}

    def test_from_dict_with_stage_results_dict(self) -> None:
        """stage_results stored as nested dicts are reconstructed into ProgramStageResult."""
        prog = _make_program()
        prog.stage_results["stage_a"] = _success_result()
        d = prog.to_dict()
        # Verify stage_results in serialized form are dicts
        assert isinstance(d["stage_results"]["stage_a"], dict)

        restored = Program.from_dict(d)
        assert "stage_a" in restored.stage_results
        sr = restored.stage_results["stage_a"]
        assert isinstance(sr, ProgramStageResult)
        assert sr.status == StageState.COMPLETED

    def test_from_dict_with_failed_stage_result(self) -> None:
        prog = _make_program()
        prog.stage_results["bad_stage"] = _failed_result("bad_stage")
        d = prog.to_dict()

        restored = Program.from_dict(d)
        sr = restored.stage_results["bad_stage"]
        assert sr.status == StageState.FAILED
        assert sr.error is not None
        assert sr.error.type == "RuntimeError"


# ---------------------------------------------------------------------------
# Program.create_child
# ---------------------------------------------------------------------------


class TestCreateChild:
    def test_create_child_sets_lineage_parents(self) -> None:
        parent = _make_program()
        child = Program.create_child(parents=[parent], code="def f(): return 2")
        assert parent.id in child.lineage.parents

    def test_create_child_increments_generation(self) -> None:
        parent = _make_program()
        assert parent.lineage.generation == 1
        child = Program.create_child(parents=[parent], code="def f(): pass")
        assert child.lineage.generation == 2

    def test_create_child_generation_from_multiple_parents(self) -> None:
        """Generation is max(parent generations) + 1."""
        p1 = Program(code="def f(): pass", lineage=Lineage(generation=3))
        p2 = Program(code="def g(): pass", lineage=Lineage(generation=5))
        child = Program.create_child(parents=[p1, p2], code="def h(): pass")
        assert child.lineage.generation == 6

    def test_create_child_stores_mutation_string(self) -> None:
        parent = _make_program()
        child = Program.create_child(
            parents=[parent], code="def f(): pass", mutation="test_mutation"
        )
        assert child.lineage.mutation == "test_mutation"

    def test_create_child_stores_name(self) -> None:
        parent = _make_program()
        child = Program.create_child(
            parents=[parent], code="def f(): pass", name="experiment-1"
        )
        assert child.name == "experiment-1"

    def test_create_child_no_parents_raises_value_error(self) -> None:
        """create_child with an empty parents list must raise ValueError."""
        with pytest.raises(ValueError, match="At least one parent"):
            Program.create_child(parents=[], code="def f(): pass")

    def test_create_child_fresh_id(self) -> None:
        """Child gets its own unique ID, different from the parent's."""
        parent = _make_program()
        child = Program.create_child(parents=[parent], code="def f(): pass")
        assert child.id != parent.id


# ---------------------------------------------------------------------------
# Program.from_mutation_spec
# ---------------------------------------------------------------------------


class TestFromMutationSpec:
    def test_from_mutation_spec_basic(self) -> None:
        """from_mutation_spec creates a child with the correct code and lineage."""
        parent = _make_program()

        spec = MagicMock()
        spec.parents = [parent]
        spec.code = "def child(): return 99"
        spec.name = "my_mutation"
        spec.metadata = {}

        child = Program.from_mutation_spec(spec)
        assert child.code == "def child(): return 99"
        assert parent.id in child.lineage.parents
        assert child.lineage.mutation == "my_mutation"

    def test_from_mutation_spec_stores_metadata(self) -> None:
        """Metadata from spec is stored on the program via set_metadata."""
        parent = _make_program()

        spec = MagicMock()
        spec.parents = [parent]
        spec.code = "def f(): pass"
        spec.name = "m"
        spec.metadata = {"archetype": "refactor", "score": 0.9}

        child = Program.from_mutation_spec(spec)
        assert child.get_metadata("archetype") == "refactor"
        assert child.get_metadata("score") == 0.9

    def test_from_mutation_spec_no_metadata_skips_set(self) -> None:
        """Empty metadata dict must not cause errors."""
        parent = _make_program()

        spec = MagicMock()
        spec.parents = [parent]
        spec.code = "def f(): pass"
        spec.name = "m"
        spec.metadata = {}

        child = Program.from_mutation_spec(spec)
        assert child.metadata == {}


# ---------------------------------------------------------------------------
# format_stage_error / format_errors
# ---------------------------------------------------------------------------


class TestFormatErrors:
    def test_format_stage_error_returns_none_for_absent_stage(self) -> None:
        prog = _make_program()
        result = prog.format_stage_error(stage="nonexistent")
        assert result is None

    def test_format_stage_error_returns_none_for_successful_stage(self) -> None:
        prog = _make_program()
        prog.stage_results["ok_stage"] = _success_result()
        result = prog.format_stage_error(stage="ok_stage")
        assert result is None

    def test_format_stage_error_returns_string_for_failed_stage(self) -> None:
        prog = _make_program()
        prog.stage_results["bad"] = _failed_result("bad")
        result = prog.format_stage_error(stage="bad")
        assert result is not None
        assert "RuntimeError" in result
        assert "boom" in result

    def test_format_stage_error_with_traceback(self) -> None:
        prog = _make_program()
        prog.stage_results["bad"] = ProgramStageResult.failure(
            error=StageError(
                type="ValueError",
                message="oops",
                stage="bad",
                traceback="Traceback...",
            )
        )
        result = prog.format_stage_error(stage="bad", include_traceback=True)
        assert result is not None
        assert "Traceback" in result

    def test_format_errors_no_failures_returns_constant(self) -> None:
        prog = _make_program()
        prog.stage_results["ok"] = _success_result()
        assert prog.format_errors() == NO_STAGE_ERRORS_MSG

    def test_format_errors_with_no_stage_results_returns_constant(self) -> None:
        prog = _make_program()
        assert prog.format_errors() == NO_STAGE_ERRORS_MSG

    def test_format_errors_with_failure_contains_stage_name(self) -> None:
        prog = _make_program()
        prog.stage_results["failing_stage"] = _failed_result("failing_stage")
        result = prog.format_errors()
        assert "failing_stage" in result
        assert "RuntimeError" in result

    def test_format_errors_multiple_failures(self) -> None:
        prog = _make_program()
        prog.stage_results["stage_a"] = _failed_result("stage_a")
        prog.stage_results["stage_b"] = _failed_result("stage_b")
        result = prog.format_errors()
        assert "stage_a" in result
        assert "stage_b" in result


# ---------------------------------------------------------------------------
# __hash__ and __eq__
# ---------------------------------------------------------------------------


class TestHashAndEquality:
    def test_same_id_programs_are_equal(self) -> None:
        uid = str(uuid.uuid4())
        p1 = Program(code="def f(): pass", id=uid)
        p2 = Program(code="def g(): pass", id=uid)
        assert p1 == p2

    def test_different_id_programs_not_equal(self) -> None:
        p1 = _make_program()
        p2 = _make_program()
        assert p1 != p2

    def test_hash_is_id_based(self) -> None:
        uid = str(uuid.uuid4())
        p1 = Program(code="def f(): pass", id=uid)
        p2 = Program(code="def g(): pass", id=uid)
        assert hash(p1) == hash(p2)

    def test_programs_usable_in_sets(self) -> None:
        uid = str(uuid.uuid4())
        p1 = Program(code="def f(): pass", id=uid)
        p2 = Program(code="def g(): pass", id=uid)
        p3 = _make_program()
        s = {p1, p2, p3}
        assert len(s) == 2  # p1 and p2 are equal

    def test_not_equal_to_non_program(self) -> None:
        prog = _make_program()
        assert prog != "not a program"
        assert prog != 42


# ---------------------------------------------------------------------------
# is_root property
# ---------------------------------------------------------------------------


class TestIsRootProperty:
    def test_is_root_true_for_genesis_program(self) -> None:
        prog = _make_program()
        assert prog.is_root is True

    def test_is_root_false_for_child(self) -> None:
        parent = _make_program()
        child = Program.create_child(parents=[parent], code="def f(): pass")
        assert child.is_root is False


# ---------------------------------------------------------------------------
# failed_stages and is_failed
# ---------------------------------------------------------------------------


class TestFailedStagesProperties:
    def test_failed_stages_empty_when_no_results(self) -> None:
        prog = _make_program()
        assert prog.failed_stages == []

    def test_failed_stages_empty_when_all_success(self) -> None:
        prog = _make_program()
        prog.stage_results["ok"] = _success_result()
        assert prog.failed_stages == []

    def test_failed_stages_lists_failed_names(self) -> None:
        prog = _make_program()
        prog.stage_results["ok"] = _success_result()
        prog.stage_results["bad"] = _failed_result()
        assert "bad" in prog.failed_stages
        assert "ok" not in prog.failed_stages

    def test_is_failed_false_when_no_failures(self) -> None:
        prog = _make_program()
        prog.stage_results["ok"] = _success_result()
        assert prog.is_failed is False

    def test_is_failed_true_when_any_failure(self) -> None:
        prog = _make_program()
        prog.stage_results["ok"] = _success_result()
        prog.stage_results["bad"] = _failed_result()
        assert prog.is_failed is True

    def test_is_failed_false_when_empty_results(self) -> None:
        prog = _make_program()
        assert prog.is_failed is False


# ---------------------------------------------------------------------------
# is_complete property
# ---------------------------------------------------------------------------


class TestIsCompleteProperty:
    def test_queued_is_not_complete(self) -> None:
        prog = Program(code="def f(): pass", state=ProgramState.QUEUED)
        assert prog.is_complete is False

    def test_running_is_not_complete(self) -> None:
        prog = Program(code="def f(): pass", state=ProgramState.RUNNING)
        assert prog.is_complete is False

    def test_done_is_complete(self) -> None:
        prog = Program(code="def f(): pass", state=ProgramState.DONE)
        assert prog.is_complete is True

    def test_discarded_is_complete(self) -> None:
        """DISCARDED is a terminal state and therefore is_complete must be True."""
        prog = Program(code="def f(): pass", state=ProgramState.DISCARDED)
        assert prog.is_complete is True


# ---------------------------------------------------------------------------
# Audit Finding 1: Full round-trip serialization (all fields)
# ---------------------------------------------------------------------------


class TestFullFieldRoundTrip:
    """Audit finding 1: to_dict -> from_dict must preserve ALL Program fields."""

    def test_all_fields_roundtrip(self) -> None:
        """Create a Program with every field populated, serialize, deserialize,
        and verify every single field matches."""

        lineage = Lineage(
            parents=["00000000-0000-0000-0000-000000000001"],
            children=["00000000-0000-0000-0000-000000000002"],
            mutation="crossover_v2",
            generation=7,
        )
        stage_ok = _success_result()
        stage_fail = _failed_result("FailStage")
        metadata = {
            "experiment": "full-test",
            "nested": {"a": 1, "b": [2, 3]},
        }

        prog = Program(
            code="def solve(x): return x + 1",
            state=ProgramState.DONE,
            name="full-test-prog",
            lineage=lineage,
            metrics={"accuracy": 0.98, "f1": 0.95},
            metadata=metadata,
            stage_results={"validate": stage_ok, "optimize": stage_fail},
        )

        d = prog.to_dict()
        restored = Program.from_dict(d)

        # 1. id
        assert restored.id == prog.id
        # 2. code
        assert restored.code == prog.code
        # 3. name
        assert restored.name == prog.name
        # 4. state
        assert restored.state == prog.state
        # 5. metrics
        assert restored.metrics == prog.metrics
        # 6. metadata (nested)
        assert restored.metadata["experiment"] == "full-test"
        assert restored.metadata["nested"]["a"] == 1
        assert restored.metadata["nested"]["b"] == [2, 3]
        # 7. lineage
        assert restored.lineage.parents == lineage.parents
        assert restored.lineage.children == lineage.children
        assert restored.lineage.mutation == lineage.mutation
        assert restored.lineage.generation == lineage.generation
        # 8. stage_results
        assert restored.stage_results["validate"].status == StageState.COMPLETED
        assert restored.stage_results["optimize"].status == StageState.FAILED
        assert restored.stage_results["optimize"].error.type == "RuntimeError"
        # 9. created_at
        assert restored.created_at == prog.created_at
        # 10. atomic_counter
        assert restored.atomic_counter == prog.atomic_counter
        # 11. generation property
        assert restored.generation == 7

    def test_empty_optional_fields_roundtrip(self) -> None:
        """A Program with all optional fields at defaults round-trips correctly."""
        prog = Program(code="def f(): pass")

        d = prog.to_dict()
        restored = Program.from_dict(d)

        assert restored.id == prog.id
        assert restored.code == prog.code
        assert restored.name is None
        assert restored.state == ProgramState.QUEUED
        assert restored.metrics == {}
        assert restored.metadata == {}
        assert restored.stage_results == {}
        assert restored.lineage.parents == []
        assert restored.lineage.children == []
        assert restored.lineage.mutation is None
        assert restored.lineage.generation == 1
        assert restored.created_at == prog.created_at
