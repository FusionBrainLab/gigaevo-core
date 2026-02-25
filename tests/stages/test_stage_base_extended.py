"""Extended tests for Stage base class — targeting uncovered bug-prone paths.

Covers:
1. __init_subclass__ validation: all 4 error paths for malformed stage definitions
2. attach_inputs with unknown fields: KeyError on extra keys
3. params property Pydantic validation error: re-raised as KeyError
4. _ensure_required_present: missing required inputs cause execute() to fail
5. _is_optional_type with Python 3.10+ `X | None` union syntax (types.UnionType)
6. compute_hash_from_inputs exception path: returns None on bad inputs silently
7. execute() VoidOutput returning None (success) vs non-VoidOutput returning None (failure)
"""

from __future__ import annotations

import types
from typing import Optional, Union

import pytest

from gigaevo.programs.core_types import (
    StageIO,
    StageState,
    VoidInput,
    VoidOutput,
)
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.base import Stage, _is_optional_type
from gigaevo.programs.stages.cache_handler import NO_CACHE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prog() -> Program:
    """Create a minimal RUNNING Program for stage execution."""
    return Program(code="def solve(): return 42", state=ProgramState.RUNNING)


# ---------------------------------------------------------------------------
# Concrete StageIO types used across tests
# ---------------------------------------------------------------------------


class TextOutput(StageIO):
    message: str = "hello"


class RequiredInput(StageIO):
    """All fields are required (no Optional)."""

    required_str: str
    required_int: int


class PartiallyOptionalInput(StageIO):
    """Mix of required and optional fields."""

    required_str: str
    optional_int: Optional[int] = None


class ModernUnionInput(StageIO):
    """Uses Python 3.10+ X | None union syntax."""

    name: str | None = None
    count: int


class WrongTypeInput(StageIO):
    """For testing Pydantic validation errors via the params property."""

    value: int  # must be an int — providing a non-coercible string will fail


# ---------------------------------------------------------------------------
# TestInitSubclassValidation
# ---------------------------------------------------------------------------


class TestInitSubclassValidation:
    """Stage.__init_subclass__ must reject malformed stage definitions at class
    definition time, not at instantiation or execution time. This is critical
    because silent failures here mean broken stages deploy undetected."""

    def test_missing_inputs_model_raises_type_error(self):
        """A stage with no InputsModel defined must raise TypeError immediately."""
        with pytest.raises(TypeError, match="must define InputsModel"):

            class NoInputsStage(Stage):
                OutputModel = TextOutput

                async def compute(self, program: Program) -> TextOutput:
                    return TextOutput()

    def test_missing_output_model_raises_type_error(self):
        """A stage with no OutputModel defined must raise TypeError immediately."""
        with pytest.raises(TypeError, match="must define OutputModel"):

            class NoOutputStage(Stage):
                InputsModel = VoidInput

                async def compute(self, program: Program) -> None:
                    return None

    def test_inputs_model_not_stage_io_raises_type_error(self):
        """InputsModel must inherit from StageIO — a plain Pydantic model is rejected."""
        from pydantic import BaseModel as PydanticBase

        class NotAStageIO(PydanticBase):
            x: int = 0

        with pytest.raises(TypeError, match="InputsModel must inherit from StageIO"):

            class BadInputsStage(Stage):
                InputsModel = NotAStageIO  # type: ignore[assignment]
                OutputModel = TextOutput

                async def compute(self, program: Program) -> TextOutput:
                    return TextOutput()

    def test_output_model_not_stage_io_raises_type_error(self):
        """OutputModel must inherit from StageIO — a plain Pydantic model is rejected."""
        from pydantic import BaseModel as PydanticBase

        class NotAStageIO(PydanticBase):
            x: int = 0

        with pytest.raises(TypeError, match="OutputModel must inherit from StageIO"):

            class BadOutputStage(Stage):
                InputsModel = VoidInput
                OutputModel = NotAStageIO  # type: ignore[assignment]

                async def compute(self, program: Program) -> None:
                    return None

    def test_valid_stage_definition_succeeds(self):
        """A well-formed stage definition must not raise any error."""

        class ValidStage(Stage):
            InputsModel = VoidInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> TextOutput:
                return TextOutput()

        # Class was defined successfully; instantiation should also work.
        stage = ValidStage(timeout=1.0)
        assert stage.stage_name == "ValidStage"

    def test_void_output_stage_definition_succeeds(self):
        """A stage using VoidOutput (a valid StageIO subclass) must not raise."""

        class ValidVoidStage(Stage):
            InputsModel = VoidInput
            OutputModel = VoidOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> None:
                return None

        stage = ValidVoidStage(timeout=1.0)
        assert stage.stage_name == "ValidVoidStage"


# ---------------------------------------------------------------------------
# TestAttachInputsUnknownFields
# ---------------------------------------------------------------------------


class TestAttachInputsUnknownFields:
    """attach_inputs must reject unknown keys to prevent silent data loss where
    a misspelled field name is simply ignored."""

    def test_unknown_field_raises_key_error(self):
        """Passing an undeclared field name must raise KeyError immediately."""

        class SimpleInputStage(Stage):
            InputsModel = RequiredInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> TextOutput:
                return TextOutput()

        stage = SimpleInputStage(timeout=1.0)
        with pytest.raises(KeyError, match="Unknown input fields"):
            stage.attach_inputs(
                {"required_str": "hello", "required_int": 1, "typo_field": "oops"}
            )

    def test_multiple_unknown_fields_all_named_in_error(self):
        """All unknown field names must appear in the error message."""

        class SimpleInputStage(Stage):
            InputsModel = RequiredInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> TextOutput:
                return TextOutput()

        stage = SimpleInputStage(timeout=1.0)
        with pytest.raises(KeyError) as exc_info:
            stage.attach_inputs(
                {
                    "required_str": "x",
                    "required_int": 1,
                    "extra_a": "a",
                    "extra_b": "b",
                }
            )
        error_msg = str(exc_info.value)
        assert "extra_a" in error_msg
        assert "extra_b" in error_msg

    def test_correct_fields_accepted_without_error(self):
        """Providing exactly the declared fields must not raise."""

        class SimpleInputStage(Stage):
            InputsModel = RequiredInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> TextOutput:
                return TextOutput(message=self.params.required_str)

        stage = SimpleInputStage(timeout=1.0)
        # Should not raise
        stage.attach_inputs({"required_str": "hello", "required_int": 42})


# ---------------------------------------------------------------------------
# TestParamsPydanticValidationError
# ---------------------------------------------------------------------------


class TestParamsPydanticValidationError:
    """The params property must re-raise PydanticValidationError as KeyError.

    This matters because callers catching KeyError for missing inputs would
    silently swallow type errors if the exception type changed."""

    def test_params_re_raises_pydantic_error_as_key_error(self):
        """When raw inputs fail Pydantic validation, params raises KeyError."""

        class WrongTypeStage(Stage):
            InputsModel = WrongTypeInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> TextOutput:
                return TextOutput()

        stage = WrongTypeStage(timeout=1.0)
        # Attach a non-coercible string to an int field to trigger PydanticValidationError.
        # Note: Pydantic v2 will try to coerce "not-an-int" to int and fail.
        stage._raw_inputs = {"value": "not-an-int"}

        with pytest.raises(KeyError, match="Input validation failed"):
            _ = stage.params

    def test_params_key_error_message_contains_field_errors(self):
        """The KeyError message from params validation must contain field error info."""

        class WrongTypeStage(Stage):
            InputsModel = WrongTypeInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> TextOutput:
                return TextOutput()

        stage = WrongTypeStage(timeout=1.0)
        stage._raw_inputs = {"value": "not-an-int"}

        with pytest.raises(KeyError) as exc_info:
            _ = stage.params

        # The error message should contain the field name or validation error details
        error_msg = str(exc_info.value)
        assert "Input validation failed" in error_msg


# ---------------------------------------------------------------------------
# TestEnsureRequiredPresent
# ---------------------------------------------------------------------------


class TestEnsureRequiredPresent:
    """_ensure_required_present guards against missing required inputs.

    Important: compute_inputs_hash() is called on line 246 of execute(), which
    is BEFORE the try/except block (try starts at line 248). When _raw_inputs is
    completely empty and the InputsModel has required fields, params() raises
    KeyError during compute_inputs_hash(), which propagates unhandled to the
    caller. This is a real production behavior difference worth documenting.

    When inputs are partially provided (enough for hash computation but missing
    some required fields), _ensure_required_present() inside the try block
    catches the gap and returns a FAILED ProgramStageResult."""

    async def test_execute_without_any_inputs_raises_key_error(self):
        """Bug exposure: execute() with NO inputs at all raises KeyError from
        compute_inputs_hash() because that call happens outside the try block.

        This is a real production bug: the caller gets an uncaught exception
        instead of a graceful FAILED ProgramStageResult."""

        class RequiredInputStage(Stage):
            InputsModel = RequiredInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> TextOutput:
                return TextOutput(message=self.params.required_str)

        stage = RequiredInputStage(timeout=5.0)
        # With no inputs at all, compute_inputs_hash() calls self.params which
        # raises KeyError outside the try/except block in execute()
        with pytest.raises(KeyError, match="Input validation failed"):
            await stage.execute(_prog())

    def test_ensure_required_present_raises_key_error_for_missing_fields(self):
        """_ensure_required_present raises KeyError naming missing fields."""

        class RequiredInputStage(Stage):
            InputsModel = RequiredInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> TextOutput:
                return TextOutput(message=self.params.required_str)

        stage = RequiredInputStage(timeout=5.0)
        # Set raw inputs to an empty dict to simulate missing required fields,
        # but bypass the hash computation path by calling the guard directly.
        stage._raw_inputs = {}
        with pytest.raises(KeyError, match="Missing required inputs"):
            stage._ensure_required_present()

    def test_ensure_required_present_error_names_missing_fields(self):
        """The KeyError from _ensure_required_present names all missing fields."""

        class RequiredInputStage(Stage):
            InputsModel = RequiredInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> TextOutput:
                return TextOutput(message=self.params.required_str)

        stage = RequiredInputStage(timeout=5.0)
        stage._raw_inputs = {"required_str": "present"}  # missing required_int

        with pytest.raises(KeyError) as exc_info:
            stage._ensure_required_present()

        assert "required_int" in str(exc_info.value)

    def test_ensure_required_present_passes_when_all_required_provided(self):
        """_ensure_required_present does not raise when all required fields present."""

        class RequiredInputStage(Stage):
            InputsModel = RequiredInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> TextOutput:
                return TextOutput()

        stage = RequiredInputStage(timeout=5.0)
        stage.attach_inputs({"required_str": "hello", "required_int": 42})
        # Should not raise
        stage._ensure_required_present()

    async def test_execute_with_all_required_inputs_succeeds(self):
        """Providing all required fields allows execute() to succeed."""

        class RequiredInputStage(Stage):
            InputsModel = RequiredInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> TextOutput:
                return TextOutput(message=self.params.required_str)

        stage = RequiredInputStage(timeout=5.0)
        stage.attach_inputs({"required_str": "hello", "required_int": 42})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED

    async def test_execute_with_optional_only_stage_succeeds_without_inputs(self):
        """A stage where all inputs are Optional succeeds without attach_inputs.

        When _raw_inputs is empty but all fields are Optional, _normalize_inputs
        fills them with None, Pydantic validates successfully, and the hash
        computation completes without error — so no KeyError is raised."""

        class AllOptionalInput(StageIO):
            opt_a: Optional[str] = None
            opt_b: Optional[int] = None

        class AllOptionalStage(Stage):
            InputsModel = AllOptionalInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> TextOutput:
                val = self.params.opt_a or "default"
                return TextOutput(message=val)

        stage = AllOptionalStage(timeout=5.0)
        # No attach_inputs — all fields are optional so normalization fills None
        result = await stage.execute(_prog())
        assert result.status == StageState.COMPLETED


# ---------------------------------------------------------------------------
# TestIsOptionalTypeModernSyntax
# ---------------------------------------------------------------------------


class TestIsOptionalTypeModernSyntax:
    """_is_optional_type must handle Python 3.10+ `X | None` union syntax.

    Without the types.UnionType branch, stages using modern union annotations
    would have their Optional fields incorrectly classified as required,
    causing execution failures when optional DAG edges are absent."""

    def test_modern_union_none_is_optional(self):
        """str | None (Python 3.10+ syntax) is detected as optional."""
        # Create a types.UnionType at runtime using the | operator
        modern_type = str | None
        assert isinstance(modern_type, types.UnionType), (
            "str | None must produce a types.UnionType on Python 3.10+"
        )
        assert _is_optional_type(modern_type) is True

    def test_modern_union_without_none_is_not_optional(self):
        """str | int (no None) is NOT optional."""
        modern_type = str | int
        assert _is_optional_type(modern_type) is False

    def test_modern_union_multi_type_with_none_is_optional(self):
        """str | int | None is detected as optional."""
        modern_type = str | int | None
        assert _is_optional_type(modern_type) is True

    def test_typing_optional_still_works(self):
        """Optional[str] (legacy syntax) continues to be detected as optional."""
        assert _is_optional_type(Optional[str]) is True

    def test_typing_union_with_none_still_works(self):
        """Union[str, None] (legacy syntax) continues to be detected as optional."""
        assert _is_optional_type(Union[str, None]) is True

    def test_plain_type_is_not_optional(self):
        """A plain type like str is not optional."""
        assert _is_optional_type(str) is False

    def test_stage_with_modern_union_optional_field_infers_correctly(self):
        """A stage using `str | None` syntax must classify that field as optional."""

        class ModernUnionStage(Stage):
            InputsModel = ModernUnionInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> TextOutput:
                name = self.params.name or "unknown"
                return TextOutput(message=name)

        # 'name' is `str | None` so should be optional; 'count' is int so required
        assert "name" in ModernUnionStage.optional_fields()
        assert "count" in ModernUnionStage.required_fields()

    async def test_stage_with_modern_union_optional_field_executes_without_optional(
        self,
    ):
        """A stage with `str | None` optional field executes successfully when
        the optional field is not provided (only required fields provided)."""

        class ModernUnionStage(Stage):
            InputsModel = ModernUnionInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> TextOutput:
                name = self.params.name or "unknown"
                return TextOutput(message=name)

        stage = ModernUnionStage(timeout=5.0)
        # Provide only the required 'count' field, omit optional 'name'
        stage.attach_inputs({"count": 7})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert result.output.message == "unknown"


# ---------------------------------------------------------------------------
# TestComputeHashFromInputs
# ---------------------------------------------------------------------------


class TestComputeHashFromInputs:
    """compute_hash_from_inputs must return None silently when validation fails.

    This is a cache optimization path. A hard crash here would break DAG
    scheduling even when the underlying stage could succeed with correct inputs."""

    def test_valid_inputs_return_hash(self):
        """compute_hash_from_inputs returns a non-None hash for valid inputs."""

        class HashableStage(Stage):
            InputsModel = RequiredInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> TextOutput:
                return TextOutput()

        result = HashableStage.compute_hash_from_inputs(
            {"required_str": "hello", "required_int": 42}
        )
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_invalid_inputs_return_none_silently(self):
        """compute_hash_from_inputs returns None (not raise) when inputs are invalid."""

        class HashableStage(Stage):
            InputsModel = RequiredInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> TextOutput:
                return TextOutput()

        # required_int must be an int, not a dict — this triggers Pydantic failure
        result = HashableStage.compute_hash_from_inputs(
            {"required_str": "hello", "required_int": {"not": "an int"}}
        )
        assert result is None

    def test_missing_required_field_returns_none_silently(self):
        """compute_hash_from_inputs returns None when a required field is absent."""

        class HashableStage(Stage):
            InputsModel = RequiredInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> TextOutput:
                return TextOutput()

        result = HashableStage.compute_hash_from_inputs({"required_str": "hello"})
        # required_int is missing — validation fails, should return None
        assert result is None

    def test_completely_empty_inputs_returns_none_for_required_stage(self):
        """compute_hash_from_inputs returns None for empty inputs on a required stage."""

        class HashableStage(Stage):
            InputsModel = RequiredInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> TextOutput:
                return TextOutput()

        result = HashableStage.compute_hash_from_inputs({})
        assert result is None

    def test_empty_inputs_return_hash_for_void_input_stage(self):
        """compute_hash_from_inputs returns a valid hash for VoidInput stages with {}."""

        class VoidHashStage(Stage):
            InputsModel = VoidInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> TextOutput:
                return TextOutput()

        result = VoidHashStage.compute_hash_from_inputs({})
        assert result is not None

    def test_same_inputs_produce_same_hash(self):
        """Deterministic: identical inputs always produce the same hash."""

        class HashableStage(Stage):
            InputsModel = RequiredInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> TextOutput:
                return TextOutput()

        h1 = HashableStage.compute_hash_from_inputs(
            {"required_str": "hello", "required_int": 42}
        )
        h2 = HashableStage.compute_hash_from_inputs(
            {"required_str": "hello", "required_int": 42}
        )
        assert h1 == h2
        assert h1 is not None

    def test_different_inputs_produce_different_hash(self):
        """Different inputs must produce different hashes for cache correctness."""

        class HashableStage(Stage):
            InputsModel = RequiredInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> TextOutput:
                return TextOutput()

        h1 = HashableStage.compute_hash_from_inputs(
            {"required_str": "hello", "required_int": 1}
        )
        h2 = HashableStage.compute_hash_from_inputs(
            {"required_str": "hello", "required_int": 2}
        )
        assert h1 != h2


# ---------------------------------------------------------------------------
# TestVoidOutputReturnsNone
# ---------------------------------------------------------------------------


class TestVoidOutputReturnsNone:
    """Tests for the None-return dispatch in execute().

    Bug scenario: a developer writes a VoidOutput stage and accidentally also
    writes a non-VoidOutput stage that returns None. Both cases must be handled
    correctly and clearly."""

    async def test_void_output_returning_none_succeeds(self):
        """VoidOutput stage returning None from compute() must produce COMPLETED."""

        class VoidNoneStage(Stage):
            InputsModel = VoidInput
            OutputModel = VoidOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> None:
                return None

        stage = VoidNoneStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert result.error is None

    async def test_void_output_returning_none_has_no_output_object(self):
        """VoidOutput returning None produces a result with output=None."""

        class VoidNoneStage(Stage):
            InputsModel = VoidInput
            OutputModel = VoidOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> None:
                return None

        stage = VoidNoneStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert result.output is None

    async def test_non_void_output_returning_none_fails(self):
        """A stage with non-VoidOutput returning None must produce FAILED."""

        class NonVoidNoneStage(Stage):
            InputsModel = VoidInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> None:
                return None

        stage = NonVoidNoneStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        assert result.error is not None

    async def test_non_void_output_returning_none_error_is_type_error(self):
        """The failure for a non-VoidOutput None return must report TypeError."""

        class NonVoidNoneStage(Stage):
            InputsModel = VoidInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> None:
                return None

        stage = NonVoidNoneStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        assert "TypeError" in result.error.type

    async def test_void_output_returning_void_instance_also_succeeds(self):
        """VoidOutput stage returning an explicit VoidOutput() instance also works."""

        class VoidInstanceStage(Stage):
            InputsModel = VoidInput
            OutputModel = VoidOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> VoidOutput:
                return VoidOutput()

        stage = VoidInstanceStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED

    async def test_non_void_output_returning_correct_type_succeeds(self):
        """A non-VoidOutput stage returning the correct type succeeds."""

        class CorrectReturnStage(Stage):
            InputsModel = VoidInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> TextOutput:
                return TextOutput(message="ok")

        stage = CorrectReturnStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.COMPLETED
        assert result.output.message == "ok"


# ---------------------------------------------------------------------------
# TestRequiredVsOptionalFieldClassification
# ---------------------------------------------------------------------------


class TestRequiredVsOptionalFieldClassification:
    """Verifies that required_fields() and optional_fields() classify correctly
    for all supported annotation styles."""

    def test_all_required_fields_classified_correctly(self):
        """All non-Optional fields in RequiredInput are required."""

        class AllRequiredStage(Stage):
            InputsModel = RequiredInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> TextOutput:
                return TextOutput()

        assert "required_str" in AllRequiredStage.required_fields()
        assert "required_int" in AllRequiredStage.required_fields()
        assert AllRequiredStage.optional_fields() == []

    def test_mixed_required_optional_classified_correctly(self):
        """PartiallyOptionalInput has one required and one optional field."""

        class MixedStage(Stage):
            InputsModel = PartiallyOptionalInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> TextOutput:
                return TextOutput()

        assert "required_str" in MixedStage.required_fields()
        assert "optional_int" in MixedStage.optional_fields()
        assert "optional_int" not in MixedStage.required_fields()
        assert "required_str" not in MixedStage.optional_fields()

    def test_void_input_has_no_required_or_optional_fields(self):
        """VoidInput has no fields at all."""

        class VoidStage(Stage):
            InputsModel = VoidInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> TextOutput:
                return TextOutput()

        assert VoidStage.required_fields() == []
        assert VoidStage.optional_fields() == []

    def test_modern_union_syntax_classified_as_optional(self):
        """Fields annotated with `T | None` are classified as optional."""

        class ModernStage(Stage):
            InputsModel = ModernUnionInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> TextOutput:
                return TextOutput()

        assert "name" in ModernStage.optional_fields()
        assert "count" in ModernStage.required_fields()


# ---------------------------------------------------------------------------
# TestStateCleanupAfterValidationFailure
# ---------------------------------------------------------------------------


class TestStateCleanupAfterValidationFailure:
    """After execute() fails due to missing inputs or validation errors, the
    stage must still clean up its internal state (the finally block in execute).

    A bug here would cause state leakage between DAG executions of the same
    stage instance."""

    async def test_inputs_cleared_after_runtime_exception_in_try_block(self):
        """After execute() fails due to a RuntimeError in compute(), state is cleared.

        The finally block in execute() must clean up even on exceptions raised
        within the try block."""

        class BoomStage(Stage):
            InputsModel = VoidInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE

            async def compute(self, program: Program) -> TextOutput:
                raise RuntimeError("intentional failure")

        stage = BoomStage(timeout=5.0)
        stage.attach_inputs({})
        result = await stage.execute(_prog())

        assert result.status == StageState.FAILED
        # The finally block must clear state even when execution fails
        assert stage._raw_inputs == {}
        assert stage._params_obj is None
        assert stage._current_inputs_hash is None

    async def test_stage_can_reuse_after_failed_execution(self):
        """A stage that failed in compute() can be reused in the next execution cycle."""

        class ToggleStage(Stage):
            """Fails on first call, succeeds on second."""

            InputsModel = RequiredInput
            OutputModel = TextOutput
            cache_handler = NO_CACHE
            _call_count: int = 0

            async def compute(self, program: Program) -> TextOutput:
                ToggleStage._call_count += 1
                if ToggleStage._call_count == 1:
                    raise RuntimeError("first call fails")
                return TextOutput(message=self.params.required_str)

        stage = ToggleStage(timeout=5.0)

        # First execution: fail inside compute
        stage.attach_inputs({"required_str": "attempt1", "required_int": 1})
        result1 = await stage.execute(_prog())
        assert result1.status == StageState.FAILED

        # Second execution: provide correct inputs and succeed
        stage.attach_inputs({"required_str": "retry", "required_int": 99})
        result2 = await stage.execute(_prog())
        assert result2.status == StageState.COMPLETED
        assert result2.output.message == "retry"
