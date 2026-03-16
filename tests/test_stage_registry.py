"""Tests for gigaevo/programs/stages/stage_registry.py — StageRegistry."""

from __future__ import annotations

from pydantic import Field

from gigaevo.programs.core_types import StageIO, VoidInput, VoidOutput
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.stage_registry import StageRegistry

# ---------------------------------------------------------------------------
# Test models & stages
# ---------------------------------------------------------------------------


class SimpleOutput(StageIO):
    value: int


class TypedInput(StageIO):
    name: str
    count: int = 0
    tags: list[str] = Field(default_factory=list)
    alias: str | None = None


class SimpleStage(Stage):
    InputsModel = VoidInput
    OutputModel = SimpleOutput

    async def compute(self, program):
        return SimpleOutput(value=1)


class TypedInputStage(Stage):
    InputsModel = TypedInput
    OutputModel = VoidOutput

    async def compute(self, program):
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStageRegistry:
    def setup_method(self):
        StageRegistry.clear()

    def test_register_basic(self) -> None:
        decorated = StageRegistry.register(description="A simple stage")(SimpleStage)
        assert decorated is SimpleStage
        info = StageRegistry.get_stage("SimpleStage")
        assert info is not None
        assert info.description == "A simple stage"
        assert info.output_model_name == "SimpleOutput"
        assert "value" in info.output_fields

    def test_register_extracts_input_types(self) -> None:
        StageRegistry.register(description="typed")(TypedInputStage)
        info = StageRegistry.get_stage("TypedInputStage")
        assert info is not None
        assert "name" in info.input_types
        assert info.input_types["name"] == "str"
        assert info.input_types["count"] == "int"
        assert "list" in info.input_types["tags"]
        # Check that alias field accepts None (modern X | None syntax shows as UnionType[str, None])
        assert (
            "Union" in info.input_types["alias"] or "None" in info.input_types["alias"]
        )

    def test_register_mandatory_and_optional(self) -> None:
        StageRegistry.register()(TypedInputStage)
        info = StageRegistry.get_stage("TypedInputStage")
        assert "name" in info.mandatory_inputs
        # Fields with defaults that are X | None (None default) are optional
        assert "alias" in info.optional_inputs

    def test_get_all_stages(self) -> None:
        StageRegistry.register()(SimpleStage)
        StageRegistry.register()(TypedInputStage)
        all_stages = StageRegistry.get_all_stages()
        assert "SimpleStage" in all_stages
        assert "TypedInputStage" in all_stages

    def test_get_stage_missing_returns_none(self) -> None:
        assert StageRegistry.get_stage("NonExistent") is None

    def test_clear_empties_registry(self) -> None:
        StageRegistry.register()(SimpleStage)
        assert len(StageRegistry.get_all_stages()) == 1
        StageRegistry.clear()
        assert len(StageRegistry.get_all_stages()) == 0

    def test_custom_import_path(self) -> None:
        StageRegistry.register(import_path="my.custom.path.SimpleStage")(SimpleStage)
        info = StageRegistry.get_stage("SimpleStage")
        assert info.import_path == "my.custom.path.SimpleStage"

    def test_auto_import_path(self) -> None:
        StageRegistry.register()(SimpleStage)
        info = StageRegistry.get_stage("SimpleStage")
        assert "SimpleStage" in info.import_path
