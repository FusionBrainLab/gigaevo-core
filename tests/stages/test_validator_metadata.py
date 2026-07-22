"""Tests for artifact-aware ProgramMetadataValidatorStage.

Contract: ``validate()`` may return ``(metrics, {"_program_metadata": {...}})``;
the stage pops the namespace at eval time so fields land on
``program.metadata`` and the emitted artifact never contains them — a fully
consumed artifact becomes None (prompt parity with dict-only returns).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from gigaevo.programs.core_types import ProgramStageResult, StageError, StageState
from gigaevo.programs.metrics.evaluation import (
    EVALUATION_MEASUREMENTS_ARTIFACT_KEY,
    EVALUATION_MEASUREMENTS_METADATA_KEY,
)
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import AnyContainer
from gigaevo.programs.stages.python_executors.execution import CallValidatorFunction
from gigaevo.programs.stages.validator_metadata import (
    PROGRAM_METADATA_ARTIFACT_KEY,
    ProgramMetadataValidatorStage,
    route_evaluation_measurements,
    route_program_metadata,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prog(**metadata) -> Program:
    return Program(
        code="def solve(): return 42",
        state=ProgramState.RUNNING,
        metadata=metadata,
    )


def _stage(tmp_path, validator_source: str) -> ProgramMetadataValidatorStage:
    validator = tmp_path / "validate.py"
    validator.write_text(validator_source)
    stage = ProgramMetadataValidatorStage(path=validator, timeout=30.0)
    stage.__class__.cache_handler = NO_CACHE
    return stage


async def _run(stage, program, payload):
    stage.attach_inputs({"payload": AnyContainer(data=payload), "context": None})
    return await stage.execute(program)


# ---------------------------------------------------------------------------
# route_program_metadata — namespace semantics
# ---------------------------------------------------------------------------


class TestRouteProgramMetadata:
    def test_moves_namespace_fields_to_metadata(self):
        program = _prog()
        artifact = {
            PROGRAM_METADATA_ARTIFACT_KEY: {"per_sample_scores": [0.1, 0.2]},
            "feedback": "two claims failed",
        }
        remaining = route_program_metadata(program, artifact)

        assert program.metadata["per_sample_scores"] == [0.1, 0.2]
        assert remaining == {"feedback": "two claims failed"}

    def test_fully_consumed_artifact_becomes_none(self):
        program = _prog()
        artifact = {PROGRAM_METADATA_ARTIFACT_KEY: {"per_sample_scores": [0.1]}}
        assert route_program_metadata(program, artifact) is None

    def test_dict_without_namespace_passes_through(self):
        program = _prog()
        artifact = {"feedback": "text"}
        assert route_program_metadata(program, artifact) == {"feedback": "text"}
        assert program.metadata == {}

    def test_none_passes_through(self):
        assert route_program_metadata(_prog(), None) is None

    def test_non_dict_passes_through(self):
        assert route_program_metadata(_prog(), "plain text") == "plain text"

    def test_reeval_refreshes_existing_metadata(self):
        # The reserved namespace is validator-owned: a re-run must refresh the
        # vector so it never decoheres from the re-derived metrics (a stale
        # vector would trip the coherence guard and silently revert the gate
        # to the point rule for this program).
        program = _prog(per_sample_scores=[9.9])
        artifact = {PROGRAM_METADATA_ARTIFACT_KEY: {"per_sample_scores": [0.1]}}
        assert route_program_metadata(program, artifact) is None
        assert program.metadata["per_sample_scores"] == [0.1]

    def test_non_dict_namespace_is_dropped_not_leaked(self):
        program = _prog()
        artifact = {PROGRAM_METADATA_ARTIFACT_KEY: 42, "feedback": "text"}
        remaining = route_program_metadata(program, artifact)

        assert program.metadata == {}
        assert remaining == {"feedback": "text"}

    def test_rejects_measurements_in_untyped_program_namespace(self):
        program = _prog()
        artifact = {
            PROGRAM_METADATA_ARTIFACT_KEY: {
                EVALUATION_MEASUREMENTS_METADATA_KEY: {"fitness": {"se": 0.1}}
            }
        }
        with pytest.raises(ValueError, match="is reserved"):
            route_program_metadata(program, artifact)


class TestRouteEvaluationMeasurements:
    def test_normalizes_sample_sd_and_strips_namespace(self):
        program = _prog()
        artifact = {
            EVALUATION_MEASUREMENTS_ARTIFACT_KEY: {
                "fitness": {
                    "sample_sd": 0.12,
                    "n": 4,
                    "method": "cross_validation",
                }
            },
            "feedback": "keep me",
        }

        remaining = route_evaluation_measurements(program, {"fitness": 0.75}, artifact)

        assert remaining == {"feedback": "keep me"}
        assert program.metadata[EVALUATION_MEASUREMENTS_METADATA_KEY] == {
            "fitness": {
                "value": 0.75,
                "sample_sd": 0.12,
                "n": 4,
                "method": "cross_validation",
            }
        }

    def test_accepts_direct_standard_error(self):
        program = _prog()
        route_evaluation_measurements(
            program,
            {"fitness": 0.75},
            {
                EVALUATION_MEASUREMENTS_ARTIFACT_KEY: {
                    "fitness": {"se": 0.03, "method": "bootstrap"}
                }
            },
        )
        assert program.metadata[EVALUATION_MEASUREMENTS_METADATA_KEY]["fitness"] == {
            "value": 0.75,
            "se": 0.03,
            "method": "bootstrap",
        }

    def test_absence_clears_stale_measurement(self):
        program = _prog(
            evaluation_measurements={
                "fitness": {"value": 0.5, "se": 0.1, "method": "old"}
            }
        )
        assert route_evaluation_measurements(program, {"fitness": 0.6}, None) is None
        assert EVALUATION_MEASUREMENTS_METADATA_KEY not in program.metadata

    def test_rejects_sample_sd_without_replicates(self):
        with pytest.raises(ValueError, match="sample_sd requires n >= 2"):
            route_evaluation_measurements(
                _prog(),
                {"fitness": 0.75},
                {
                    EVALUATION_MEASUREMENTS_ARTIFACT_KEY: {
                        "fitness": {
                            "sample_sd": 0.1,
                            "n": 1,
                            "method": "cross_validation",
                        }
                    }
                },
            )


# ---------------------------------------------------------------------------
# ProgramMetadataValidatorStage — stage behavior
# ---------------------------------------------------------------------------

_VECTOR_VALIDATOR = """
def validate(payload):
    scores = [0.4, 0.6, 0.5]
    return {
        "fitness": sum(scores) / len(scores),
        "is_valid": 1,
    }, {"_program_metadata": {"per_sample_scores": scores}}
"""

_DICT_VALIDATOR = """
def validate(payload):
    return {"fitness": 0.5, "is_valid": 1}
"""

_MEASUREMENT_VALIDATOR = """
def validate(payload):
    return {"fitness": 0.5, "is_valid": 1}, {
        "_evaluation_measurements": {
            "fitness": {
                "sample_sd": 0.1,
                "n": 4,
                "method": "cross_validation",
            }
        },
        "feedback": "visible",
    }
"""


class TestProgramMetadataValidatorStage:
    async def test_end_to_end_vector_lands_on_metadata(self, tmp_path):
        program = _prog()
        stage = _stage(tmp_path, _VECTOR_VALIDATOR)
        result = await _run(stage, program, "payload")

        assert result.status == StageState.COMPLETED
        metrics, artifact = result.output.data
        assert metrics["fitness"] == 0.5
        assert artifact is None
        assert program.metadata["per_sample_scores"] == [0.4, 0.6, 0.5]

    async def test_dict_only_validator_is_untouched(self, tmp_path):
        program = _prog()
        stage = _stage(tmp_path, _DICT_VALIDATOR)
        result = await _run(stage, program, "payload")

        assert result.status == StageState.COMPLETED
        metrics, artifact = result.output.data
        assert metrics == {"fitness": 0.5, "is_valid": 1}
        assert artifact is None
        assert program.metadata == {}

    async def test_measurement_lands_on_metadata_and_not_artifact(self, tmp_path):
        program = _prog()
        result = await _run(
            _stage(tmp_path, _MEASUREMENT_VALIDATOR), program, "payload"
        )

        assert result.status == StageState.COMPLETED
        metrics, artifact = result.output.data
        assert metrics["fitness"] == 0.5
        assert artifact == {"feedback": "visible"}
        assert program.metadata[EVALUATION_MEASUREMENTS_METADATA_KEY] == {
            "fitness": {
                "value": 0.5,
                "sample_sd": 0.1,
                "n": 4,
                "method": "cross_validation",
            }
        }

    async def test_failure_result_passes_through_untouched(self, tmp_path):
        program = _prog()
        stage = _stage(tmp_path, _VECTOR_VALIDATOR)
        failure = ProgramStageResult.failure(
            error=StageError(
                type="SubprocessError",
                message="boom",
                stage="ProgramMetadataValidatorStage",
            )
        )

        with patch.object(CallValidatorFunction, "compute", return_value=failure):
            result = await _run(stage, program, "payload")

        assert result.status == StageState.FAILED
        assert program.metadata == {}
