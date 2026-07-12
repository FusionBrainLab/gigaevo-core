"""Validator stage routing ``artifact["_program_metadata"]`` onto program.metadata.

Archive-replacement decisions are made by the archive selector at MAP-Elites
insertion — outside the DAG — so per-sample vectors must persist ON the
program. This stage pops the reserved namespace from the validator's artifact
at eval time: fields land in ``program.metadata`` and never enter the artifact
stream (formatters, prompts, stage-result payloads).
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from gigaevo.programs.core_types import ProgramStageResult, StageError
from gigaevo.programs.program import Program
from gigaevo.programs.stages.common import Box
from gigaevo.programs.stages.python_executors.execution import CallValidatorFunction
from gigaevo.programs.stages.stage_registry import StageRegistry

PROGRAM_METADATA_ARTIFACT_KEY = "_program_metadata"


def route_program_metadata(program: Program, artifact: Any) -> Any:
    """Pop the reserved namespace from ``artifact`` into ``program.metadata``.

    Returns the artifact minus the namespace; a fully consumed artifact
    becomes None so downstream formatters skip exactly as for dict-only
    validator returns. Namespace keys are validator-owned: a re-run refreshes
    them so the vector tracks the re-derived metrics instead of decohering
    (a stale vector trips the coherence guard and silently reverts the paired
    gate to the point rule for this program).
    """
    if not isinstance(artifact, dict) or PROGRAM_METADATA_ARTIFACT_KEY not in artifact:
        return artifact
    remaining = dict(artifact)
    namespace = remaining.pop(PROGRAM_METADATA_ARTIFACT_KEY)
    if not isinstance(namespace, dict):
        logger.warning(
            "[route_program_metadata] {}: dropping non-dict {!r} namespace (type={})",
            program.id[:8],
            PROGRAM_METADATA_ARTIFACT_KEY,
            type(namespace).__name__,
        )
        return remaining or None
    for key, value in namespace.items():
        if key in program.metadata:
            logger.debug(
                "[route_program_metadata] {}: refreshing metadata key {!r}",
                program.id[:8],
                key,
            )
        program.set_metadata(key, value)
    return remaining or None


@StageRegistry.register(
    description=(
        "CallValidatorFunction that persists artifact['_program_metadata'] "
        "fields to program.metadata (never into prompts)."
    )
)
class ProgramMetadataValidatorStage(CallValidatorFunction):
    """Drop-in ``CallValidatorFunction`` honoring the reserved namespace.

    ``validate()`` returns ``(metrics, {"_program_metadata": {...}, ...})``;
    the namespace is stripped here, before the tuple reaches any downstream
    stage or storage.
    """

    async def compute(self, program: Program) -> ProgramStageResult | Box[Any]:
        result = await super().compute(program)
        if isinstance(result, ProgramStageResult):
            return result
        if not (isinstance(result.data, tuple) and len(result.data) == 2):
            return ProgramStageResult.failure(
                error=StageError(
                    type="ValidatorContractError",
                    message=(
                        "expected (metrics, artifact) 2-tuple from validator, "
                        f"got {type(result.data).__name__!r}"
                    ),
                    stage=self.__class__.__name__,
                )
            )
        metrics, artifact = result.data
        return self.__class__.OutputModel(
            data=(metrics, route_program_metadata(program, artifact))
        )
