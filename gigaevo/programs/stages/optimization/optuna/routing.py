"""Routing stages for Optuna program-output bypass.

When Optuna succeeds it already has the best program output — these stages
let the DAG skip the expensive ``CallProgramFunction`` re-execution and
forward the captured output directly to ``CallValidatorFunction``.

``OptunaPayloadBridge``
    Extracts ``best_program_output`` from the Optuna output and wraps it
    in ``Box[Any]`` (same type as ``CallProgramFunction.OutputModel``).
    When Optuna fails, the required data dep makes this stage IMPOSSIBLE →
    SKIPPED automatically by the DAG automata.

``PayloadResolver``
    Picks whichever payload source completed — the bridge (Optuna path) or
    ``CallProgramFunction`` (fallback path).
"""

from __future__ import annotations

from typing import Any, cast

from gigaevo.programs.core_types import StageIO
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.common import AnyContainer, Box
from gigaevo.programs.stages.optimization.optuna.models import (
    OptunaOptimizationOutput,
)
from gigaevo.programs.stages.stage_registry import StageRegistry

# ------------------------------------------------------------------
# OptunaPayloadBridge
# ------------------------------------------------------------------


class OptunaPayloadBridgeInput(StageIO):
    """Required input: the Optuna stage output."""

    optuna_output: OptunaOptimizationOutput


@StageRegistry.register(
    description="Extract best_program_output from Optuna and wrap as Box[Any]"
)
class OptunaPayloadBridge(Stage):
    """Forward Optuna's captured program output into the DAG data flow.

    When Optuna FAILS the required ``optuna_output`` data dependency becomes
    IMPOSSIBLE and the DAG automata automatically SKIP this stage, which is
    the desired behaviour — the fallback path via ``CallProgramFunction``
    takes over.
    """

    InputsModel = OptunaPayloadBridgeInput
    OutputModel = Box[Any]

    async def compute(self, program: Program) -> Box[Any]:
        po = cast(
            OptunaPayloadBridgeInput, self.params
        ).optuna_output.best_program_output
        if po is None:
            raise ValueError("No program output captured from Optuna best trial")
        return Box[Any](data=po)


# ------------------------------------------------------------------
# PayloadResolver
# ------------------------------------------------------------------


class PayloadResolverInput(StageIO):
    """Both inputs optional — exactly one should be non-None at runtime."""

    optuna_payload: AnyContainer | None = None
    program_payload: AnyContainer | None = None


@StageRegistry.register(
    description="Pick whichever payload source completed (Optuna bridge or CallProgramFunction)"
)
class PayloadResolver(Stage):
    """Select the payload for ``CallValidatorFunction``.

    Prefers ``optuna_payload`` (from the bridge) when available; falls back
    to ``program_payload`` (from ``CallProgramFunction``).
    """

    InputsModel = PayloadResolverInput
    OutputModel = Box[Any]

    async def compute(self, program: Program) -> Box[Any]:
        params = cast(PayloadResolverInput, self.params)
        if params.optuna_payload is not None:
            return Box[Any](data=params.optuna_payload.data)
        if params.program_payload is not None:
            return Box[Any](data=params.program_payload.data)
        raise ValueError("No payload source available")
