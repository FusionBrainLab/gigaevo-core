"""Optimization stages (CMA-ES, Optuna) and shared utilities."""

from gigaevo.programs.stages.optimization.cma import (
    CMANumericalOptimizationStage,
    CMAOptimizationOutput,
    _ConstantInfo,
    _extract_constants,
    _parameterize,
    _substitute,
)
from gigaevo.programs.stages.optimization.optuna import (
    OptunaOptimizationOutput,
    OptunaOptimizationStage,
    OptunaSearchSpace,
    ParamSpec,
    desubstitute_params,
)
from gigaevo.programs.stages.optimization.utils import (
    OptimizationInput,
    build_eval_code,
    evaluate_single,
    read_validator,
)

__all__ = [
    # CMA
    "CMANumericalOptimizationStage",
    "CMAOptimizationOutput",
    "_ConstantInfo",
    "_extract_constants",
    "_parameterize",
    "_substitute",
    # Optuna
    "OptunaOptimizationOutput",
    "OptunaOptimizationStage",
    "OptunaSearchSpace",
    "ParamSpec",
    "desubstitute_params",
    # Utils
    "OptimizationInput",
    "build_eval_code",
    "evaluate_single",
    "read_validator",
]
