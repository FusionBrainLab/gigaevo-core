"""LLM-guided Optuna hyperparameter optimization stage.

Re-exports for backward compatibility -- all external import sites
(``optimization/__init__.py``, tests, entrypoint) continue working
unchanged.
"""

from gigaevo.programs.stages.optimization.optuna.desubstitution import (
    desubstitute_params,
)
from gigaevo.programs.stages.optimization.optuna.models import (
    _OPTUNA_PARAMS_NAME,
    CodeModification,
    OptunaOptimizationConfig,
    OptunaOptimizationOutput,
    OptunaSearchSpace,
    ParamSpec,
)
from gigaevo.programs.stages.optimization.optuna.stage import OptunaOptimizationStage

__all__ = [
    "CodeModification",
    "OptunaOptimizationConfig",
    "OptunaOptimizationOutput",
    "OptunaOptimizationStage",
    "OptunaSearchSpace",
    "ParamSpec",
    "_OPTUNA_PARAMS_NAME",
    "desubstitute_params",
]
