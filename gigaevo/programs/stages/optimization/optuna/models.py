"""Pydantic models, constants, and type aliases for the Optuna stage.

Contains all data structures shared across the Optuna sub-package:
parameter specs, search-space proposals, stage config, and stage output.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from gigaevo.programs.core_types import StageIO

# ---------------------------------------------------------------------------
# Constants & type aliases
# ---------------------------------------------------------------------------

_PARAM_TYPES = Literal["float", "int", "log_float", "categorical"]

#: Union of all value types a parameter can hold.
_ParamValue = Union[float, int, str, bool]

#: Name of the params dict injected into the parameterized code.
_OPTUNA_PARAMS_NAME = "_optuna_params"

#: Default float precision for display and suggestion rounding.
_DEFAULT_PRECISION = 6

# ---------------------------------------------------------------------------
# LLM structured output models
# ---------------------------------------------------------------------------


class ParamSpec(BaseModel):
    """One independent tuneable parameter proposed by the LLM.

    Supports numeric parameters (float, int, log_float) as well as
    categorical parameters whose choices can be strings, booleans,
    or numbers -- enabling algorithm selection, method sweeps, and
    feature toggles.
    """

    name: str = Field(
        description=(
            "Short, snake_case identifier for this parameter "
            "(e.g. 'learning_rate', 'num_iterations', 'solver_method')."
        )
    )
    initial_value: _ParamValue = Field(
        description=(
            "The current / default value of this parameter.  "
            "Can be a number, string, or boolean."
        ),
    )
    param_type: _PARAM_TYPES = Field(
        description=(
            "Search-space type: 'float' for continuous, 'int' for discrete "
            "integer, 'log_float' for log-uniform continuous, "
            "'categorical' for a finite set of choices (numbers, strings, "
            "or booleans)."
        )
    )
    low: Optional[float] = Field(
        default=None,
        description="Lower bound (required for float / int / log_float).",
    )
    high: Optional[float] = Field(
        default=None,
        description="Upper bound (required for float / int / log_float).",
    )
    choices: Optional[list[_ParamValue]] = Field(
        default=None,
        description=(
            "List of allowed values (required for categorical).  "
            "Can include strings, booleans, and numbers."
        ),
    )
    reason: str = Field(
        description="One-sentence explanation of why this parameter is tuneable.",
    )

    @model_validator(mode="after")
    def _validate_search_space(self) -> ParamSpec:
        """Validate cross-field constraints and clamp bad LLM output."""
        if self.param_type in ("float", "int", "log_float"):
            if self.low is None or self.high is None:
                raise ValueError(
                    f"ParamSpec '{self.name}': low and high are required "
                    f"for param_type='{self.param_type}'"
                )
            if self.low > self.high:
                # Swap rather than reject — common LLM mistake.
                self.low, self.high = self.high, self.low
            if self.param_type == "log_float" and self.low <= 0:
                raise ValueError(
                    f"ParamSpec '{self.name}': log_float requires low > 0, "
                    f"got low={self.low}"
                )
            if isinstance(self.initial_value, (int, float)):
                iv = float(self.initial_value)
                if self.param_type == "log_float" and iv <= 0:
                    iv = self.low  # Clamp to lower bound for log scale.
                if iv < self.low or iv > self.high:
                    # Clamp rather than reject — LLM often proposes bounds that
                    # exclude the original value by a small margin.
                    iv = max(self.low, min(self.high, iv))
                if self.param_type == "int":
                    self.initial_value = int(round(iv))
                else:
                    self.initial_value = iv
        elif self.param_type == "categorical":
            if not self.choices:
                raise ValueError(
                    f"ParamSpec '{self.name}': choices must be non-empty "
                    "for param_type='categorical'"
                )
            if self.initial_value not in self.choices:
                # Fall back to first choice rather than hard-fail.
                self.initial_value = self.choices[0]
        return self


class CodeModification(BaseModel):
    """A single patch applied to a specific line range."""

    start_line: int = Field(
        description="The 1-indexed starting line number of the block to replace."
    )
    end_line: int = Field(
        description="The 1-indexed ending line number of the block to replace (inclusive)."
    )
    parameterized_snippet: str = Field(
        description=(
            "The replacement code block with _optuna_params references. Use relative "
            "indentation only: first line has no leading spaces; indent following lines "
            "relative to that (e.g. 4 spaces per nesting level)."
        )
    )


class OptunaSearchSpace(BaseModel):
    """Structured search-space proposal returned by the LLM.

    Contains parameter specifications and a list of patches to apply
    to the code.
    """

    parameters: list[ParamSpec] = Field(
        description="List of independent tuneable parameters.",
    )
    modifications: list[CodeModification] = Field(
        description="List of code patches to inject parameters.",
    )
    new_imports: list[str] = Field(
        default_factory=list,
        description=(
            "List of new import statements required by the parameters (e.g. "
            "'import numpy as np')."
        ),
    )
    reasoning: str = Field(
        description=(
            "Very brief (1-2 sentences) overall strategy: why these parameters matter and "
            "what trade-offs tuning them explores."
        ),
    )


# ---------------------------------------------------------------------------
# Stage configuration
# ---------------------------------------------------------------------------


class OptunaOptimizationConfig(BaseModel):
    """Configuration for Optuna features and sampler settings."""

    # Reproducibility
    random_state: Optional[int] = Field(
        default=None,
        description="Random seed for the sampler. Set for reproducible runs.",
    )

    # TPE sampler
    n_startup_trials: Optional[int] = Field(
        default=None,
        description="Number of random trials run before TPE (in addition to n_trials). Total runs = n_startup_trials + n_trials. If None, uses min(25, max(10, n_trials // 2)).",
    )
    multivariate: bool = Field(
        default=True,
        description="Use multivariate TPE to model parameter correlations.",
    )

    # Dynamic Feature Importance
    importance_freezing: bool = Field(
        default=True, description="Enable freezing of low-impact parameters."
    )
    importance_check_at: Optional[int] = Field(
        default=None,
        description="Number of trials after which to check importance (always enforced to be after TPE phase, i.e. >= n_startup_trials + 1). If None, uses total_trials // 3.",
    )
    min_trials_for_importance: int = Field(
        default=10, description="Minimum trials before importance check."
    )
    importance_threshold_ratio: float = Field(
        default=0.1,
        description="Ratio of average importance below which a parameter is frozen.",
    )
    importance_absolute_threshold: float = Field(
        default=0.01,
        description="Absolute importance value below which a parameter is frozen.",
    )

    # Early stopping
    early_stopping_patience: Optional[int] = Field(
        default=None,
        description="Stop optimization after this many consecutive trials without improvement. If None, no early stopping.",
    )


# ---------------------------------------------------------------------------
# Stage output
# ---------------------------------------------------------------------------


class OptunaOptimizationOutput(StageIO):
    """Output produced by :class:`OptunaOptimizationStage`."""

    optimized_code: str
    best_scores: dict[str, float]
    best_params: dict[str, Any]
    n_params: int
    n_trials: int
    search_space_summary: list[dict[str, Any]]
