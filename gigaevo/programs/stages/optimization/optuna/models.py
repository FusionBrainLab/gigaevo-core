"""Pydantic models, constants, and type aliases for the Optuna stage.

Contains all data structures shared across the Optuna sub-package:
parameter specs, search-space proposals, stage config, and stage output.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from gigaevo.programs.core_types import StageIO

# ---------------------------------------------------------------------------
# Constants & type aliases
# ---------------------------------------------------------------------------

#: Minimum random startup trials before TPE begins.
_MIN_STARTUP_TRIALS: int = 10

#: Maximum random startup trials before TPE begins.
_MAX_STARTUP_TRIALS: int = 25

#: Hard floor: minimum completed trials before any importance check fires.
#: PED-ANOVA needs >=5 trials in its top quantile bucket; with target_quantile=0.25
#: that requires ceil(5/0.25)=20 trials. Matches Optuna team guidance.
_IMPORTANCE_CHECK_MIN_TRIALS: int = 20

#: Minimum post-startup (TPE-phase) trials before any importance check.
#: TPE needs several observations beyond startup to build a useful model.
_MIN_POST_STARTUP_TRIALS: int = 15

#: Minimum number of parameters required to run importance freezing.
_MIN_PARAMS_FOR_IMPORTANCE: int = 3

#: Log progress every N completed trials.
_PROGRESS_LOG_INTERVAL: int = 10

#: Maximum number of failure reasons shown when all trials fail.
_MAX_FAILURE_REASONS: int = 5

#: Minimum gap between early and late importance checks.
_MIN_IMPORTANCE_CHECK_GAP: int = 5


def default_n_startup_trials(n_trials: int) -> int:
    """Compute the default number of random startup trials for TPE.

    Formula: ``min(_MAX_STARTUP_TRIALS, max(_MIN_STARTUP_TRIALS, n_trials // 2))``.
    """
    return min(_MAX_STARTUP_TRIALS, max(_MIN_STARTUP_TRIALS, n_trials // 2))


def default_max_params(n_trials: int) -> int:
    """Auto-compute max parameters from trial budget.

    Formula: ``min(5, max(3, n_trials // 15))``.
    At 60 trials (85 total) → 4 params (~21 trials each).
    """
    return min(5, max(3, n_trials // 15))


def compute_eval_timeout(
    baseline_s: float | None,
    *,
    budget: float,
    min_timeout: float = 30.0,
    safety_mult: float = 3.0,
) -> float:
    """Derive per-trial eval timeout from measured baseline runtime.

    When *baseline_s* is known, the timeout is ``baseline_s * safety_mult``
    clamped to ``[min_timeout, budget / 2]``.  When unknown, falls back to
    ``budget * 0.05`` (5 % of total budget) clamped to the same range.
    """
    if baseline_s is not None and baseline_s > 0:
        raw = baseline_s * safety_mult
    else:
        raw = budget * 0.05
    return max(min_timeout, min(raw, budget / 2))


def compute_n_trials(
    budget: float,
    eval_timeout: float,
    max_parallel: int,
    *,
    llm_overhead: float = 60.0,
    min_trials: int = 20,
    max_trials: int = 100,
) -> int:
    """Derive number of TPE trials from remaining budget.

    Estimates how many sequential rounds fit into
    ``budget - llm_overhead``, multiplied by ``max_parallel``, then subtracts
    the startup trials (which are additional).  Result is clamped to
    ``[min_trials, max_trials]``.
    """
    usable = max(0.0, budget - llm_overhead)
    if eval_timeout <= 0:
        return min_trials
    n_rounds = math.floor(usable / eval_timeout)
    raw = n_rounds * max_parallel
    return max(min_trials, min(raw, max_trials))


_PARAM_TYPES = Literal["float", "int", "log_float", "categorical"]

#: Union of all value types a parameter can hold.
_ParamValue = Union[float, int, str, bool]

#: Name of the params dict injected into the parameterized code.
_OPTUNA_PARAMS_NAME = "_optuna_params"

#: Default float precision for display and suggestion rounding.
_DEFAULT_PRECISION = 6

#: Floor for log_float low bound — log scale requires low > 0.
_LOG_FLOAT_EPSILON: float = 1e-10

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
        description=(
            "One sentence: why this parameter will have high impact on the "
            "score, and why it was selected over alternatives."
        ),
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
                # Clamp rather than reject — common LLM mistake.
                self.low = _LOG_FLOAT_EPSILON
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
            # Coerce int-like strings ("3" → 3) so Optuna returns ints directly
            self.choices = [
                int(c) if isinstance(c, str) and c.strip().lstrip("-").isdigit() else c
                for c in self.choices
            ]
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
            "Selection rationale: which candidates you considered, why the "
            "chosen ones rank highest for score impact, and what you excluded."
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
        description=(
            "Number of random trials run before TPE (in addition to n_trials). "
            "Total runs = n_startup_trials + n_trials. If None, uses "
            "default_n_startup_trials(n_trials)."
        ),
    )
    multivariate: bool = Field(
        default=True,
        description="Use multivariate TPE to model parameter correlations.",
    )

    # Dynamic Feature Importance (two-phase)
    importance_freezing: bool = Field(
        default=True, description="Enable freezing of low-impact parameters."
    )
    early_tpe_fraction: float = Field(
        default=1 / 3,
        description=(
            "Fraction of TPE trials at which the early importance check fires "
            "(~33% into TPE by default)."
        ),
    )
    late_tpe_fraction: float = Field(
        default=3 / 4,
        description=(
            "Fraction of TPE trials at which the late importance check fires "
            "(~75% into TPE by default)."
        ),
    )
    early_threshold_multiplier: float = Field(
        default=0.5,
        description=(
            "Multiplier applied to importance_threshold_ratio for the early check. "
            "Early check is conservative: 0.5x the standard threshold."
        ),
    )
    ped_anova_early_quantile: float = Field(
        default=0.25,
        description="PED-ANOVA target_quantile for the early check (more inclusive for stability).",
    )
    ped_anova_late_quantile: float = Field(
        default=0.10,
        description="PED-ANOVA target_quantile for the late check (standard selectivity).",
    )
    max_params: Optional[int] = Field(
        default=None,
        description=(
            "Maximum parameters the LLM should propose. "
            "If None, auto-computed as min(5, max(3, n_trials // 15))."
        ),
    )
    importance_check_at: Optional[int] = Field(
        default=None,
        description=(
            "Trial count for the EARLY importance check. If None, auto-computed "
            "as n_startup + max(_MIN_POST_STARTUP_TRIALS, n_tpe // 3). "
            "Uses a conservative threshold (0.5x standard)."
        ),
    )
    importance_check_late_at: Optional[int] = Field(
        default=None,
        description=(
            "Trial count for the LATE importance check. If None, auto-computed "
            "as n_startup + max(_MIN_POST_STARTUP_TRIALS + 5, n_tpe * 3 // 4). "
            "Uses the standard threshold."
        ),
    )
    min_trials_for_importance: int = Field(
        default=20,
        description=(
            "Minimum completed trials in the study before importance is evaluated. "
            "PED-ANOVA needs sufficient top-quantile observations for reliability."
        ),
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
        description=(
            "Stop optimization after this many consecutive trials without "
            "improvement. If None, no early stopping."
        ),
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
    best_program_output: Any | None = None
