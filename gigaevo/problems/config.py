from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from gigaevo.programs.metrics.context import VALIDITY_KEY, MetricSpec


class FunctionSignature(BaseModel):
    """Function signature specification.
    Function names are fixed: 'validate' or 'entrypoint'.
    """

    params: list[str] = Field(
        default_factory=list,
        description="List of parameter names (empty list for no parameters)",
    )
    returns: str | None = Field(
        default=None,
        description="Human-readable description of return value (for documentation)",
    )
    inputs: str | None = Field(
        default=None,
        description="Human-readable description of input parameters (for documentation)",
    )


class TaskDescription(BaseModel):
    """Task description with optional hints."""

    objective: str = Field(
        description="Problem description with objective, rules, goals"
    )
    hints: list[str] | None = Field(default=None, description="Optional strategy hints")


class InitialProgram(BaseModel):
    """Specification for an initial/seed program."""

    name: str = Field(description="Program filename (without .py)")
    description: str = Field(description="Strategy description")


class ProblemConfig(BaseModel):
    """Complete problem specification for scaffolding."""

    name: str = Field(description="Problem directory name")
    description: str = Field(description="Short problem description")

    entrypoint: FunctionSignature
    validation: FunctionSignature

    metrics: dict[str, MetricSpec] = Field(
        description="Metric specifications (is_valid auto-generated, do NOT include)"
    )
    display_order: list[str] = Field(
        default_factory=list,
        description="Metric display order (defaults to metric definition order if not specified)",
    )

    task_description: TaskDescription

    add_context: bool = Field(default=False, description="Generate context.py")
    add_helper: bool = Field(default=False, description="Generate helper.py")

    initial_programs: list[InitialProgram] = Field(
        default_factory=list,
        description="Initial seed programs",
    )

    @model_validator(mode="after")
    def _validate_metrics(self) -> ProblemConfig:
        """Validate metrics structure."""
        if VALIDITY_KEY in self.metrics:
            raise ValueError(
                f"'{VALIDITY_KEY}' is auto-generated. Remove it from config."
            )

        primary_count = sum(1 for s in self.metrics.values() if s.is_primary)
        if primary_count != 1:
            raise ValueError(
                f"Exactly one metric must be primary, found {primary_count}"
            )

        if self.display_order:
            unknown = set(self.display_order) - set(self.metrics.keys())
            if unknown:
                raise ValueError(f"display_order has unknown metrics: {unknown}")

        return self

    @model_validator(mode="after")
    def _validate_context_signatures(self) -> ProblemConfig:
        """Validate function signatures match add_context setting."""
        if self.add_context:
            if "context" not in self.validation.params:
                raise ValueError(
                    f"add_context=True requires 'context' in validation params. "
                    f"Got: {self.validation.params}"
                )
            if "context" not in self.entrypoint.params:
                raise ValueError(
                    f"add_context=True requires 'context' in entrypoint params. "
                    f"Got: {self.entrypoint.params}"
                )

        return self
