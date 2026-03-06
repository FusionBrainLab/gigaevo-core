from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from gigaevo.programs.metrics.context import VALIDITY_KEY, MetricSpec


class ParameterSpec(BaseModel):
    """Specification for a function parameter."""

    name: str = Field(description="Parameter name")
    type_hint: str | None = Field(
        default=None,
        description="Type hint string (e.g., 'np.ndarray', 'dict[str, float]')",
    )
    description: str | None = Field(
        default=None,
        description="Parameter description for docstring",
    )
    default: str | None = Field(
        default=None,
        description="Default value expression (if any)",
    )


class ReturnSpec(BaseModel):
    """Specification for return value."""

    type_hint: str | None = Field(
        default=None,
        description="Return type hint (e.g., 'dict[str, float]')",
    )
    description: str | None = Field(
        default=None,
        description="Human-readable description of return value",
    )
    fields: dict[str, str] | None = Field(
        default=None,
        description="For dict returns: mapping of key -> type/description",
    )


class FunctionSignature(BaseModel):
    """Function signature with structured parameter and return specs."""

    params: list[ParameterSpec] = Field(
        default_factory=list,
        description="List of parameter specifications",
    )
    returns: ReturnSpec | None = Field(
        default=None,
        description="Return value specification",
    )

    def get_param_names(self) -> list[str]:
        """Extract parameter names."""
        return [p.name for p in self.params]

    def get_param_string(self, with_types: bool = True) -> str:
        """Generate function parameter string with optional type hints."""
        parts = []
        for p in self.params:
            if with_types and p.type_hint:
                parts.append(f"{p.name}: {p.type_hint}")
            else:
                parts.append(p.name)
        return ", ".join(parts)


class HelperFunctionSpec(BaseModel):
    """Specification for a helper function stub."""

    name: str = Field(description="Function name")
    description: str = Field(description="What this helper does")
    signature: FunctionSignature = Field(description="Function signature")


class ContextSpec(BaseModel):
    """Specification for build_context() return value."""

    description: str | None = Field(
        default=None,
        description="What this context contains",
    )
    fields: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of context key -> type/description",
    )


class UtilsImportSpec(BaseModel):
    """Specification for importing functions from problem type's utils.py."""

    functions: list[str] = Field(
        description="Function names to import. Use ['*'] for wildcard import."
    )

    @model_validator(mode="after")
    def _validate_functions(self) -> UtilsImportSpec:
        """Validate function list is non-empty and wildcard usage."""
        if not self.functions:
            raise ValueError("functions must contain at least one name or '*'")
        if "*" in self.functions and len(self.functions) > 1:
            raise ValueError("'*' cannot be combined with other function names")
        return self


class UtilsConfig(BaseModel):
    """Configuration for utils imports across generated files."""

    validator: UtilsImportSpec | None = Field(
        default=None,
        description="Utils imports for validate.py",
    )
    helper: UtilsImportSpec | None = Field(
        default=None,
        description="Utils imports for helper.py",
    )
    context: UtilsImportSpec | None = Field(
        default=None,
        description="Utils imports for context.py",
    )
    initial_programs: UtilsImportSpec | None = Field(
        default=None,
        description="Utils imports for initial_programs/*.py",
    )


class TaskDescription(BaseModel):
    """Task description with optional hints and metadata."""

    objective: str = Field(
        description="Problem description with objective, rules, goals"
    )
    hints: list[str] | None = Field(
        default=None,
        description="Optional strategy hints",
    )
    constraints: list[str] | None = Field(
        default=None,
        description="List of constraints for the solution",
    )
    failure_modes: list[str] | None = Field(
        default=None,
        description="Common errors to avoid",
    )
    output_shape: str | None = Field(
        default=None,
        description="Expected output shape (e.g., '(11, 2) NumPy array')",
    )
    fitness_goal: str | None = Field(
        default=None,
        description="Fitness target (e.g., 'min_area ≥ 0.0365')",
    )
    complexity_notes: str | None = Field(
        default=None,
        description="Notes about problem complexity and search landscape",
    )
    validation_notes: list[str] | None = Field(
        default=None,
        description="Critical validation and efficiency requirements",
    )


class InitialProgram(BaseModel):
    """Specification for an initial/seed program."""

    name: str = Field(description="Program filename (without .py)")
    description: str = Field(description="Strategy description")


class ProblemConfigValidator:
    """Centralized validation for ProblemConfig."""

    @classmethod
    def validate_all(cls, config: ProblemConfig) -> list[str]:
        """Run all validations, return list of error messages."""
        errors: list[str] = []
        errors.extend(cls._validate_metrics(config))
        errors.extend(cls._validate_context_signatures(config))
        errors.extend(cls._validate_helper_config(config))
        errors.extend(cls._validate_context_config(config))
        return errors

    @staticmethod
    def _validate_metrics(config: ProblemConfig) -> list[str]:
        """Validate metrics structure."""
        errors = []
        if VALIDITY_KEY in config.metrics:
            errors.append(f"'{VALIDITY_KEY}' is auto-generated. Remove it from config.")
        primary_count = sum(1 for s in config.metrics.values() if s.is_primary)
        if primary_count != 1:
            errors.append(f"Exactly one metric must be primary, found {primary_count}")
        return errors

    @staticmethod
    def _validate_context_signatures(config: ProblemConfig) -> list[str]:
        """Validate function signatures match add_context setting."""
        if not config.add_context:
            return []
        errors = []
        if "context" not in config.validation.get_param_names():
            errors.append("add_context=True requires 'context' in validation params.")
        if "context" not in config.entrypoint.get_param_names():
            errors.append("add_context=True requires 'context' in entrypoint params.")
        return errors

    @staticmethod
    def _validate_helper_config(config: ProblemConfig) -> list[str]:
        """Validate helper configuration."""
        if config.helper_functions and not config.add_helper:
            return ["helper_functions specified but add_helper=False."]
        return []

    @staticmethod
    def _validate_context_config(config: ProblemConfig) -> list[str]:
        """Validate context configuration."""
        if config.context_spec and not config.add_context:
            return ["context_spec specified but add_context=False."]
        return []


class ProblemConfig(BaseModel):
    """Complete problem specification for scaffolding."""

    name: str = Field(description="Problem directory name")
    description: str = Field(description="Short problem description")

    entrypoint: FunctionSignature
    validation: FunctionSignature

    metrics: dict[str, MetricSpec] = Field(
        description="Metric specifications (is_valid auto-generated, do NOT include)"
    )

    task_description: TaskDescription

    add_context: bool = Field(default=False, description="Generate context.py")
    add_helper: bool = Field(default=False, description="Generate helper.py")

    initial_programs: list[InitialProgram] = Field(
        default_factory=list,
        description="Initial seed programs",
    )

    helper_functions: list[HelperFunctionSpec] | None = Field(
        default=None,
        description="Helper function specifications (if add_helper=True)",
    )
    context_spec: ContextSpec | None = Field(
        default=None,
        description="Context specification (if add_context=True)",
    )
    utils_imports: UtilsConfig | None = Field(
        default=None,
        description="Utils imports configuration for generated files",
    )

    @model_validator(mode="after")
    def _run_validations(self) -> ProblemConfig:
        """Run all config validations."""
        errors = ProblemConfigValidator.validate_all(self)
        if errors:
            raise ValueError("\n".join(errors))
        return self
