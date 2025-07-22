from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Set

from pydantic import BaseModel, Field, computed_field

from src.programs.program_state import ProgramState


class ValidationFailureReason(str, Enum):
    """Specific reasons why a program failed validation (mirrors previous monolith)."""

    MISSING_ID = "missing_id"
    MISSING_CODE = "missing_code"
    NOT_COMPLETE = "not_complete"
    IS_DISCARDED = "is_discarded"
    EMPTY_METRICS = "empty_metrics"
    MISSING_REQUIRED_KEYS = "missing_required_keys"
    VALIDATION_EXCEPTION = "validation_exception"


class ProgramValidationResult(BaseModel):
    """Detailed result of program validation with explicit contract information."""

    is_valid: bool = Field(description="Whether the program passed validation")
    program_id: str | None = Field(
        default=None, description="ID of the validated program"
    )
    reason: Optional[ValidationFailureReason] = Field(
        default=None, description="Specific reason for validation failure"
    )
    missing_keys: Optional[Set[str]] = Field(
        default=None, description="Required behavior keys that are missing"
    )
    available_keys: Optional[Set[str]] = Field(
        default=None, description="Behavior keys that are present in metrics"
    )
    metrics_count: int = Field(
        default=0, description="Number of metrics present in the program"
    )
    is_complete: bool = Field(
        default=False, description="Whether program completed DAG pipeline"
    )
    contract_summary: Optional[str] = Field(
        default=None, description="Short summary derived from validation result"
    )
    detailed_message: str | None = Field(
        default=None,
        description="Human-readable explanation of validation result",
        repr=False,
    )

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow",  # tolerate extra fields from legacy callers
    }

    @computed_field  # type: ignore[misc]
    @property
    def computed_contract_summary(self) -> str:
        """Generate a fallback contract summary if caller not provided one."""
        if self.contract_summary:
            return self.contract_summary
        if self.is_valid:
            return "✅ Program passes validation"
        return f"❌ Program invalid: {self.reason.value if self.reason else 'unknown'}"

    # ---- helpers -------------------------------------------------
    @classmethod
    def failure(
        cls,
        reason: ValidationFailureReason,
        summary: str,
        details: str | None = None,
    ) -> "ProgramValidationResult":
        return cls(
            is_valid=False,
            reason=reason,
            contract_summary=summary,
            detailed_message=details,
        )


# ---------------------------------------------------------------------------
# High-level validation helper (extracted from EvolutionEngine.core)
# ---------------------------------------------------------------------------

from typing import Any, Set  # after pydantic imports to avoid circular


def validate_program(
    program: "Program",  # forward ref to avoid heavy import cost
    *,
    required_behavior_keys: Set[str] | None = None,
) -> ProgramValidationResult:
    """Standalone program validation used by EvolutionEngine and tests."""

    from src.programs.program import Program  # local import to avoid cycles

    required_behavior_keys = required_behavior_keys or set()

    try:
        if not program.id:
            return ProgramValidationResult.failure(
                ValidationFailureReason.MISSING_ID,
                "❌ Program has no ID — indicates creation issue",
                "Program object is missing required 'id' field",
            )

        if not program.code:
            return ProgramValidationResult.failure(
                ValidationFailureReason.MISSING_CODE,
                "❌ Program has no code",
                f"Program {program.id} has empty or missing code field",
            )

        if program.state != ProgramState.DAG_PROCESSING_COMPLETED:
            return ProgramValidationResult.failure(
                ValidationFailureReason.NOT_COMPLETE,
                f"❌ Program has not completed DAG pipeline (state: {program.state})",
                f"Program {program.id} is in state '{program.state}' but needs to be in 'dag_processing_completed'",
            )

        if program.state == ProgramState.DISCARDED:
            return ProgramValidationResult.failure(
                ValidationFailureReason.IS_DISCARDED,
                "❌ Program explicitly marked as discarded",
                f"Program {program.id} has been marked as discarded and cannot participate in evolution",
            )

        if not program.metrics:
            return ProgramValidationResult.failure(
                ValidationFailureReason.EMPTY_METRICS,
                "❌ Program metrics empty",
                f"Program {program.id} has no metrics - likely DAG execution failed",
            )

        if required_behavior_keys:
            present = set(program.metrics.keys())
            missing = required_behavior_keys.difference(present)
            if missing:
                return ProgramValidationResult(
                    is_valid=False,
                    program_id=program.id,
                    reason=ValidationFailureReason.MISSING_REQUIRED_KEYS,
                    missing_keys=missing,
                    available_keys=present,
                    metrics_count=len(program.metrics),
                    contract_summary=f"❌ Program missing required behaviour keys: {sorted(missing)}",
                    detailed_message=f"Program {program.id} missing keys: {sorted(missing)}; present: {sorted(present)}",
                )

        return ProgramValidationResult(
            is_valid=True,
            program_id=program.id,
            metrics_count=len(program.metrics),
            is_complete=True,
            available_keys=set(program.metrics.keys()),
            contract_summary=f"metrics={len(program.metrics)} ✅ (has metrics - eligible for evolution)",
            detailed_message=f"Program {program.id} passed all validation checks",
        )

    except Exception as exc:  # pylint: disable=broad-except
        return ProgramValidationResult.failure(
            ValidationFailureReason.VALIDATION_EXCEPTION,
            f"❌ Unhandled exception during validation: {type(exc).__name__}",
            f"Program {getattr(program, 'id', 'unknown')} validation failed with exception: {exc}",
        )
