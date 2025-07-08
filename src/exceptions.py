"""Clean exception hierarchy with Pydantic validation.

Focused on critical error cases without over-engineering.
"""

from datetime import datetime, timezone
from typing import Any, Dict, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ErrorContext(BaseModel):
    """Context information for errors."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error_id: str = Field(default="")
    context: Dict[str, Any] = Field(default_factory=dict)


class MetaEvolveError(Exception):
    """Base exception with structured context."""

    def __init__(self, message: str, **context):
        super().__init__(message)
        self.message = message
        self.context = ErrorContext(context=context)


# Critical error types
class ValidationError(MetaEvolveError):
    """Data validation failures."""

    pass


class StorageError(MetaEvolveError):
    """Storage operation failures."""

    pass


class ProgramError(MetaEvolveError):
    """Program execution failures."""

    pass


class EvolutionError(MetaEvolveError):
    """Evolution process failures."""

    pass


class SecurityError(MetaEvolveError):
    """Security violations."""

    pass


class LLMError(MetaEvolveError):
    """Base exception for LLM wrapper errors."""

    pass


class LLMValidationError(LLMError):
    """Raised when LLM input validation fails."""

    pass


class LLMAPIError(LLMError):
    """Raised when LLM API calls fail after retries."""

    pass


# Stage-specific errors
class StageError(MetaEvolveError):
    """Stage execution failures."""

    pass


class ProgramValidationError(ProgramError):
    """Program validation failures."""

    pass


class ProgramExecutionError(ProgramError):
    """Program execution failures."""

    pass


class ProgramTimeoutError(ProgramError):
    """Program timeout failures."""

    pass


class SecurityViolationError(SecurityError):
    """Security violations in program execution."""

    pass


class ResourceError(MetaEvolveError):
    """Resource limit violations."""

    pass


class MutationError(MetaEvolveError):
    """Mutation failures."""

    pass


# Quick validation helpers
def ensure_not_none(value: T, name: str) -> T:
    if value is None:
        raise ValidationError(f"{name} cannot be None")
    return value


def ensure_positive(value: float, name: str) -> float:
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")
    return value


def ensure_non_negative(value: float, name: str) -> float:
    if value < 0:
        raise ValidationError(f"{name} must be non-negative, got {value}")
    return value
