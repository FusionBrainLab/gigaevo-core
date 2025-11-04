class GigaEvoError(Exception):
    """Base for all GigaEvo exceptions."""

    pass


# High-level families
class ValidationError(GigaEvoError):
    """Data validation failures."""

    pass


class StorageError(GigaEvoError):
    """Storage operation failures."""

    pass


class ProgramError(GigaEvoError):
    """Program execution failures."""

    pass


class EvolutionError(GigaEvoError):
    """Evolution process failures."""

    pass


class SecurityError(GigaEvoError):
    """Security violations."""

    pass


class LLMError(GigaEvoError):
    """Base exception for LLM wrapper errors."""

    pass


# LLM subtypes
class LLMValidationError(LLMError):
    """Raised when LLM input validation fails."""

    pass


class LLMAPIError(LLMError):
    """Raised when LLM API calls fail after retries."""

    pass


# Stage / Program subtypes
class StageExecutionError(GigaEvoError):
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


class ResourceError(GigaEvoError):
    """Resource limit violations."""

    pass


class MutationError(GigaEvoError):
    """Mutation failures."""

    pass
