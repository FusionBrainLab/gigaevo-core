class MetaEvolveError(Exception):
    """Base for all MetaEvolve exceptions."""

    pass


# High-level families
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


# LLM subtypes
class LLMValidationError(LLMError):
    """Raised when LLM input validation fails."""

    pass


class LLMAPIError(LLMError):
    """Raised when LLM API calls fail after retries."""

    pass


# Stage / Program subtypes
class StageExecutionError(MetaEvolveError):
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
