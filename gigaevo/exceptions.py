class GigaEvoError(Exception):
    """Base for all GigaEvo exceptions."""


# High-level families
class ValidationError(GigaEvoError):
    """Data validation failures."""


class StorageError(GigaEvoError):
    """Storage operation failures."""


class ProgramError(GigaEvoError):
    """Program execution failures."""


class EvolutionError(GigaEvoError):
    """Evolution process failures."""


class SecurityError(GigaEvoError):
    """Security violations."""


class LLMError(GigaEvoError):
    """Base exception for LLM wrapper errors."""


# LLM subtypes
class LLMValidationError(LLMError):
    """Raised when LLM input validation fails."""


class LLMAPIError(LLMError):
    """Raised when LLM API calls fail after retries."""


# Stage / Program subtypes
class StageExecutionError(GigaEvoError):
    """Stage execution failures."""


class ProgramValidationError(ProgramError):
    """Program validation failures."""


class ProgramExecutionError(ProgramError):
    """Program execution failures."""


class ProgramTimeoutError(ProgramError):
    """Program timeout failures."""


class SecurityViolationError(SecurityError):
    """Security violations in program execution."""


class ResourceError(GigaEvoError):
    """Resource limit violations."""


class MutationError(GigaEvoError):
    """Mutation failures."""


# Memory subsystem
class MemoryError(GigaEvoError):
    """Base exception for memory subsystem errors."""


class MemoryRetrieverError(MemoryError):
    """GAM/retriever build or initialization failures."""


class MemorySearchError(MemoryError):
    """Memory search or retrieval failures."""


class MemoryStorageError(MemoryError):
    """Card persistence, index I/O, or API sync failures."""
