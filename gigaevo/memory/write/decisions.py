"""Typed decisions and outcome labels for the memory write path."""

from enum import StrEnum


class WriteDecision(StrEnum):
    """The complete logical result space of a librarian proposal."""

    DROP = "DROP"
    NEW = "NEW"
    EQUIVALENT = "EQUIVALENT"


class ValidityStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"


class ArchiveStatus(StrEnum):
    ARCHIVED = "archived"
    REJECTED = "rejected"
