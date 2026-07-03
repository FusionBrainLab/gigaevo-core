"""Genome-agnostic diff contract for StructuredDiffMutationOperator.

A subclass owns one genome family (e.g. CARL chain DAGs in gigaevo/chains/)
and defines which changes are representable; the operator never inspects
genome internals. Parents are keyed by prompt namespace ("A", "B", ...) to
genome code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DiffSchema:
    json_schema: dict[str, Any]
    validate: Callable[[Any], Any]


class AllowedChanges(ABC):
    @abstractmethod
    def build_schema(self, parents: dict[str, str]) -> DiffSchema:
        """Per-call wire schema (guided decoding) + validator for these parents."""

    @abstractmethod
    def render_parents(self, parents: dict[str, str]) -> str:
        """Render parent genomes for the mutation prompt, with stable step ids."""

    @abstractmethod
    def apply(self, diff: Any, parents: dict[str, str]) -> str:
        """Transcribe a validated diff into child genome code; MutationError on failure."""

    @abstractmethod
    def describe(self) -> str:
        """Prose description of the diff language for the system prompt."""
