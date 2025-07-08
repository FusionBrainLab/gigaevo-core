"""Modular Evolution Engine package.

This sub-package exposes a clean public surface while delegating heavy
implementation details to sub-modules.  Importing from
`src.evolution.engine` continues to behave as before:

    from src.evolution.engine import EvolutionEngine, EngineConfig
"""
from __future__ import annotations

from .config import EngineConfig
from .metrics import EngineMetrics
from .validation import ValidationFailureReason, ProgramValidationResult
from .core import EvolutionEngine  # noqa: F401  (public symbol)

__all__ = [
    "EngineConfig",
    "EngineMetrics",
    "ValidationFailureReason",
    "ProgramValidationResult",
    "EvolutionEngine",
] 