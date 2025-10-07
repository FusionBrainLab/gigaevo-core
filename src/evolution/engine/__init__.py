"""Modular Evolution Engine package.

This sub-package exposes a clean public surface while delegating heavy
implementation details to sub-modules.  Importing from
`src.evolution.engine` continues to behave as before:

    from src.evolution.engine import EvolutionEngine, EngineConfig
"""

from __future__ import annotations

from .acceptor import ProgramEvolutionAcceptor, DefaultProgramEvolutionAcceptor, RequiredBehaviorKeysAcceptor
from .config import EngineConfig
from .core import EvolutionEngine  # noqa: F401  (public symbol)
from .metrics import EngineMetrics

__all__ = [
    "EngineConfig",
    "EngineMetrics",
    "ProgramEvolutionAcceptor",
    "DefaultProgramEvolutionAcceptor",
    "RequiredBehaviorKeysAcceptor",
    "EvolutionEngine",
]
