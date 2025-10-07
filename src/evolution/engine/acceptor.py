from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Set
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.programs.program import Program

from src.programs.program_state import ProgramState

__all__ = [
    "ProgramEvolutionAcceptor",
    "DefaultProgramEvolutionAcceptor",
    "RequiredBehaviorKeysAcceptor",
]


class ProgramEvolutionAcceptor(ABC):
    """Abstract base class for determining if a program should be accepted for evolution."""
    
    @abstractmethod
    def is_accepted(self, program: Program) -> bool:
        """Check if a program should be accepted for evolution.
        
        Args:
            program: The program to validate
            
        Returns:
            True if the program should be accepted, False otherwise
        """
        ...


class DefaultProgramEvolutionAcceptor(ProgramEvolutionAcceptor):
    """Default implementation that checks basic program validity."""
    
    def is_accepted(self, program: Program) -> bool:
        """Check if program meets basic requirements for evolution.
        
        Args:
            program: The program to validate
            
        Returns:
            True if the program should be accepted for evolution
        """
        # Check if program is discarded
        if program.state == ProgramState.DISCARDED:
            logger.debug(
                f"[DefaultAcceptor] Program {program.id} rejected: "
                f"explicitly marked as discarded"
            )
            return False
            
        # Check if DAG processing completed
        if program.state != ProgramState.DAG_PROCESSING_COMPLETED:
            logger.debug(
                f"[DefaultAcceptor] Program {program.id} rejected: "
                f"not completed (state: {program.state}, expected: {ProgramState.DAG_PROCESSING_COMPLETED})"
            )
            return False
            
        # Check if metrics available
        if not program.metrics:
            logger.debug(
                f"[DefaultAcceptor] Program {program.id} rejected: "
                f"no metrics available (likely DAG execution failed)"
            )
            return False
        
        logger.debug(
            f"[DefaultAcceptor] Program {program.id} accepted by EvolutionEngine: "
            f"state={program.state}, metrics={len(program.metrics)} keys"
        )
        return True


class RequiredBehaviorKeysAcceptor(ProgramEvolutionAcceptor):
    """Acceptor that validates programs have required behavior keys."""
    
    def __init__(self, required_behavior_keys: Set[str]) -> None:
        """Initialize with required behavior keys.
        
        Args:
            required_behavior_keys: Set of behavior keys that must be present in program metrics
        """
        self.required_behavior_keys = required_behavior_keys
    
    def is_accepted(self, program: Program) -> bool:
        """Check if program has all required behavior keys.
        
        Args:
            program: The program to validate
            
        Returns:
            True if the program has all required behavior keys
        """
        # First check basic validity
        if not DefaultProgramEvolutionAcceptor().is_accepted(program):
            logger.debug(
                f"[RequiredKeysAcceptor] Program {program.id} rejected by EvolutionEngine: "
                f"failed basic validation"
            )
            return False
        
        # Check required behavior keys
        present_keys = set(program.metrics.keys())
        missing_keys = self.required_behavior_keys - present_keys
        if missing_keys:
            logger.debug(
                f"[RequiredKeysAcceptor] Program {program.id} rejected by EvolutionEngine: "
                f"missing required keys {sorted(missing_keys)} "
                f"(present: {sorted(present_keys)}, required: {sorted(self.required_behavior_keys)})"
            )
            return False
        
        logger.debug(
            f"[RequiredKeysAcceptor] Program {program.id} accepted by EvolutionEngine: "
            f"has all required keys {sorted(self.required_behavior_keys)} "
            f"(total metrics: {len(program.metrics)})"
        )
        return True