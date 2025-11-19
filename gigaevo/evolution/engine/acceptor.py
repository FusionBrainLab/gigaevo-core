from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Set

from loguru import logger

from gigaevo.evolution.mutation.context import MUTATION_CONTEXT_METADATA_KEY
from gigaevo.programs.metrics.context import VALIDITY_KEY
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState


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


class CompositeAcceptor(ProgramEvolutionAcceptor):
    """Acceptor that runs a sequence of other acceptors."""

    def __init__(self, acceptors: list[ProgramEvolutionAcceptor]) -> None:
        self.acceptors = acceptors

    def is_accepted(self, program: Program) -> bool:
        for acceptor in self.acceptors:
            if not acceptor.is_accepted(program):
                return False
        return True


class StateAcceptor(ProgramEvolutionAcceptor):
    """Checks if program is in the correct state."""

    def is_accepted(self, program: Program) -> bool:
        if program.state == ProgramState.DISCARDED:
            logger.debug(
                f"[StateAcceptor] Program {program.id} rejected: explicitly marked as discarded"
            )
            return False

        if program.state != ProgramState.DAG_PROCESSING_COMPLETED:
            logger.debug(
                f"[StateAcceptor] Program {program.id} rejected: "
                f"not completed (state: {program.state}, expected: {ProgramState.DAG_PROCESSING_COMPLETED})"
            )
            return False
        return True


class MetricsExistenceAcceptor(ProgramEvolutionAcceptor):
    """Checks if metrics are present."""

    def is_accepted(self, program: Program) -> bool:
        if not program.metrics:
            logger.debug(
                f"[MetricsExistenceAcceptor] Program {program.id} rejected: "
                f"no metrics available (likely DAG execution failed)"
            )
            return False
        return True


class ValidityMetricAcceptor(ProgramEvolutionAcceptor):
    """Checks if the program is marked as valid in metrics."""

    def __init__(self, validity_key: str = VALIDITY_KEY) -> None:
        self.validity_key = validity_key

    def is_accepted(self, program: Program) -> bool:
        is_valid = program.metrics.get(self.validity_key)

        # We treat missing key as invalid, or explicit False/0 as invalid
        if not is_valid:
            logger.debug(
                f"[ValidityMetricAcceptor] Program {program.id} rejected: "
                f"{self.validity_key}={is_valid}"
            )
            return False
        return True


class RequiredBehaviorKeysAcceptor(ProgramEvolutionAcceptor):
    """Acceptor that validates programs have required behavior keys."""

    def __init__(self, required_behavior_keys: Set[str]) -> None:
        self.required_behavior_keys = required_behavior_keys

    def is_accepted(self, program: Program) -> bool:
        present_keys = set(program.metrics.keys())
        missing_keys = self.required_behavior_keys - present_keys
        if missing_keys:
            logger.debug(
                f"[RequiredKeysAcceptor] Program {program.id} rejected: "
                f"missing required keys {sorted(missing_keys)} "
                f"(present: {sorted(present_keys)}, required: {sorted(self.required_behavior_keys)})"
            )
            return False
        return True


class MutationContextAcceptor(ProgramEvolutionAcceptor):
    """Acceptor that validates programs have a mutation context."""

    def is_accepted(self, program: Program) -> bool:
        if program.get_metadata(MUTATION_CONTEXT_METADATA_KEY) is None:
            logger.debug(
                f"[MutationContextAcceptor] Program {program.id} rejected: no mutation context"
            )
            return False
        return True


class DefaultProgramEvolutionAcceptor(CompositeAcceptor):
    """Legacy default acceptor checking state and metrics existence."""

    def __init__(self) -> None:
        super().__init__([StateAcceptor(), MetricsExistenceAcceptor()])


class StandardEvolutionAcceptor(CompositeAcceptor):
    """Standard composition for most experiments."""

    def __init__(
        self, required_behavior_keys: Set[str], validity_key: str = VALIDITY_KEY
    ) -> None:
        super().__init__(
            [
                StateAcceptor(),
                MetricsExistenceAcceptor(),
                ValidityMetricAcceptor(validity_key=validity_key),
                RequiredBehaviorKeysAcceptor(required_behavior_keys),
                MutationContextAcceptor(),
            ]
        )


class MutationContextAndBehaviorKeysAcceptor(CompositeAcceptor):
    """Legacy compatibility class."""

    def __init__(self, required_behavior_keys: Set[str]) -> None:
        super().__init__(
            [
                StateAcceptor(),
                MetricsExistenceAcceptor(),
                RequiredBehaviorKeysAcceptor(required_behavior_keys),
                MutationContextAcceptor(),
            ]
        )
