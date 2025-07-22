from abc import ABC, abstractmethod
import random
from typing import Callable, List, Protocol

from loguru import logger

from src.evolution.strategies.utils import dominates, extract_fitness_values
from src.programs.program import Program


class ArchiveRemoverProtocol(Protocol):
    """Protocol for archive remover implementations."""

    def __call__(
        self, programs: List[Program], max_size_to_keep: int
    ) -> List[Program]:
        """Return a list of programs to remove from the archive, so at most `max_size_to_keep` programs are kept.
        The returned list should be a subset of the input list.
        """
        ...


class ArchiveRemover(ABC):
    """Base class for archive remover implementations."""

    @abstractmethod
    def __call__(
        self, programs: List[Program], max_size_to_keep: int
    ) -> List[Program]:
        """Return a list of programs to remove from the archive."""


class ScoreArchiveRemover(ArchiveRemover):
    """Archive remover that removes programs based on a score.
    It is assumed that higher score -> better program.
    """

    def __call__(
        self, programs: List[Program], max_size_to_keep: int
    ) -> List[Program]:
        """Return a list of programs to remove from the archive."""
        if not programs:
            return []

        if len(programs) <= max_size_to_keep:
            return []

        try:
            sorted_programs = sorted(programs, key=self.score)
        except Exception as e:
            # Fallback to random removal if scoring fails
            logger.warning(
                f"Score-based removal failed: {e}. Falling back to random removal."
            )
            num_to_remove = len(programs) - max_size_to_keep
            return random.sample(programs, num_to_remove)

        # remove the worst programs
        num_to_remove = max(0, len(sorted_programs) - max_size_to_keep)
        return sorted_programs[:num_to_remove]

    @abstractmethod
    def score(self, program: Program) -> float:
        """Calculate score for program."""


class OldestArchiveRemover(ScoreArchiveRemover):
    """Archive remover that removes the oldest programs."""

    def score(self, program: Program) -> float:
        """Calculate score for program."""
        return program.created_at.timestamp()


class RandomArchiveRemover(ScoreArchiveRemover):
    """Archive remover that removes programs randomly."""

    def score(self, program: Program) -> float:
        """Calculate score for program."""
        return random.random()


class FitnessArchiveRemover(ScoreArchiveRemover):
    """Archive remover that removes programs based on fitness."""

    def __init__(
        self, fitness_key: str, fitness_key_higher_is_better: bool = True
    ):
        super().__init__()
        self.fitness_key = fitness_key
        self.fitness_key_higher_is_better = fitness_key_higher_is_better

    def score(self, program: Program) -> float:
        """Calculate score for program."""
        if not hasattr(program, "metrics") or program.metrics is None:
            return (
                float("-inf")
                if self.fitness_key_higher_is_better
                else float("inf")
            )

        if self.fitness_key not in program.metrics:
            logger.warning(
                f"Fitness key {self.fitness_key} not found in program {program.id} metrics. Available keys: {list(program.metrics.keys())}"
            )
            return (
                float("-inf")
                if self.fitness_key_higher_is_better
                else float("inf")
            )

        fitness_value = program.metrics.get(self.fitness_key, 0.0)
        try:
            return (
                float(fitness_value)
                if self.fitness_key_higher_is_better
                else -float(fitness_value)
            )
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid fitness value for program {program.id}: {fitness_value}"
            )
            return (
                float("-inf")
                if self.fitness_key_higher_is_better
                else float("inf")
            )


class ParetoFrontArchiveRemover(ArchiveRemover):
    """Archive remover that removes programs based on Pareto front."""

    def __init__(
        self,
        fitness_keys: list[str],
        tie_breaker: Callable[[Program], float],
        fitness_key_higher_is_better: dict[str, bool] | None = None,
    ):
        # by default, all fitness keys are assumed to be higher is better
        # tie_breaker is used to rank programs that are both on the same Pareto front level (have the same dominated count); higher score -> better program
        super().__init__()
        self.fitness_keys = fitness_keys
        self.fitness_key_higher_is_better = fitness_key_higher_is_better or {
            key: True for key in fitness_keys
        }
        self.tie_breaker = tie_breaker

    def __call__(
        self, programs: List[Program], max_size_to_keep: int
    ) -> List[Program]:
        """Return a list of programs to remove from the archive."""
        if not programs:
            return []

        if len(programs) <= max_size_to_keep:
            return []

        try:
            sorted_programs = self.order_candidates(programs)
        except Exception as e:
            # Fallback to random removal if Pareto sorting fails
            logger.warning(
                f"Pareto front removal failed: {e}. Falling back to random removal."
            )
            num_to_remove = len(programs) - max_size_to_keep
            return random.sample(programs, num_to_remove)

        num_to_remove = max(0, len(sorted_programs) - max_size_to_keep)
        return sorted_programs[:num_to_remove]

    def order_candidates(self, programs: List[Program]) -> List[Program]:
        """Return programs sorted worst-to-best by dominated count."""
        try:
            fitness_vectors = [
                extract_fitness_values(
                    p, self.fitness_keys, self.fitness_key_higher_is_better
                )
                for p in programs
            ]
        except Exception as e:
            logger.warning(
                f"Failed to extract fitness values for Pareto sorting: {e}"
            )
            # Return programs in random order as fallback
            random.shuffle(programs)
            return programs

        n = len(programs)

        # Compute dominated count for each program
        dominated_counts = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j:
                    try:
                        if dominates(fitness_vectors[j], fitness_vectors[i]):
                            dominated_counts[i] += 1
                    except Exception:
                        # Skip if domination check fails
                        continue

        programs_with_dominated_count = []
        for i in range(n):
            try:
                tie_break_score = self.tie_breaker(programs[i])
            except Exception:
                tie_break_score = random.random()  # Fallback to random

            programs_with_dominated_count.append(
                (programs[i], dominated_counts[i], tie_break_score)
            )

        # Sort by dominated count desc (worst first), then tie breaker asc (worst first)
        programs_with_dominated_count.sort(key=lambda x: (-x[1], x[2]))

        # Return just the programs: dominated programs first, then the rest
        return [p for p, _, _ in programs_with_dominated_count]


class ParetoFrontArchiveRemoverDropOldest(ParetoFrontArchiveRemover):
    """Archive remover that removes the oldest programs from the Pareto front."""

    def __init__(
        self,
        fitness_keys: list[str],
        fitness_key_higher_is_better: dict[str, bool] | None = None,
    ):
        super().__init__(
            fitness_keys,
            lambda x: x.created_at.timestamp(),
            fitness_key_higher_is_better,
        )
