from abc import ABC, abstractmethod
import random
from typing import Callable

from gigaevo.evolution.strategies.utils import dominates, extract_fitness_values
from gigaevo.programs.program import Program


class ArchiveRemover(ABC):
    """Base class for archive remover implementations."""

    @abstractmethod
    def __call__(self, programs: list[Program], max_size_to_keep: int) -> list[Program]:
        """Return a list of programs to remove from the archive."""


class ScoreArchiveRemover(ArchiveRemover):
    """Archive remover that removes programs based on a score.
    It is assumed that higher score -> better program.
    """

    def __call__(self, programs: list[Program], max_size_to_keep: int) -> list[Program]:
        """Return a list of programs to remove from the archive."""
        if not programs:
            return []

        if len(programs) <= max_size_to_keep:
            return []

        sorted_programs = sorted(programs, key=self.score)

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

    def __init__(self, fitness_key: str, fitness_key_higher_is_better: bool = True):
        super().__init__()
        self.fitness_key = fitness_key
        self.fitness_key_higher_is_better = fitness_key_higher_is_better

    def score(self, program: Program) -> float:
        if self.fitness_key not in program.metrics:
            raise ValueError(
                f"Fitness key {self.fitness_key} not found in program {program.id} metrics. Available keys: {list(program.metrics.keys())}"
            )

        fitness_value: float = program.metrics[self.fitness_key]
        return fitness_value if self.fitness_key_higher_is_better else -fitness_value


class ParetoFrontArchiveRemover(ArchiveRemover):
    """Archive remover that removes programs based on Pareto front."""

    def __init__(
        self,
        fitness_keys: list[str],
        tie_breaker: Callable[[Program], float],
        fitness_key_higher_is_better: dict[str, bool] | None = None,
    ):
        super().__init__()
        self.fitness_keys = fitness_keys
        self.fitness_key_higher_is_better = fitness_key_higher_is_better or {
            key: True for key in fitness_keys
        }
        self.tie_breaker = tie_breaker

    def __call__(self, programs: list[Program], max_size_to_keep: int) -> list[Program]:
        """Return a list of programs to remove from the archive."""
        if not programs:
            return []

        if len(programs) <= max_size_to_keep:
            return []

        sorted_programs = self.order_candidates(programs)

        num_to_remove = max(0, len(sorted_programs) - max_size_to_keep)
        return sorted_programs[:num_to_remove]

    def order_candidates(self, programs: list[Program]) -> list[Program]:
        """Return programs sorted worst-to-best by dominated count."""
        fitness_vectors = [
            extract_fitness_values(
                p, self.fitness_keys, self.fitness_key_higher_is_better
            )
            for p in programs
        ]

        n = len(programs)

        dominated_counts = [0] * n
        for i in range(n):
            for j in range(n):
                if i != j:
                    if dominates(fitness_vectors[j], fitness_vectors[i]):
                        dominated_counts[i] += 1

        programs_with_dominated_count = []
        for i in range(n):
            tie_break_score = self.tie_breaker(programs[i])

            programs_with_dominated_count.append(
                (programs[i], dominated_counts[i], tie_break_score)
            )

        programs_with_dominated_count.sort(key=lambda x: (-x[1], x[2]))

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
