from abc import ABC, abstractmethod

from src.evolution.strategies.utils import dominates, extract_fitness_values
from src.programs.program import Program


class ArchiveSelector(ABC):
    """Base class for archive selection strategies."""

    def __init__(
        self,
        fitness_keys: list[str],
        fitness_key_higher_is_better: list[bool] | None = None,
    ):
        if not fitness_keys:
            raise ValueError("fitness_keys cannot be empty")
        self.fitness_keys = fitness_keys
        if fitness_key_higher_is_better is None:
            fitness_key_higher_is_better = [True] * len(fitness_keys)
        self.fitness_key_higher_is_better = dict(
            zip(fitness_keys, fitness_key_higher_is_better)
        )

    @abstractmethod
    def __call__(self, new: Program, current: Program) -> bool:
        """Determine if new program should replace current elite."""


class SumArchiveSelector(ArchiveSelector):
    def __init__(self, *args, weights: list[float] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = weights or [1.0] * len(self.fitness_keys)

    def __call__(self, new: Program, current: Program) -> bool:
        new_sum = sum(
            [
                v * w
                for v, w in zip(
                    extract_fitness_values(
                        new,
                        self.fitness_keys,
                        self.fitness_key_higher_is_better,
                    ),
                    self.weights,
                )
            ]
        )
        current_sum = sum(
            [
                v * w
                for v, w in zip(
                    extract_fitness_values(
                        current,
                        self.fitness_keys,
                        self.fitness_key_higher_is_better,
                    ),
                    self.weights,
                )
            ]
        )
        return new_sum > current_sum

    def score(self, program: Program) -> float:
        return sum(
            [
                v * w
                for v, w in zip(
                    extract_fitness_values(
                        program,
                        self.fitness_keys,
                        self.fitness_key_higher_is_better,
                    ),
                    self.weights,
                )
            ]
        )


class ParetoFrontSelector(ArchiveSelector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, new: Program, current: Program) -> bool:
        new_values = extract_fitness_values(
            new, self.fitness_keys, self.fitness_key_higher_is_better
        )
        current_values = extract_fitness_values(
            current, self.fitness_keys, self.fitness_key_higher_is_better
        )
        return dominates(new_values, current_values)
