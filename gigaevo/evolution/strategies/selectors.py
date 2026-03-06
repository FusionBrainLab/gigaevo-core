from abc import ABC, abstractmethod

from gigaevo.evolution.strategies.utils import dominates, extract_fitness_values
from gigaevo.programs.program import Program


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
        from loguru import logger

        new_values = extract_fitness_values(
            new,
            self.fitness_keys,
            self.fitness_key_higher_is_better,
        )
        current_values = extract_fitness_values(
            current,
            self.fitness_keys,
            self.fitness_key_higher_is_better,
        )

        new_sum = sum([v * w for v, w in zip(new_values, self.weights)])
        current_sum = sum([v * w for v, w in zip(current_values, self.weights)])

        result = new_sum > current_sum
        logger.debug(
            "SumArchiveSelector: {} vs {} -> {} (new={:.3f}, current={:.3f}, keys={})",
            new.id,
            current.id,
            "ACCEPT" if result else "REJECT",
            new_sum,
            current_sum,
            self.fitness_keys,
        )

        return result

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
        from loguru import logger

        new_values = extract_fitness_values(
            new, self.fitness_keys, self.fitness_key_higher_is_better
        )
        current_values = extract_fitness_values(
            current, self.fitness_keys, self.fitness_key_higher_is_better
        )
        result = dominates(new_values, current_values)

        logger.debug(
            "ParetoFrontSelector: {} vs {} -> {} (new={}, current={}, keys={})",
            new.id,
            current.id,
            "DOMINATES" if result else "DOES_NOT_DOMINATE",
            [f"{v:.3f}" for v in new_values],
            [f"{v:.3f}" for v in current_values],
            self.fitness_keys,
        )

        return result
