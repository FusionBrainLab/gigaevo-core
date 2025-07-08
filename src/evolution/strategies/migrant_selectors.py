import random
from abc import ABC, abstractmethod
from typing import List

from src.programs.program import Program
from src.evolution.strategies.utils import extract_fitness_values, dominates


class MigrantSelector(ABC):
    """Abstract base class for selecting programs to migrate."""

    @abstractmethod
    def __call__(self, programs: List[Program], count: int) -> List[Program]:
        ...


class TopFitnessMigrantSelector(MigrantSelector):
    """Selects top programs by scalar fitness."""
    def __init__(self, fitness_key: str, fitness_key_higher_is_better: bool = True):
        self.fitness_key = fitness_key
        self.fitness_key_higher_is_better = fitness_key_higher_is_better

    def __call__(self, programs: List[Program], count: int) -> List[Program]:
        if not programs:
            return []

        fitness_values = extract_fitness_values(programs, [self.fitness_key], {self.fitness_key: self.fitness_key_higher_is_better})
        scored_programs = [
            (prog, fitness_values[i]) for i, prog in enumerate(programs) if fitness_values[i] is not None
        ]

        if not scored_programs:
            return random.sample(programs, min(count, len(programs)))

        sorted_programs = sorted(scored_programs, key=lambda x: x[1], reverse=True)
        return [p for p, _ in sorted_programs[:count]]


class ParetoFrontMigrantSelector(MigrantSelector):
    """Selects from the Pareto front (non-dominated set)."""
    def __init__(self, fitness_keys: List[str], fitness_key_higher_is_better: dict[str, bool] | None = None):
        self.fitness_keys = fitness_keys
        self.fitness_key_higher_is_better = fitness_key_higher_is_better or {key: True for key in fitness_keys}

    def __call__(self, programs: List[Program], count: int) -> List[Program]:
        if not programs:
            return []

        pareto_front = self._compute_pareto_front(programs, self.fitness_keys, self.fitness_key_higher_is_better)

        if len(pareto_front) >= count:
            return random.sample(pareto_front, count)
        else:
            remaining = list(set(programs) - set(pareto_front))
            filler = random.sample(remaining, min(count - len(pareto_front), len(remaining)))
            return pareto_front + filler

    def _compute_pareto_front(self, programs: List[Program], fitness_keys: List[str], fitness_key_higher_is_better: dict[str, bool]) -> List[Program]:
        pareto_front = []

        for p in programs:
            if all(not dominates(extract_fitness_values(other, fitness_keys, fitness_key_higher_is_better), extract_fitness_values(p, fitness_keys, fitness_key_higher_is_better)) for other in programs if other != p):
                pareto_front.append(p)

        return pareto_front