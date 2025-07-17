from abc import ABC, abstractmethod
from typing import List, Optional, Protocol, Dict
import math

from src.programs.program import Program
from src.evolution.strategies.utils import extract_fitness_values, dominates


class ArchiveSelectorProtocol(Protocol):
    """Protocol for archive selector implementations."""
    
    def __call__(self, new: Program, current: Program) -> bool:
        """Determine if new program should replace current elite."""
        ...

class ArchiveSelector(ABC):
    """Base class for archive selection strategies."""
    
    def __init__(self, fitness_keys: List[str], fitness_key_higher_is_better: dict[str, bool] | None = None):
        if not fitness_keys:
            raise ValueError("fitness_keys cannot be empty")
        self.fitness_keys = fitness_keys
        self.fitness_key_higher_is_better = fitness_key_higher_is_better or {key: True for key in fitness_keys}

    @abstractmethod
    def __call__(self, new: Program, current: Program) -> bool:
        """Determine if new program should replace current elite."""
        pass


class SumArchiveSelector(ArchiveSelector):
    def __init__(self, *args, weights: Optional[List[float]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = weights or [1.0] * len(self.fitness_keys)

    def __call__(self, new: Program, current: Program) -> bool:
        new_sum = sum([v * w for v, w in zip(extract_fitness_values(new, self.fitness_keys, self.fitness_key_higher_is_better), self.weights)])
        current_sum = sum([v * w for v, w in zip(extract_fitness_values(current, self.fitness_keys, self.fitness_key_higher_is_better), self.weights)])
        return new_sum > current_sum

    def score(self, program: Program) -> float:
        return sum([v * w for v, w in zip(extract_fitness_values(program, self.fitness_keys, self.fitness_key_higher_is_better), self.weights)])


class ParetoFrontSelector(ArchiveSelector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, new: Program, current: Program) -> bool:
        new_values = extract_fitness_values(new, self.fitness_keys, self.fitness_key_higher_is_better)
        current_values = extract_fitness_values(current, self.fitness_keys, self.fitness_key_higher_is_better)
        return dominates(new_values, current_values)

