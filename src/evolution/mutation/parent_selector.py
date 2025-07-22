from abc import ABC, abstractmethod
from itertools import combinations
import random
from typing import List, Optional

from src.programs.program import Program


class ParentSelector(ABC):
    """Abstract base class for selecting parents for mutation."""

    @abstractmethod
    def select_parents(
        self, available_parents: List[Program]
    ) -> Optional[List[Program]]:
        """Select parents from the available pool.

        Args:
            available_parents: List of programs available for selection

        Returns:
            Selected parents for mutation, or None if no valid selection can be made
        """

    @abstractmethod
    def has_more_selections(self) -> bool:
        """Check if more parent selections are available."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the selector to start from the beginning."""


class RandomParentSelector(ParentSelector):
    """Randomly selects parents from the available pool."""

    def __init__(self, num_parents: int = 1):
        self.num_parents = num_parents

    def select_parents(
        self, available_parents: List[Program]
    ) -> Optional[List[Program]]:
        if len(available_parents) < self.num_parents:
            return None
        return random.sample(available_parents, self.num_parents)

    def has_more_selections(self) -> bool:
        return True  # Random selection always has more possibilities

    def reset(self) -> None:
        pass  # Nothing to reset for random selection


class AllCombinationsParentSelector(ParentSelector):
    """Exhaustively iterates through all combinations of parents."""

    def __init__(self, num_parents: int = 1):
        self.num_parents = num_parents
        self.current_index = 0
        self.all_combinations = []

    def select_parents(
        self, available_parents: List[Program]
    ) -> Optional[List[Program]]:
        if len(available_parents) < self.num_parents:
            return None

        # Generate combinations if not already done or parents changed
        expected_combinations = list(
            combinations(available_parents, self.num_parents)
        )
        if self.all_combinations != expected_combinations:
            self.all_combinations = expected_combinations
            self.current_index = 0

        if self.current_index >= len(self.all_combinations):
            return None

        selected = list(self.all_combinations[self.current_index])
        self.current_index += 1
        return selected

    def has_more_selections(self) -> bool:
        return self.current_index < len(self.all_combinations)

    def reset(self) -> None:
        self.current_index = 0


class WeightedRandomParentSelector(ParentSelector):
    """Selects parents using fitness-weighted random selection."""

    def __init__(
        self,
        num_parents: int = 1,
        fitness_key: str = "fitness",
        higher_is_better: bool = True,
    ):
        self.num_parents = num_parents
        self.fitness_key = fitness_key
        self.higher_is_better = higher_is_better

    def select_parents(
        self, available_parents: List[Program]
    ) -> Optional[List[Program]]:
        if len(available_parents) < self.num_parents:
            return None

        # Extract fitness values
        fitnesses = []
        valid_parents = []
        for parent in available_parents:
            if self.fitness_key in parent.metrics:
                fitness = parent.metrics[self.fitness_key]
                if self.higher_is_better:
                    fitnesses.append(max(0, fitness))  # Ensure non-negative
                else:
                    fitnesses.append(
                        max(0, -fitness)
                    )  # Invert for minimization
                valid_parents.append(parent)

        if len(valid_parents) < self.num_parents:
            return None

        # Handle case where all fitnesses are zero
        if all(f == 0 for f in fitnesses):
            return random.sample(valid_parents, self.num_parents)

        # Weighted selection without replacement
        selected = []
        remaining_parents = valid_parents.copy()
        remaining_fitnesses = fitnesses.copy()

        for _ in range(self.num_parents):
            if not remaining_parents:
                break

            chosen = random.choices(
                remaining_parents, weights=remaining_fitnesses, k=1
            )[0]
            selected.append(chosen)

            # Remove from remaining pools
            idx = remaining_parents.index(chosen)
            remaining_parents.pop(idx)
            remaining_fitnesses.pop(idx)

        return selected if len(selected) == self.num_parents else None

    def has_more_selections(self) -> bool:
        return True  # Weighted random selection always has more possibilities

    def reset(self) -> None:
        pass  # Nothing to reset for weighted random selection
