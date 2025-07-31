from abc import ABC, abstractmethod
from itertools import combinations
import random
from typing import Iterator, List, Optional

from src.programs.program import Program


class ParentSelector(ABC):
    """Abstract base class for selecting parents for mutation."""

    @abstractmethod
    def create_parent_iterator(
        self, available_parents: List[Program]
    ) -> Iterator[List[Program]]:
        """Create an iterator that yields parent selections.

        Args:
            available_parents: List of programs available for selection

        Returns:
            Iterator that yields selected parents for mutation
        """


class RandomParentSelector(ParentSelector):
    """Randomly selects parents from the available pool."""

    def __init__(self, num_parents: int = 1, max_selections: Optional[int] = None):
        self.num_parents = num_parents
        self.max_selections = max_selections

    def create_parent_iterator(
        self, available_parents: List[Program]
    ) -> Iterator[List[Program]]:
        """Create iterator for random parent selection."""
        if len(available_parents) < self.num_parents:
            return
        
        count = 0
        while self.max_selections is None or count < self.max_selections:
            yield random.sample(available_parents, self.num_parents)
            count += 1


class AllCombinationsParentSelector(ParentSelector):
    """Exhaustively iterates through all combinations of parents."""

    def __init__(self, num_parents: int = 1):
        self.num_parents = num_parents

    def create_parent_iterator(
        self, available_parents: List[Program]
    ) -> Iterator[List[Program]]:
        """Create iterator for all combinations of parents."""
        if len(available_parents) < self.num_parents:
            return
        
        if self.num_parents == 1:
            # Yield each parent individually
            for parent in available_parents:
                yield [parent]
        else:
            # Yield all combinations
            for combo in combinations(available_parents, self.num_parents):
                yield list(combo)


class WeightedRandomParentSelector(ParentSelector):
    """Selects parents using fitness-weighted random selection."""

    def __init__(
        self,
        num_parents: int = 1,
        fitness_key: str = "fitness",
        higher_is_better: bool = True,
        max_selections: Optional[int] = None,
    ):
        self.num_parents = num_parents
        self.fitness_key = fitness_key
        self.higher_is_better = higher_is_better
        self.max_selections = max_selections

    def create_parent_iterator(
        self, available_parents: List[Program]
    ) -> Iterator[List[Program]]:
        """Create iterator for weighted random parent selection."""
        if len(available_parents) < self.num_parents:
            return

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
            return

        # Handle case where all fitnesses are zero
        if all(f == 0 for f in fitnesses):
            count = 0
            while self.max_selections is None or count < self.max_selections:
                yield random.sample(valid_parents, self.num_parents)
                count += 1
            return

        # Weighted selection with replacement (since we want multiple selections)
        count = 0
        while self.max_selections is None or count < self.max_selections:
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

                # Remove from remaining pools for selection without replacement within this group
                idx = remaining_parents.index(chosen)
                remaining_parents.pop(idx)
                remaining_fitnesses.pop(idx)

            if len(selected) == self.num_parents:
                yield selected
            count += 1
