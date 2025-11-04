from abc import ABC, abstractmethod
from itertools import combinations
import random
from typing import Iterator

from loguru import logger

from gigaevo.programs.program import Program


class ParentSelector(ABC):
    """Abstract base class for selecting parents for mutation."""

    @abstractmethod
    def create_parent_iterator(
        self, available_parents: list[Program]
    ) -> Iterator[list[Program]]:
        """Create an iterator that yields parent selections.

        Args:
            available_parents: List of programs available for selection

        Returns:
            Iterator that yields selected parents for mutation
        """


class RandomParentSelector(ParentSelector):
    """Randomly selects parents from the available pool."""

    def __init__(self, num_parents: int = 1):
        if num_parents < 1:
            raise ValueError(f"num_parents must be at least 1, got {num_parents}")
        self.num_parents = num_parents

    def create_parent_iterator(
        self, available_parents: list[Program]
    ) -> Iterator[list[Program]]:
        """Create iterator for random parent selection.

        Yields infinite random selections of parents (consumer controls limit via break).
        If fewer parents are available than requested, returns all available.
        """
        if not available_parents:
            return
        while True:
            yield random.sample(
                available_parents, min(self.num_parents, len(available_parents))
            )


class AllCombinationsParentSelector(ParentSelector):
    """Exhaustively iterates through all combinations of parents."""

    def __init__(self, num_parents: int = 1):
        if num_parents < 1:
            raise ValueError(f"num_parents must be at least 1, got {num_parents}")
        self.num_parents = num_parents

    def create_parent_iterator(
        self, available_parents: list[Program]
    ) -> Iterator[list[Program]]:
        """Create iterator for all combinations of parents.

        Yields all possible combinations of the requested number of parents.
        Combinations are shuffled for randomness.
        If fewer parents are available than requested, yields all available parents once.
        """
        if not available_parents:
            return

        parents_copy = available_parents.copy()
        random.shuffle(parents_copy)

        if len(parents_copy) < self.num_parents:
            logger.info(
                f"[AllCombinationsParentSelector] Only {len(parents_copy)} parents available, yielding all"
            )
            yield parents_copy
            return

        for combo in combinations(parents_copy, self.num_parents):
            yield list(combo)
