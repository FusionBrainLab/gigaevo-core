from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import random

from gigaevo.evolution.strategies.island import MapElitesIsland
from gigaevo.programs.program import Program


class IslandCompatibilityMixin:
    """Common compatibility check: behavior keys present and either empty cell or improves elite."""

    @staticmethod
    async def _can_accept_program(island: MapElitesIsland, program: Program) -> bool:
        required = set(island.config.behavior_space.behavior_keys)
        if not required.issubset(program.metrics.keys()):
            return False

        cell = island.config.behavior_space.get_cell(program.metrics)
        current = await island.archive_storage.get_elite(cell)
        return (current is None) or island.config.archive_selector(program, current)

    @staticmethod
    async def _filter_accepting_islands(
        islands: list[MapElitesIsland], program: Program
    ) -> list[MapElitesIsland]:
        """Filter islands that can accept the program (parallel checks)."""
        if not islands:
            return []
        results = await asyncio.gather(
            *[IslandCompatibilityMixin._can_accept_program(i, program) for i in islands]
        )
        return [i for i, ok in zip(islands, results) if ok]


class IslandSelector(ABC):
    """Abstract base class for island selection strategies."""

    @abstractmethod
    async def select_island(
        self, program: Program, islands: list[MapElitesIsland]
    ) -> MapElitesIsland | None:
        """Return a destination island or None."""


class WeightedIslandSelector(IslandSelector, IslandCompatibilityMixin):
    """Weighted random selection favoring smaller islands (uses len(island))."""

    async def select_island(
        self, program: Program, islands: list[MapElitesIsland]
    ) -> MapElitesIsland | None:
        if not islands:
            return None

        accepting = await self._filter_accepting_islands(islands, program)
        if not accepting:
            return None

        # Inverse-size weighting: weight = 1 / (len(island) + 1)
        # Fetch sizes in parallel
        sizes = await asyncio.gather(*[i.__len__() for i in accepting])
        weights = [1.0 / (size + 1) for size in sizes]
        return random.choices(accepting, weights=weights, k=1)[0]


class RoundRobinIslandSelector(IslandSelector, IslandCompatibilityMixin):
    """Round-robin among compatible islands."""

    def __init__(self) -> None:
        self._idx = -1

    async def select_island(
        self, program: Program, islands: list[MapElitesIsland]
    ) -> MapElitesIsland | None:
        if not islands:
            return None

        accepting = await self._filter_accepting_islands(islands, program)
        if not accepting:
            return None

        self._idx = (self._idx + 1) % len(accepting)
        return accepting[self._idx]


class RandomIslandSelector(IslandSelector, IslandCompatibilityMixin):
    """Uniform random among compatible islands."""

    async def select_island(
        self, program: Program, islands: list[MapElitesIsland]
    ) -> MapElitesIsland | None:
        if not islands:
            return None

        accepting = await self._filter_accepting_islands(islands, program)
        if not accepting:
            return None

        return random.choice(accepting)
