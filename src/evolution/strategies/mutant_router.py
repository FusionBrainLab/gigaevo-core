import random
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger

from src.evolution.strategies.island import MapElitesIsland
from src.evolution.strategies.island_selector import IslandCompatibilityMixin
from src.programs.program import Program


class MutantRouter(ABC):
    """Abstract base class for mutant routing strategies."""

    @abstractmethod
    async def route_mutant(
        self,
        mutant: Program,
        islands: list[MapElitesIsland],
        context: dict[str, Any] | None = None,
    ) -> MapElitesIsland | None:
        """
        Route a mutant (new program) to an appropriate island.

        Args:
            mutant: The new program to route
            islands: List of available islands
            context: Optional context information (e.g., generation, fitness history)

        Returns:
            Selected island or None if no suitable island found
        """


class RandomMutantRouter(MutantRouter, IslandCompatibilityMixin):
    """
    Routes programs to random accepting islands.
    """

    async def route_mutant(
        self,
        mutant: Program,
        islands: list[MapElitesIsland],
        context: dict[str, Any] | None = None,
    ) -> MapElitesIsland | None:
        if not islands:
            return None

        compatible_islands = await self._get_compatible_islands(mutant, islands)

        if not compatible_islands:
            logger.debug(f"🚫 No compatible islands found for mutant {mutant.id}")
            return None

        selected = random.choice(compatible_islands)

        logger.debug(
            f"🏝️ Routed mutant {mutant.id} to {selected.config.island_id} (random selection)"
        )

        return selected

    async def _get_compatible_islands(
        self, mutant: Program, islands: list[MapElitesIsland]
    ) -> list[MapElitesIsland]:
        compatible_islands = []
        for island in islands:
            if await self._can_accept_program(island, mutant):
                compatible_islands.append(island)
        return compatible_islands
