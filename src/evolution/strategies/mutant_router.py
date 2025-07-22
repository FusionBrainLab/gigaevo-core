from abc import ABC, abstractmethod
import random
from typing import Any, Dict, List, Optional

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
        islands: List[MapElitesIsland],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[MapElitesIsland]:
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
        islands: List[MapElitesIsland],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[MapElitesIsland]:
        """Route mutant to a random accepting island with logging."""
        if not islands:
            return None

        # Get compatible islands with logging
        compatible_islands = await self._get_compatible_islands(mutant, islands)

        if not compatible_islands:
            logger.debug(
                f"🚫 No compatible islands found for mutant {mutant.id}"
            )
            return None

        # Select random island
        selected = random.choice(compatible_islands)

        logger.debug(
            f"🏝️ Routed mutant {mutant.id} to {selected.config.island_id} (random selection)"
        )

        return selected

    async def _get_compatible_islands(
        self, mutant: Program, islands: List[MapElitesIsland]
    ) -> List[MapElitesIsland]:
        """Get list of islands that can accept the mutant with logging."""
        compatible_islands = []
        for island in islands:
            if await self._can_accept_program(island, mutant):
                compatible_islands.append(island)
        return compatible_islands
