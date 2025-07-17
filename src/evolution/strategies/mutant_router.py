import random
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from loguru import logger
from src.programs.program import Program
from src.evolution.strategies.island import MapElitesIsland
from src.evolution.strategies.island_selector import IslandCompatibilityMixin

class MutantRouter(ABC):
    """Abstract base class for mutant routing strategies."""
    
    @abstractmethod
    async def route_mutant(self, mutant: Program, islands: List[MapElitesIsland], 
                          context: Optional[Dict[str, Any]] = None) -> Optional[MapElitesIsland]:
        """
        Route a mutant (new program) to an appropriate island.
        
        Args:
            mutant: The new program to route
            islands: List of available islands
            context: Optional context information (e.g., generation, fitness history)
            
        Returns:
            Selected island or None if no suitable island found
        """
        pass

class RandomMutantRouter(MutantRouter, IslandCompatibilityMixin):
    """
    Routes programs to random accepting islands. Always logs selection.
    """
    def __init__(self):
        """Initialize the random mutant router."""
        pass

    async def route_mutant(self, mutant: Program, islands: List[MapElitesIsland], 
                          context: Optional[Dict[str, Any]] = None) -> Optional[MapElitesIsland]:
        """Route mutant to a random accepting island."""
        if not islands:
            return None

        # Get compatible islands
        compatible_islands = await self._get_compatible_islands(mutant, islands)

        if not compatible_islands:
            return None

        # Select random island
        selected = random.choice(compatible_islands)
        logger.debug(f"Selected island {selected.config.island_id} via random selection")
        return selected

    async def _get_compatible_islands(self, mutant: Program, islands: List[MapElitesIsland]) -> List[MapElitesIsland]:
        """Get list of islands that can accept the mutant."""
        compatible_islands = []
        for island in islands:
            if await self._can_accept_program(island, mutant):
                compatible_islands.append(island)
        return compatible_islands 