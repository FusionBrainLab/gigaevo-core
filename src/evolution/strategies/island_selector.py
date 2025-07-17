"""
Island selector for handling intelligent island selection for programs.
"""

import random
from typing import List, Optional
from src.evolution.strategies.island import MapElitesIsland
from src.programs.program import Program
from loguru import logger
from abc import ABC, abstractmethod

class IslandCompatibilityMixin:
    """Mixin providing common island compatibility checking functionality."""
    
    @staticmethod
    async def _can_accept_program(island: MapElitesIsland, program: Program) -> bool:
        """Check if island can accept program."""
        try:
            required_keys = set(island.config.behavior_space.behavior_keys)
            if not required_keys.issubset(program.metrics.keys()):
                logger.debug(f"Island {island.config.island_id} rejected program {program.id}: missing keys {required_keys - program.metrics.keys()}")
                return False
                
            cell = island.config.behavior_space.get_cell(program.metrics)
            existing = await island.archive_storage.get_elite(cell)
            
            can_accept = existing is None or island.config.archive_selector(program, existing)
            logger.debug(f"Island {island.config.island_id} can accept program {program.id}: {can_accept} (existing: {existing is not None})")
            return can_accept
            
        except Exception as e:
            logger.warning(f"Error evaluating island {island.config.island_id} for program {program.id}: {e}")
            return False

class IslandSelector(ABC):
    """Abstract base class for island selection strategies."""
    
    @abstractmethod
    async def select_island(self, program: Program, islands: List[MapElitesIsland]) -> Optional[MapElitesIsland]:
        """
        Select the best island for a program.
        
        Args:
            program: The program to place
            islands: List of available islands
            
        Returns:
            Selected island or None if no suitable island found
        """
        pass

class WeightedIslandSelector(IslandSelector, IslandCompatibilityMixin):
    """Weighted random selection based on island size and compatibility."""
    
    async def select_island(self, program: Program, islands: List[MapElitesIsland]) -> Optional[MapElitesIsland]:
        """Select island using weighted random selection based on size."""
        if not islands:
            return None
            
        # Find islands that can accept the program
        accepting_islands = []
        for island in islands:
            try:
                if await self._can_accept_program(island, program):
                    accepting_islands.append(island)
            except Exception as e:
                logger.warning(f"Error evaluating island {island.config.island_id} for program {program.id}: {e}")
        
        if not accepting_islands:
            return None
            
        # Weighted selection based on size
        return await self._weighted_select(accepting_islands)
    
    @staticmethod
    async def _weighted_select(islands: List[MapElitesIsland]) -> MapElitesIsland:
        """Select island using weighted random selection based on size."""
        if not islands:
            return None
            
        # Get sizes for all islands
        island_info = []
        for island in islands:
            try:
                size = await island.get_elite_count()
                island_info.append((island, size))
            except Exception as e:
                logger.warning(f"Failed to get size for island {island.config.island_id}: {e}")
                island_info.append((island, 0))
        
        if not island_info:
            return None
        
        # Calculate weights (inverse of size to favor less populated islands)
        total_size = sum(size for _, size in island_info)
        if total_size == 0:
            # If all islands are empty, select randomly
            return random.choice(islands)
        
        weights = []
        for island, size in island_info:
            # Weight inversely proportional to size (smaller islands get higher weight)
            weight = 1.0 / (size + 1)  # Add 1 to avoid division by zero
            weights.append(weight)
            logger.debug(f"Island {island.config.island_id}: size={size}, weight={weight:.3f}")
        
        # Weighted random selection
        selected_island = random.choices(islands, weights=weights, k=1)[0]
        logger.debug(f"Selected island {selected_island.config.island_id} via weighted random selection")
        return selected_island

class RoundRobinIslandSelector(IslandSelector, IslandCompatibilityMixin):
    """Round-robin selection for even distribution."""
    
    def __init__(self):
        self._last_index = -1
    
    async def select_island(self, program: Program, islands: List[MapElitesIsland]) -> Optional[MapElitesIsland]:
        """Select island using round-robin selection."""
        if not islands:
            return None
            
        # Find islands that can accept the program
        accepting_islands = []
        for island in islands:
            try:
                if await self._can_accept_program(island, program):
                    accepting_islands.append(island)
            except Exception as e:
                logger.warning(f"Error evaluating island {island.config.island_id} for program {program.id}: {e}")
        
        if not accepting_islands:
            return None
        
        # Round-robin selection
        self._last_index = (self._last_index + 1) % len(accepting_islands)
        selected = accepting_islands[self._last_index]
        logger.debug(f"Selected island {selected.config.island_id} via round-robin selection")
        return selected

class RandomIslandSelector(IslandSelector, IslandCompatibilityMixin):
    """Random selection among compatible islands."""
    
    async def select_island(self, program: Program, islands: List[MapElitesIsland]) -> Optional[MapElitesIsland]:
        """Select island using random selection."""
        if not islands:
            return None
            
        # Find islands that can accept the program
        accepting_islands = []
        for island in islands:
            try:
                if await self._can_accept_program(island, program):
                    accepting_islands.append(island)
            except Exception as e:
                logger.warning(f"Error evaluating island {island.config.island_id} for program {program.id}: {e}")
        
        if not accepting_islands:
            return None
        
        # Random selection
        selected = random.choice(accepting_islands)
        logger.debug(f"Selected island {selected.config.island_id} via random selection")
        return selected 