"""
MAP-Elites Strategy Implementation

This module provides a comprehensive MAP-Elites implementation with:
- Multi-island architecture for diverse evolution
- Quality-Diversity optimization
- Inter-island migration
- Size-bounded archives
- Multiple selection strategies

The implementation is split across multiple modules for maintainability:
- models.py: Core data models and behavior spaces
- selectors.py: Archive selection strategies
- island.py: Single island implementation
- multi_island.py: Multi-island orchestration
"""

from .island import (
    IslandConfig,
    MapElitesIsland,
)

# Core imports
from .models import (
    DEFAULT_MIGRATION_RATE,
    DEFAULT_REDIS_PREFIX,
    MIN_COORDINATE,
    BehaviorSpace,
    BinningType,
    QualityDiversityMetrics,
    SelectionMode,
)
from .multi_island import (
    MapElitesMultiIsland,
)
from .selectors import (
    ArchiveSelector,
    ArchiveSelectorProtocol,
    ParetoFrontSelector,
    SumArchiveSelector,
)

# Expose all classes at module level
__all__ = [
    # Enums and Constants
    "SelectionMode",
    "BinningType",
    "DEFAULT_REDIS_PREFIX",
    "MIN_COORDINATE",
    "DEFAULT_MIGRATION_RATE",
    # Core Models
    "QualityDiversityMetrics",
    "BehaviorSpace",
    # Selectors
    "ArchiveSelector",
    "ArchiveSelectorProtocol",
    "SumArchiveSelector",
    "ParetoFrontSelector",
    # Island Components
    "IslandConfig",
    "MapElitesIsland",
    # Multi-Island Implementation
    "MapElitesMultiIsland",
]
