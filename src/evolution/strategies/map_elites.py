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

from .models import *
from .selectors import *
from .multi_island import *
from .island import *
from .removers import *
from .mutant_router import *
from .migrant_selectors import *
from .island_selector import *
from .elite_selectors import *
