from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, computed_field

from gigaevo.programs.program import Program


class StrategyMetrics(BaseModel):
    """Generic metrics that any evolution strategy can provide."""

    total_programs: int = Field(
        default=0, ge=0, description="Total number of programs in the strategy"
    )

    active_populations: int = Field(
        default=0, ge=0, description="Number of active populations/islands"
    )

    strategy_specific_metrics: Optional[Dict[str, Any]] = Field(
        default=None, description="Strategy-specific metrics and statistics"
    )

    @computed_field
    @property
    def programs_per_population(self) -> float:
        """Calculate average programs per population."""
        if self.active_populations == 0:
            return 0.0
        return self.total_programs / self.active_populations

    @computed_field
    @property
    def has_programs(self) -> bool:
        """Check if strategy contains any programs."""
        return self.total_programs > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with computed fields."""
        result = {
            "total_programs": self.total_programs,
            "active_populations": self.active_populations,
            "programs_per_population": round(self.programs_per_population, 2),
            "has_programs": self.has_programs,
        }
        if self.strategy_specific_metrics:
            result.update(self.strategy_specific_metrics)
        return result


class EvolutionStrategy(ABC):
    """
    Abstract base class for evolution strategies.

    Defines the core interface that all evolution strategies must implement,
    along with optional capabilities for enhanced monitoring and control.
    """

    @abstractmethod
    async def add(self, program: Program) -> bool:
        """
        Add a program to the evolution strategy.

        Args:
            program: The program to add

        Returns:
            True if program was added/updated, False otherwise
        """
        ...

    @abstractmethod
    async def select_elites(self, total: int) -> list[Program]:
        """
        Select elite programs from the strategy.

        Args:
            total: Number of elites to select

        Returns:
            List of selected elite programs
        """
        ...

    @abstractmethod
    async def get_program_ids(self) -> list[Program]:
        """
        Get all programs managed by this strategy.

        Returns:
            List of all Program objects in the strategy
        """
        ...

    async def remove_program_by_id(self, program_id: str) -> bool:
        """
        Remove a program from the strategy by ID.

        Args:
            program_id: ID of the program to remove

        Returns:
            True if program was removed, False if not found
        """
        raise NotImplementedError("Strategy does not support program removal")

    # Optional capabilities - strategies can override these for enhanced functionality

    async def get_metrics(self) -> StrategyMetrics | None:
        """
        Get strategy-specific metrics.

        Returns:
            StrategyMetrics object or None if not supported
        """
        return None

    async def cleanup(self) -> None:
        """
        Perform cleanup operations.

        Override this method if strategy supports cleanup operations.
        """

    async def pause(self) -> None:
        """
        Pause strategy operations.

        Override this method if strategy supports pause/resume.
        """

    async def resume(self) -> None:
        """
        Resume strategy operations.

        Override this method if strategy supports pause/resume.
        """
