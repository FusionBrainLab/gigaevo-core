"""Base classes for MetaEvolve stages."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

from loguru import logger
import psutil

from src.exceptions import (
    ResourceError,
    ensure_positive,
)
from src.programs.constants import DEFAULT_STAGE_TIMEOUT
from src.programs.stages.guard import stage_guard
from src.programs.core_types import ProgramStageResult

if TYPE_CHECKING:
    from src.programs.program import Program


@dataclass
class StageMetrics:
    executions: int = 0
    successes: int = 0
    failures: int = 0
    total_time: float = 0.0

    def record_execution(self, duration: float, success: bool):
        """Record stage execution metrics."""
        self.executions += 1
        self.total_time += duration
        if success:
            self.successes += 1
        else:
            self.failures += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "executions": self.executions,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": (self.successes / max(1, self.executions)) * 100,
            "average_time": self.total_time / max(1, self.executions),
        }


class Stage:
    """Base class for DAG stages with error handling and resource monitoring."""

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_STAGE_TIMEOUT,
        max_memory_mb: int = 256,
        enable_resource_monitoring: bool = True,
        stage_name: Optional[str] = None,
    ):
        self.timeout = ensure_positive(timeout, "timeout")
        self.max_memory_mb = ensure_positive(max_memory_mb, "max_memory_mb")
        self.enable_resource_monitoring = enable_resource_monitoring
        self.stage_name = stage_name or self.__class__.__name__
        self.metrics = StageMetrics()
        self._named_inputs: Dict[str, Any] = {}

        logger.debug(
            f"[{self.stage_name}] Initialized with timeout={timeout}s, max_memory={max_memory_mb}MB"
        )

    async def run(self, program: "Program") -> ProgramStageResult:
        """Execute stage via the shared stage_guard wrapper."""

        return await stage_guard(self, program)

    async def _execute_stage(
        self, program: "Program", started_at: datetime
    ) -> ProgramStageResult:
        """Override this method in subclasses to implement stage logic."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _execute_stage() method"
        )

    @asynccontextmanager
    async def _resource_monitor(self, program: "Program"):
        """Context manager for resource monitoring."""
        if not self.enable_resource_monitoring:
            yield
            return

        process = psutil.Process()
        initial_memory = process.memory_info().rss

        try:
            yield
        finally:
            try:
                memory_delta = process.memory_info().rss - initial_memory
                memory_mb = memory_delta / (1024 * 1024)

                logger.debug(
                    f"[{self.stage_name}] Program {program.id}: "
                    f"Memory delta: {memory_mb:.1f}MB"
                )

                if memory_mb > self.max_memory_mb:
                    raise ResourceError(
                        f"Memory limit exceeded: {memory_mb:.1f}MB > {self.max_memory_mb}MB",
                        resource_type="memory",
                        limit=self.max_memory_mb,
                    )

            except Exception as e:
                logger.warning(
                    f"[{self.stage_name}] Resource monitoring error: {e}"
                )

    def get_metrics(self) -> Dict[str, Any]:
        """Get stage execution metrics."""
        return self.metrics.to_dict()

    def set_named_inputs(self, inputs: Dict[str, Any]) -> None:
        """Set the named inputs for this stage.
        
        Args:
            inputs: Dictionary mapping input names to their values
        """
        self._named_inputs = inputs or {}

    def get_input(self, input_name: str) -> Any:
        """Get input by semantic name.
        
        Args:
            input_name: The semantic name of the input as declared in mandatory_inputs() or optional_inputs()
            
        Returns:
            The input data for the named input
            
        Raises:
            KeyError: If the input_name is not found
        """
        if input_name not in self._named_inputs:
            available = list(self._named_inputs.keys())
            declared_mandatory = self.__class__.mandatory_inputs()
            declared_optional = self.__class__.optional_inputs()
            raise KeyError(
                f"Input '{input_name}' not found. Available inputs: {available}. "
                f"Declared mandatory inputs: {declared_mandatory}, optional inputs: {declared_optional}"
            )
        return self._named_inputs[input_name]

    def get_input_optional(self, input_name: str, default: Any = None) -> Any:
        """Get input by semantic name, returning default if not present.
        
        Args:
            input_name: The semantic name of the input
            default: Value to return if input is not present
            
        Returns:
            The input data or default value
        """
        return self._named_inputs.get(input_name, default)

    def get_all_inputs(self) -> Dict[str, Any]:
        """Get all named inputs.
        
        Returns:
            Dictionary of all available inputs
        """
        return self._named_inputs.copy()

    @classmethod
    def mandatory_inputs(cls) -> list[str]:
        """Return list of mandatory input names.
        
        These inputs must be provided for the stage to run successfully.
        
        Returns:
            List of mandatory input names
        """
        return []

    @classmethod
    def optional_inputs(cls) -> list[str]:
        """Return list of optional input names.
        
        These inputs may be provided but are not required for the stage to run.
        
        Returns:
            List of optional input names
        """
        return []


