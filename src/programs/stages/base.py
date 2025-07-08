"""Base classes for MetaEvolve stages."""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

from loguru import logger
import psutil
from copy import deepcopy

from src.exceptions import (
    MetaEvolveError,
    ResourceError,
    SecurityViolationError,
    ValidationError,
    ensure_not_none,
    ensure_positive,
)
from src.programs.constants import DEFAULT_STAGE_TIMEOUT
from src.programs.stages.state import ProgramStageResult, StageState
from src.programs.utils import build_stage_result
from .prometheus import StagePrometheusExporter

if TYPE_CHECKING:
    from src.programs.program import Program


@dataclass
class StageMetrics:
    executions: int = 0
    successes: int = 0
    failures: int = 0
    timeouts: int = 0
    security_violations: int = 0
    total_time: float = 0.0
    max_memory: int = 0
    error_types: Dict[str, int] = field(default_factory=dict)

    def record_execution(
        self,
        duration: float,
        success: bool,
        error_type: Optional[str] = None,
        memory_used: int = 0,
    ):
        """Record stage execution metrics."""
        self.executions += 1
        self.total_time += duration
        self.max_memory = max(self.max_memory, memory_used)

        if success:
            self.successes += 1
        else:
            self.failures += 1
            if error_type:
                self.error_types[error_type] = (
                    self.error_types.get(error_type, 0) + 1
                )

    def record_timeout(self):
        """Record timeout occurrence."""
        self.timeouts += 1

    def record_security_violation(self):
        """Record security violation."""
        self.security_violations += 1

    def get_success_rate(self) -> float:
        """Get success rate as percentage."""
        return (self.successes / max(1, self.executions)) * 100

    def get_average_time(self) -> float:
        """Get average execution time."""
        return self.total_time / max(1, self.executions)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "executions": self.executions,
            "successes": self.successes,
            "failures": self.failures,
            "timeouts": self.timeouts,
            "security_violations": self.security_violations,
            "success_rate": self.get_success_rate(),
            "average_time": self.get_average_time(),
            "max_memory": self.max_memory,
            "error_types": dict(self.error_types),
        }


class Stage:
    """
    Base class for bulletproof DAG stages with comprehensive error handling.

    All stages inherit from this base class and must implement the run() method.
    This base class provides:
    - Error handling and recovery
    - Resource monitoring
    - Security validation
    - Performance metrics
    - Timeout management
    """

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_STAGE_TIMEOUT,
        max_memory_mb: int = 256,
        enable_security_checks: bool = True,
        enable_resource_monitoring: bool = True,
        stage_name: Optional[str] = None,
    ):
        self.timeout = ensure_positive(timeout, "timeout")
        self.max_memory_mb = ensure_positive(max_memory_mb, "max_memory_mb")
        self.enable_security_checks = enable_security_checks
        self.enable_resource_monitoring = enable_resource_monitoring
        self.stage_name = stage_name or self.__class__.__name__
        self.metrics = StageMetrics()

        logger.debug(
            f"[{self.stage_name}] Initialized with timeout={timeout}s, max_memory={max_memory_mb}MB"
        )

    async def run(self, program: "Program") -> ProgramStageResult:
        """Execute stage via the shared :pyfunc:`stage_guard` wrapper."""
        from .guard import stage_guard

        return await stage_guard(self, program)

    async def _execute_stage(
        self, program: "Program", started_at: datetime
    ) -> ProgramStageResult:
        """
        Override this method in subclasses to implement stage logic.
        This method should return a ProgramStageResult directly.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _execute_stage() method"
        )

    async def _validate_program(self, program: "Program") -> None:
        """Validate program before execution."""
        ensure_not_none(program, "program")

        if not program.id:
            raise ValidationError("Program must have an ID", field="id")

        if (
            not program.code
            and hasattr(self, "_requires_code")
            and self._requires_code
        ):
            raise ValidationError(
                "Program must have code", field="code", value=program.code
            )

    async def _validate_result(self, result: ProgramStageResult) -> None:
        """Validate stage result."""
        ensure_not_none(result, "result")

        if not hasattr(result, "status"):
            raise ValidationError(
                "Stage result must have status", field="status"
            )

        if result.status not in [
            StageState.COMPLETED,
            StageState.FAILED,
            StageState.RUNNING,
        ]:
            raise ValidationError(
                f"Invalid stage status: {result.status}",
                field="status",
                value=result.status,
            )

    @asynccontextmanager
    async def _resource_monitor(self, program: "Program"):
        """Context manager for resource monitoring."""
        if not self.enable_resource_monitoring:
            yield
            return

        process = psutil.Process()
        initial_memory = process.memory_info().rss
        initial_cpu_time = process.cpu_times()

        try:
            yield
        finally:
            try:
                final_memory = process.memory_info().rss
                final_cpu_time = process.cpu_times()

                memory_delta = final_memory - initial_memory
                cpu_delta = (final_cpu_time.user + final_cpu_time.system) - (
                    initial_cpu_time.user + initial_cpu_time.system
                )

                logger.debug(
                    f"[{self.stage_name}] Program {program.id}: "
                    f"Memory delta: {memory_delta / (1024*1024):.1f}MB, "
                    f"CPU time: {cpu_delta:.2f}s"
                )

                # Check memory limit after execution
                memory_mb = memory_delta / (1024 * 1024)
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

    def clone(self) -> "Stage":
        """Create a fresh copy of this stage with reset metrics."""
        # Get constructor arguments by filtering out runtime-only attributes
        cp = deepcopy(self)
        cp.metrics = StageMetrics()
        return cp
