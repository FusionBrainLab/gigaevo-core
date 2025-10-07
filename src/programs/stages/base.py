from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.programs.constants import DEFAULT_STAGE_TIMEOUT
from src.programs.core_types import ProgramStageResult
from src.programs.stages.guard import stage_guard

if TYPE_CHECKING:
    from src.programs.program import Program


class Stage:
    """Base class for DAG stages with uniform timeout/error handling.

    Subclasses must implement:
        `_execute_stage(self, program, started_at) -> ProgramStageResult`

    Inputs are always validated against `mandatory_inputs()` before running.
    """

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_STAGE_TIMEOUT,
        stage_name: str | None = None,
    ):
        self.timeout = float(timeout)
        self.stage_name = stage_name or self.__class__.__name__
        self._named_inputs: dict[str, Any] = {}

        logger.debug("[{stage}] init timeout={t:.2f}s", stage=self.stage_name, t=self.timeout)

    async def run(self, program: "Program") -> ProgramStageResult:
        """Execute stage via the shared stage_guard wrapper."""
        self._ensure_mandatory_inputs_present()
        return await stage_guard(self, program)

    async def _execute_stage(
        self, program: "Program", started_at: datetime
    ) -> ProgramStageResult:
        """Override this method in subclasses to implement stage logic."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement _execute_stage()")

    def set_named_inputs(self, inputs: dict[str, Any]) -> None:
        self._named_inputs = dict(inputs)

    def get_input(self, input_name: str) -> Any:
        if input_name not in self._named_inputs:
            raise KeyError(
                f"[{self.stage_name}] Missing input '{input_name}'. "
                f"Mandatory: {self.__class__.mandatory_inputs()} | "
                f"Optional: {self.__class__.optional_inputs()} | "
                f"Available: {list(self._named_inputs.keys())}"
            )
        return self._named_inputs[input_name]

    def get_input_optional(self, input_name: str, default: Any = None) -> Any:
        return self._named_inputs.get(input_name, default)

    def get_all_inputs(self) -> dict[str, Any]:
        return dict(self._named_inputs)

    @classmethod
    def mandatory_inputs(cls) -> list[str]:
        return []

    @classmethod
    def optional_inputs(cls) -> list[str]:
        return []

    def _ensure_mandatory_inputs_present(self) -> None:
        missing = [n for n in self.__class__.mandatory_inputs() if n not in self._named_inputs]
        if missing:
            raise KeyError(
                f"[{self.stage_name}] Missing mandatory inputs: {missing}. "
                f"Available: {list(self._named_inputs.keys())}. "
                f"Optional: {self.__class__.optional_inputs()}"
            )


