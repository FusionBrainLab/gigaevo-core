from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import time
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Mapping,
    Optional,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from loguru import logger
from pydantic import ValidationError as PydanticValidationError

from gigaevo.programs.core_types import (
    FINAL_STATES,
    ProgramStageResult,
    StageError,
    StageIO,
    VoidOutput,
)

if TYPE_CHECKING:
    from gigaevo.programs.program import Program

I = TypeVar("I", bound=StageIO)  # noqa: E741
O = TypeVar("O", bound=StageIO)  # noqa: E741


def _is_optional_type(tp: Any) -> bool:
    origin = get_origin(tp)
    if origin is Union:
        return any(arg is type(None) for arg in get_args(tp))  # noqa: E721
    return False


class Stage:
    """
    Minimal, typed stage API (strict; one StageIO base for Inputs/Outputs).

    Subclasses MUST define:
        InputsModel: Type[StageIO]   (fields with Optional[...] are optional inputs)
        OutputModel: Type[StageIO]   (use VoidOutput for no-output stages)

    Public surface:
        - timeout: float
        - cacheable: ClassVar[bool]
        - attach_inputs(data: Mapping[str, Any]) -> None
        - params: InputsModel            (read-only; validated)
        - execute(program) -> ProgramStageResult
        - required_fields() / optional_fields()

    Subclasses implement:
        - compute(program) -> OutputModel | ProgramStageResult | None
          (None allowed only if OutputModel is VoidOutput)
    """

    InputsModel: ClassVar[Type[I]]
    OutputModel: ClassVar[Type[O]]
    cacheable: ClassVar[bool] = True

    _required_names: ClassVar[list[str]]
    _optional_names: ClassVar[list[str]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not hasattr(cls, "InputsModel") or cls.InputsModel is None:
            raise TypeError(f"{cls.__name__} must define InputsModel = Type[StageIO]")
        if not hasattr(cls, "OutputModel") or cls.OutputModel is None:
            raise TypeError(f"{cls.__name__} must define OutputModel = Type[StageIO]")

        if not issubclass(cls.InputsModel, StageIO):  # type: ignore[arg-type]
            raise TypeError(f"{cls.__name__}.InputsModel must inherit from StageIO")
        if not issubclass(cls.OutputModel, StageIO):  # type: ignore[arg-type]
            raise TypeError(f"{cls.__name__}.OutputModel must inherit from StageIO")

        req, opt = [], []
        for name, field in cls.InputsModel.model_fields.items():  # type: ignore[attr-defined]
            (
                (_ := opt.append(name))
                if _is_optional_type(field.annotation)
                else req.append(name)
            )
        cls._required_names, cls._optional_names = req, opt

    def __init__(self, *, timeout: float):
        self.timeout = timeout
        self._raw_inputs: dict[str, Any] = {}
        self._params_obj: Optional[I] = None

    @property
    def stage_name(self) -> str:
        return self.__class__.__name__

    @classmethod
    def required_fields(cls) -> list[str]:
        return list(cls._required_names)

    @classmethod
    def optional_fields(cls) -> list[str]:
        return list(cls._optional_names)

    def attach_inputs(self, data: Mapping[str, Any]) -> None:
        declared = set(self.__class__.InputsModel.model_fields.keys())  # type: ignore[attr-defined]
        payload = dict(data)
        extras = set(payload.keys()) - declared
        if extras:
            raise KeyError(
                f"[{self.stage_name}] Unknown input fields: {sorted(extras)}; allowed={sorted(declared)}"
            )
        for n in self.__class__._optional_names:
            if n not in payload:
                payload[n] = None
        self._raw_inputs = payload
        self._params_obj = None

    @property
    def params(self) -> I:
        if self._params_obj is None:
            try:
                self._params_obj = self.__class__.InputsModel.model_validate(
                    self._raw_inputs
                )  # type: ignore[assignment]
            except PydanticValidationError as exc:
                raise KeyError(
                    f"[{self.stage_name}] Input validation failed: {exc.errors()}"
                ) from exc
        return self._params_obj

    def _ensure_required_present(self) -> None:
        missing = [
            n for n in self.__class__._required_names if n not in self._raw_inputs
        ]
        if missing:
            raise KeyError(
                f"[{self.stage_name}] Missing required inputs: {missing}. "
                f"Available: {list(self._raw_inputs.keys())}. "
                f"Optional: {self.__class__.optional_fields()}"
            )

    async def execute(self, program: "Program") -> ProgramStageResult:
        started_at = datetime.now(timezone.utc)
        t0 = time.monotonic()
        logger.info(f"[{self.stage_name}] Executing for {program.id[:8]}")

        try:
            self._ensure_required_present()
            result = await asyncio.wait_for(self.compute(program), timeout=self.timeout)

            # Pass-through if already a ProgramStageResult
            if isinstance(result, ProgramStageResult):
                if result.started_at is None:
                    result.started_at = started_at
                if result.finished_at is None and result.status in FINAL_STATES:
                    result.finished_at = datetime.now(timezone.utc)
                logger.debug(
                    "[{stage}] ok (pass-through) in {dur:.2f}s",
                    stage=self.stage_name,
                    dur=(time.monotonic() - t0),
                )
                return result

            # None → only legal for VoidOutput stages
            if result is None:
                if self.__class__.OutputModel is VoidOutput:
                    ok = ProgramStageResult.success(started_at=started_at)
                    logger.debug(
                        "[{stage}] ok (void) in {dur:.2f}s",
                        stage=self.stage_name,
                        dur=(time.monotonic() - t0),
                    )
                    return ok
                raise TypeError(
                    f"{self.stage_name} returned None but OutputModel is not VoidOutput"
                )

            # Normal case: got a StageIO instance
            if not isinstance(result, self.__class__.OutputModel):
                raise TypeError(
                    f"{self.stage_name} must return {self.__class__.OutputModel.__name__} "
                    f"or ProgramStageResult (got {type(result).__name__})"
                )

            ok = ProgramStageResult.success(output=result, started_at=started_at)
            logger.debug(
                "[{stage}] ok in {dur:.2f}s",
                stage=self.stage_name,
                dur=(time.monotonic() - t0),
            )
            return ok

        except Exception as exc:
            logger.exception(
                "[{stage}] Failed after {dur:.2f}s",
                stage=self.stage_name,
                dur=(time.monotonic() - t0),
            )
            return ProgramStageResult.failure(
                error=StageError.from_exception(exc, stage=self.stage_name),
                started_at=started_at,
            )

    async def compute(self, program: "Program") -> O | ProgramStageResult | None:
        """Override in subclasses."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement compute()")
