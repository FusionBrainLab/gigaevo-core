from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import time
import types
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
from gigaevo.programs.stages.cache_handler import (
    DEFAULT_CACHE,
    CacheHandler,
)

if TYPE_CHECKING:
    from gigaevo.programs.program import Program

I = TypeVar("I", bound=StageIO)  # noqa: E741
O = TypeVar("O", bound=StageIO)  # noqa: E741


def _is_optional_type(tp: Any) -> bool:
    """Check if a type annotation represents an optional type (allows None).

    Handles both:
      - typing.Optional[X] / typing.Union[X, None]
      - X | None (Python 3.10+ union syntax using types.UnionType)

    Fields with optional types will be automatically set to None when not
    provided via DAG data flow edges. This ensures consistent behavior
    between stage execution and cache hash computation.

    Examples:
        >>> _is_optional_type(Optional[str])
        True
        >>> _is_optional_type(str | None)  # Python 3.10+
        True
        >>> _is_optional_type(Union[str, int, None])
        True
        >>> _is_optional_type(str)
        False
    """
    origin = get_origin(tp)

    # Handle typing.Union (includes Optional[X] which is Union[X, None])
    if origin is Union:
        return any(arg is type(None) for arg in get_args(tp))  # noqa: E721

    # Handle Python 3.10+ union syntax: X | None (types.UnionType)
    if isinstance(tp, types.UnionType):
        return any(arg is type(None) for arg in get_args(tp))  # noqa: E721

    return False


class Stage:
    """
    Minimal, typed stage API (strict; one StageIO base for Inputs/Outputs).

    Subclasses MUST define:
        InputsModel: Type[StageIO]   (fields with Optional[...] are optional inputs)
        OutputModel: Type[StageIO]   (use VoidOutput for no-output stages)

    Optional Input Fields:
        Fields annotated with Optional[X] or X | None are considered optional.
        When not provided via DAG data flow edges, they are automatically set
        to None. This applies to both stage execution and cache hash computation.

        Example InputsModel with optional field:
            class MyInputs(StageIO):
                required_field: SomeType      # Must be provided via DAG edge
                optional_field: Optional[X]   # Set to None if no edge provides it

        IMPORTANT: Use Optional[X] for fields that may not have a DAG edge.
        Without Optional, missing fields will cause validation errors.

    Public surface:
        - timeout: float
        - cache_handler: CacheHandler (controls caching behavior)
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

    # Caching behavior
    cache_handler: ClassVar[CacheHandler] = DEFAULT_CACHE

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
        self._current_inputs_hash: Optional[str] = None

    @property
    def stage_name(self) -> str:
        return self.__class__.__name__

    def get_cache_handler(self) -> CacheHandler:
        """Get the cache handler for this stage."""
        return self.__class__.cache_handler

    def compute_inputs_hash(self) -> str | None:
        """Compute hash of current inputs for cache invalidation."""
        return self.compute_hash(self.params)

    @classmethod
    def compute_hash(cls, params: StageIO) -> str | None:
        """Compute hash from validated params object.

        Override this to customize hashing logic (e.g. ignore certain fields).
        """
        return params.content_hash

    @classmethod
    def _normalize_inputs(cls, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize raw inputs by setting missing optional fields to None.

        This ensures consistent hash computation between execution time and
        cache check time. Without this, optional fields missing from inputs
        would cause Pydantic validation to fail during cache checks.
        """
        payload = dict(inputs)
        for name in cls._optional_names:
            if name not in payload:
                payload[name] = None
        return payload

    @classmethod
    def compute_hash_from_inputs(cls, inputs: Mapping[str, Any]) -> str | None:
        """Compute hash from raw inputs without instantiating the stage.

        Normalizes inputs first (setting optional fields to None) to ensure
        the hash matches what would be computed during actual execution.
        """
        try:
            normalized = cls._normalize_inputs(inputs)
            params = cls.InputsModel.model_validate(normalized)
            return cls.compute_hash(params)
        except Exception:
            # If validation fails, we can't compute a hash
            return None

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
        # Use shared normalization to ensure consistency with hash computation
        self._raw_inputs = self.__class__._normalize_inputs(payload)
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

        # Compute inputs hash before execution (for cache handler)
        self._current_inputs_hash = self.compute_inputs_hash()

        try:
            self._ensure_required_present()
            result = await asyncio.wait_for(self.compute(program), timeout=self.timeout)

            # Pass-through if already a ProgramStageResult
            if isinstance(result, ProgramStageResult):
                if result.started_at is None:
                    result.started_at = started_at
                if result.finished_at is None and result.status in FINAL_STATES:
                    result.finished_at = datetime.now(timezone.utc)
                # Let cache handler augment result
                result = self.get_cache_handler().on_complete(
                    result, self._current_inputs_hash
                )
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
                    ok = self.get_cache_handler().on_complete(
                        ok, self._current_inputs_hash
                    )
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
            ok = self.get_cache_handler().on_complete(ok, self._current_inputs_hash)
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
        finally:
            self._raw_inputs.clear()
            self._params_obj = None
            self._current_inputs_hash = None

    async def compute(self, program: "Program") -> O | ProgramStageResult | None:
        """Override in subclasses."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement compute()")
