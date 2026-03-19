from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeVar, cast

from loguru import logger

from gigaevo.exceptions import ValidationError
from gigaevo.programs.core_types import (
    ProgramStageResult,
    StageError,
    StageIO,
    VoidInput,
)
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.common import AnyContainer, Box
from gigaevo.programs.stages.python_executors.wrapper import (
    ExecRunnerError,
    run_exec_runner,
)
from gigaevo.programs.stages.stage_registry import StageRegistry
from gigaevo.programs.utils import dedent_code

T = TypeVar("T")


class PythonCodeExecutor[T](Stage):
    """
    Execute a user function from dynamic code in an isolated subprocess.

    The subprocess has resource limits applied for safety:
    - Memory limits (via resource.RLIMIT_AS) prevent RAM exhaustion
    - Timeout limits prevent infinite loops
    - Output size limits prevent excessive data generation

    The output is a Box[T] containing the result of the function call.

    Args:
        function_name: Name of the function to call in the user code
        python_path: Additional paths to add to sys.path
        max_output_size: Maximum size of output in bytes (default: 64MB)
        max_memory_mb: Maximum memory in MB (default: None = unlimited)
        timeout: Maximum execution time in seconds (inherited from Stage)

    Subclasses must implement `_build_call(self, program) -> (args, kwargs)`.
    """

    InputsModel = VoidInput
    OutputModel = Box[T]

    def __init__(
        self,
        *,
        function_name: str = "run_code",
        python_path: list[Path] | None = None,
        max_output_size: int = 64 * 1024 * 1024,
        max_memory_mb: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.function_name = function_name
        self.python_path = python_path or []
        self.max_output_size = int(max_output_size)
        self.max_memory_mb = int(max_memory_mb) if max_memory_mb is not None else None

    def _code_str(self, program: Program) -> str:
        return program.code

    def _build_call(self, program: Program) -> tuple[Sequence[Any], dict[str, Any]]:
        return (), {}

    def parse_output(self, x: Any) -> T:
        """Parse raw subprocess return value; override in subclasses."""
        return cast(T, x)

    async def compute(self, program: Program) -> ProgramStageResult | Box[Any]:
        stage_name = self.__class__.__name__
        code_str = self._code_str(program)
        args, kwargs = self._build_call(program)

        logger.debug(
            "[{}] calling '{}' with {} arg(s), {} kwarg(s)",
            stage_name,
            self.function_name,
            len(args),
            len(kwargs),
        )

        try:
            value, stdout_bytes, stderr_text = await run_exec_runner(
                code=dedent_code(code_str),
                function_name=self.function_name,
                args=args,
                kwargs=kwargs,
                python_path=self.python_path,
                env_updates={
                    "GIGAEVO_PROGRAM_ID": program.id,
                    "GIGAEVO_PROGRAM_ID_SHORT": program.id[:8],
                },
                timeout=int(self.timeout),
                max_memory_mb=self.max_memory_mb,
                max_output_size=self.max_output_size,
            )

            value_parsed = self.parse_output(value)

            del stdout_bytes
            del stderr_text

            return self.__class__.OutputModel(data=value_parsed)

        except ExecRunnerError as e:
            # Detect memory limit errors
            error_type = "SubprocessError"
            error_msg = str(e)

            if e.stderr and (
                "MemoryError" in e.stderr or "Cannot allocate memory" in e.stderr
            ):
                error_type = "MemoryLimitExceeded"
                error_msg = (
                    f"Process exceeded memory limit of {self.max_memory_mb} MB"
                    if self.max_memory_mb
                    else "Process ran out of memory"
                )

            return ProgramStageResult.failure(
                error=StageError(
                    type=error_type,
                    message=error_msg,
                    stage=stage_name,
                    traceback=e.stderr,
                )
            )
        except Exception as e:
            return ProgramStageResult.failure(
                error=StageError.from_exception(e, stage=stage_name)
            )


class ContextInputModel(StageIO):
    context: AnyContainer | None


@StageRegistry.register(
    description="Call a function defined in Program.code, wiring optional DAG input."
)
class CallProgramFunction(PythonCodeExecutor):
    """Calls the user function in Program.code. Accepts optional input from DAG wiring."""

    InputsModel = ContextInputModel
    OutputModel = Box[Any]

    def _build_call(self, program: Program) -> tuple[Sequence[Any], dict[str, Any]]:
        params = cast(ContextInputModel, self.params)
        args: list[Any] = []
        context: AnyContainer | None = params.context
        if context is not None:
            args.append(context.data)
        return args, {}


@StageRegistry.register(
    description="Call a function in Program.code with fixed args provided at construction."
)
class CallProgramFunctionWithFixedArgs(PythonCodeExecutor):
    """Calls the user function in Program.code with fixed args (and/or kwargs) supplied when building the stage."""

    OutputModel = Box[Any]

    def __init__(
        self,
        *,
        args: Sequence[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        **kw: Any,
    ):
        super().__init__(**kw)
        self._fixed_args = list(args or [])
        self._fixed_kwargs = dict(kwargs or {})

    def _build_call(self, program: Program) -> tuple[Sequence[Any], dict[str, Any]]:
        return self._fixed_args, self._fixed_kwargs


@StageRegistry.register(
    description="Call a function defined in a Python file (no DAG inputs)."
)
class CallFileFunction(PythonCodeExecutor):
    """Loads Python code from a file and calls a function (default: build_context)."""

    OutputModel = Box[Any]

    def __init__(
        self, *, path: Path, function_name: str = "build_context", **kwargs: Any
    ):
        super().__init__(
            function_name=function_name, python_path=[Path(path).parent], **kwargs
        )
        p = Path(path)
        if not p.exists():
            raise ValidationError(f"Python file not found: {p}")
        try:
            self._file_code = p.read_text(encoding="utf-8")
        except OSError as e:
            raise ValidationError(f"Failed to read file: {e}") from e

    def _code_str(self, program: Program) -> str:
        return self._file_code

    def _build_call(self, program: Program) -> tuple[Sequence[Any], dict[str, Any]]:
        return [], {}


class ValidatorInput(StageIO):
    payload: AnyContainer
    context: AnyContainer | None


ValidatorOutput = Box[tuple[dict[str, float], Any]]


@StageRegistry.register(
    description="Call a validator function from a Python file on program output (+ optional context)."
)
class CallValidatorFunction(PythonCodeExecutor):
    """Loads validator file and calls function `validate(context?, program_output)`."""

    InputsModel = ValidatorInput
    OutputModel = ValidatorOutput  # metrics dict and execution artifact

    def __init__(self, *, path: Path, function_name: str = "validate", **kwargs: Any):
        super().__init__(
            function_name=function_name, python_path=[Path(path).parent], **kwargs
        )
        p = Path(path)
        if not p.exists():
            raise ValidationError(f"Validator file not found: {p}")
        try:
            self._validator_code = p.read_text(encoding="utf-8")
        except OSError as e:
            raise ValidationError(f"Failed to read validator file: {e}") from e

    def _code_str(self, program: Program) -> str:
        return self._validator_code

    def parse_output(self, x: Any) -> tuple[dict[str, float], Any]:
        # Validate the return type early to give a clear error pointing back to
        # validate().  Allowing None or non-dict/tuple values through produces
        # cryptic downstream errors (e.g. "TypeError: {**None}") with no hint
        # that the problem is in the researcher's validate() return value.
        if isinstance(x, tuple):
            return x
        if not isinstance(x, dict):
            raise ValueError(
                f"validate() must return a dict or (dict, artifact) tuple, "
                f"got {type(x).__name__!r}: {x!r}"
            )
        return (x, None)

    def _build_call(self, program: Program) -> tuple[Sequence[Any], dict[str, Any]]:
        params = cast(ValidatorInput, self.params)
        payload = params.payload.data
        if params.context is not None:
            context = params.context.data
        else:
            context = None
        return ([context, payload] if context is not None else [payload]), {}


class ValidationResult(StageIO):
    validation_result: ValidatorOutput


@StageRegistry.register(
    description="Extract metrics dict from a validation result (ValidatorOutput)."
)
class FetchMetrics(Stage):
    InputsModel = ValidationResult
    OutputModel = Box[dict[str, float]]

    async def compute(self, program: Program) -> Box[dict[str, float]]:
        params = cast(ValidationResult, self.params)
        return Box[dict[str, float]](data=params.validation_result.data[0])


@StageRegistry.register(
    description="Extract execution artifact from a validation result (ValidatorOutput)."
)
class FetchArtifact(Stage):
    InputsModel = ValidationResult
    OutputModel = Box[Any]

    async def compute(self, program: Program) -> Box[Any]:
        params = cast(ValidationResult, self.params)
        return Box[Any](data=params.validation_result.data[1])
