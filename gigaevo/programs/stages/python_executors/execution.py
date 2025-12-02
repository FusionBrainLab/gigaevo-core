from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

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


class PythonCodeExecutor(Stage):
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
    OutputModel = Box[Any]

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
                timeout=int(self.timeout),
                max_memory_mb=self.max_memory_mb,
                max_output_size=self.max_output_size,
            )

            del stdout_bytes
            del stderr_text

            return self.__class__.OutputModel(data=value)

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
    context: Optional[AnyContainer]


@StageRegistry.register(
    description="Call a function defined in Program.code, wiring optional DAG input."
)
class CallProgramFunction(PythonCodeExecutor):
    """Calls the user function in Program.code. Accepts optional input from DAG wiring."""

    InputsModel = ContextInputModel
    OutputModel = Box[Any]

    def _build_call(self, program: Program):
        params: ContextInputModel = self.params
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
    context: Optional[AnyContainer]


@StageRegistry.register(
    description="Call a validator function from a Python file on program output (+ optional context)."
)
class CallValidatorFunction(PythonCodeExecutor):
    """Loads validator file and calls function `validate(context?, program_output)`."""

    InputsModel = ValidatorInput
    OutputModel = Box[dict[str, float]]

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

    def _build_call(self, program: Program):
        params: ValidatorInput = self.params
        payload = params.payload.data
        if params.context is not None:
            context = params.context.data
        else:
            context = None
        return ([context, payload] if context is not None else [payload]), {}
