"""Code execution stages for MetaEvolve."""

import base64
from datetime import datetime
from pathlib import Path
import pickle
import resource
import tempfile
from typing import Any, List, Optional

from loguru import logger

from src.exceptions import (
    ProgramExecutionError,
    StageError,
    ValidationError,
)
from src.programs.program import Program, ProgramStageResult, StageState
from src.programs.utils import (
    build_stage_result,
    construct_exec_code,
    dedent_code,
    run_python_snippet,
)
from src.programs.stages.base import Stage
from src.runner.stage_registry import StageRegistry

INPUT_SIZE_THRESHOLD = 8 * 1024


class PythonCodeExecutor(Stage):
    def __init__(
        self,
        function_name: str = "run_code",
        python_path: Optional[List[Path]] = None,
        code: Optional[str] = None,
        enable_sandboxing: bool = False,
        max_output_size: int = 1024 * 1024 * 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.function_name = function_name
        self.python_path = python_path or []
        self.enable_sandboxing = enable_sandboxing
        self.max_output_size = max_output_size
        self.code = code

    async def _run_with(
        self,
        *,
        program: Program,
        started_at: datetime,
        code_str: str,
        argument: Optional[Any],
    ) -> ProgramStageResult:
        logger.debug(
            f"[{self.stage_name}] Program {program.id}: Executing Python code"
        )

        # Prepare input
        input_b64, input_file_path = None, None
        temp_file_path = None

        if argument is not None:
            try:
                input_pickle = pickle.dumps(argument)
                input_b64_bytes = base64.b64encode(input_pickle)

                if len(input_b64_bytes) > INPUT_SIZE_THRESHOLD:
                    temp_file = tempfile.NamedTemporaryFile(
                        mode="wb", delete=False
                    )
                    temp_file.write(input_pickle)
                    temp_file.close()
                    input_file_path = temp_file.name
                    temp_file_path = temp_file.name
                else:
                    input_b64 = input_b64_bytes.decode("utf-8")

            except Exception as e:
                raise ProgramExecutionError(
                    f"Failed to serialize input object: {e}",
                    program_id=program.id,
                    cause=e,
                ) from e

        try:
            # Execute code
            input_code = code_str
            exec_code = construct_exec_code(
                user_code=dedent_code(input_code),
                function_name=self.function_name,
                input_b64=input_b64,
                input_file_path=input_file_path,
                python_path=self.python_path,
            )

            if self.enable_sandboxing:
                result = await self._run_sandboxed(
                    exec_code, started_at, program.id
                )
            else:
                result = await run_python_snippet(
                    exec_code,
                    started_at,
                    timeout=self.timeout,
                    stage_name=self.stage_name,
                )

            # Validate output size
            if result.output is not None:
                size = len(pickle.dumps(result.output))
                if size > self.max_output_size:
                    return build_stage_result(
                        status=StageState.FAILED,
                        started_at=started_at,
                        error=f"Output too large: {size} > {self.max_output_size}",
                        stage_name=self.stage_name,
                        context=f"Output size limit exceeded: {size} bytes",
                    )

            return result

        except (ProgramExecutionError, RuntimeError, OSError) as e:
            return build_stage_result(
                status=StageState.FAILED,
                started_at=started_at,
                error=e,
                stage_name=self.stage_name,
                context="Code execution failed with unexpected error",
            )
        except Exception as e:  # pylint: disable=broad-except
            return build_stage_result(
                status=StageState.FAILED,
                started_at=started_at,
                error=e,
                stage_name=self.stage_name,
                context="Code execution failed (unclassified)",
            )
        finally:
            if temp_file_path:
                try:
                    Path(temp_file_path).unlink()
                except OSError as e:
                    logger.warning(
                        f"Failed to clean up temporary file {temp_file_path}: {e}"
                    )

    async def _run_sandboxed(
        self, code: str, started_at: datetime, program_id: str
    ) -> ProgramStageResult:
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                await self._set_resource_limits()
                return await run_python_snippet(
                    code, started_at, timeout=self.timeout, cwd=Path(temp_dir)
                )
            except Exception as e:
                raise ProgramExecutionError(
                    f"Sandboxed execution failed: {e}",
                    program_id=program_id,
                    cause=e,
                ) from e

    async def _set_resource_limits(self) -> None:
        try:
            memory_limit = 512 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (int(self.timeout * 2), int(self.timeout * 2)),
            )
            resource.setrlimit(
                resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024)
            )
        except (ValueError, OSError, AttributeError) as e:
            logger.warning(
                f"[{self.stage_name}] Failed to set resource limits: {e}"
            )

    async def _execute_stage(  # pragma: no cover
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        # Subclasses must map edges/fields to (code_str, argument)
        raise NotImplementedError


@StageRegistry.register(
    description="Execute program code with optional inputs"
)
class RunProgramCodeWithOptionalProducedData(PythonCodeExecutor):
    """Execute program.code with up to one optional data input from DAG edges."""

    @classmethod
    def mandatory_inputs(cls) -> list[str]:
        return []

    @classmethod
    def optional_inputs(cls) -> list[str]:
        return ["program_output", "context_data"]

    async def _execute_stage(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        # Get optional inputs
        program_output = self.get_input_optional("program_output")
        context_data = self.get_input_optional("context_data")
        
        # If we have context_data, pass it as the argument
        # If we have program_output, pass it as the argument
        # If we have both, pass them as a tuple
        if context_data is not None and program_output is not None:
            arg = (program_output, context_data)
        elif context_data is not None:
            arg = context_data
        else:
            arg = program_output
            
        code_str = program.code
        return await self._run_with(
            program=program,
            started_at=started_at,
            code_str=code_str,
            argument=arg,
        )

    # Inherit sandboxing and resource limits from base


class RunProgramCodeWithConstantData(PythonCodeExecutor):
    """Execute program.code with a constant data payload provided at construction."""

    def __init__(self, *, constant_data: Any, **kwargs):
        super().__init__(**kwargs)
        self._constant_data = constant_data

    @classmethod
    def mandatory_inputs(cls) -> list[str]:
        return []

    @classmethod
    def optional_inputs(cls) -> list[str]:
        return []

    async def _execute_stage(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        code_str = program.code
        return await self._run_with(
            program=program,
            started_at=started_at,
            code_str=code_str,
            argument=self._constant_data,
        )


@StageRegistry.register(
    description="Run constant Python code from a file"
)
class RunConstantPythonCode(PythonCodeExecutor):
    """Run a constant Python file function (e.g., build_context) with no inputs.

    Produces whatever the function returns as this stage's output.
    """

    def __init__(self, *, context_path: Path, function_name: str = "build_context", **kwargs):
        self.context_path = Path(context_path)
        if not self.context_path.exists():
            raise ValidationError(f"Context file not found: {self.context_path}")
        try:
            self.context_code = self.context_path.read_text(encoding="utf-8")
        except OSError as e:
            raise ValidationError(f"Failed to read context file: {e}") from e

        super().__init__(
            function_name=function_name,
            python_path=[self.context_path.parent],
            **kwargs,
        )

    @classmethod
    def mandatory_inputs(cls) -> list[str]:
        return []

    @classmethod
    def optional_inputs(cls) -> list[str]:
        return []

    async def _execute_stage(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        return await self._run_with(
            program=program,
            started_at=started_at,
            code_str=self.context_code,
            argument=None,
        )

@StageRegistry.register(
    description="Run validation code on program output"
)
class ValidatorCodeExecutor(PythonCodeExecutor):
    """Run fixed validator code (from file path) on 1 required + 1 optional input."""

    def __init__(
        self,
        validator_path: Path,
        function_name: str = "validate",
        **kwargs,
    ):
        self.validator_path = Path(validator_path)
        if not self.validator_path.exists():
            raise ValidationError(
                f"Validator file not found: {self.validator_path}"
            )
        try:
            self.validator_code = self.validator_path.read_text(
                encoding="utf-8"
            )
        except OSError as e:
            raise ValidationError(f"Failed to read validator file: {e}") from e

        super().__init__(
            function_name=function_name,
            python_path=[self.validator_path.parent],
            **kwargs,
        )

    @classmethod
    def mandatory_inputs(cls) -> list[str]:
        return ["program_output"]

    @classmethod
    def optional_inputs(cls) -> list[str]:
        return ["context_data"]


    async def _execute_stage(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        # Get inputs by semantic name
        program_output = self.get_input("program_output")
        context_data = self.get_input_optional("context_data")
        
        # Validation function expects (context, output_of_program)
        # So context should be first, program_output should be second
        arg = program_output if context_data is None else (context_data, program_output)
        
        return await self._run_with(
            program=program,
            started_at=started_at,
            code_str=self.validator_code,
            argument=arg,
        )
