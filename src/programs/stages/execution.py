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

from .base import Stage
from .decorators import retry

INPUT_SIZE_THRESHOLD = 8 * 1024


class RunPythonCode(Stage):
    def __init__(
        self,
        function_name: str = "run_code",
        python_path: Optional[List[Path]] = None,
        input_obj: Optional[Any] = None,
        code: Optional[str] = None,
        enable_sandboxing: bool = False,
        max_output_size: int = 1024 * 1024 * 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.function_name = function_name
        self.python_path = python_path or []
        self.input_obj = input_obj
        self.enable_sandboxing = enable_sandboxing
        self.max_output_size = max_output_size
        self.code = code

    async def _execute_stage(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        logger.debug(
            f"[{self.stage_name}] Program {program.id}: Executing Python code"
        )

        # Prepare input
        input_b64, input_file_path = None, None
        temp_file_path = None

        if self.input_obj is not None:
            try:
                input_pickle = pickle.dumps(self.input_obj)
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
                )

        try:
            # Execute code
            input_code = self.code if self.code is not None else self._get_code(program)
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

        except Exception as e:
            return build_stage_result(
                status=StageState.FAILED,
                started_at=started_at,
                error=e,
                stage_name=self.stage_name,
                context="Code execution failed with unexpected error",
            )
        finally:
            if temp_file_path:
                try:
                    Path(temp_file_path).unlink()
                except Exception as e:
                    logger.warning(
                        f"Failed to clean up temporary file {temp_file_path}: {e}"
                    )

    def _get_code(self, program: Program) -> str:
        """Get the code to execute. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _get_code()")

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
                )

    async def _set_resource_limits(self) -> None:
        try:
            memory_limit = self.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (int(self.timeout * 2), int(self.timeout * 2)),
            )
            resource.setrlimit(
                resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024)
            )
        except Exception as e:
            logger.warning(
                f"[{self.stage_name}] Failed to set resource limits: {e}"
            )


class RunCodeStage(RunPythonCode):
    def __init__(
        self,
        function_name: str = "run_code",
        context_stage: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(function_name=function_name, **kwargs)
        self.context_stage = context_stage

    def _get_code(self, program: Program) -> str:
        return program.code

    async def _execute_stage(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        if self.context_stage:
            context_result = program.stage_results.get(self.context_stage)
            if not context_result or not context_result.is_completed():
                raise StageError(
                    f"Context stage '{self.context_stage}' did not complete successfully",
                    stage_name=self.stage_name,
                    stage_type="execution",
                )
            self.input_obj = context_result.output

        return await super()._execute_stage(program, started_at)


@retry(times=2, backoff=0.1)
class RunValidationStage(RunPythonCode):
    def __init__(
        self,
        validator_path: Path,
        data_to_validate_stage: str,
        context_stage: Optional[str] = None,
        function_name: str = "validate",
        **kwargs,
    ):
        self.validator_path = Path(validator_path)
        self.data_to_validate_stage = data_to_validate_stage
        self.context_stage = context_stage

        if not self.validator_path.exists():
            raise ValidationError(
                f"Validator file not found: {self.validator_path}"
            )

        try:
            self.validator_code = self.validator_path.read_text()
        except Exception as e:
            raise ValidationError(f"Failed to read validator file: {e}")

        super().__init__(
            function_name=function_name,
            python_path=[self.validator_path.parent],
            **kwargs,
        )

    def _get_code(self, program: Program) -> str:
        return self.validator_code

    async def _execute_stage(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        logger.debug(
            f"[{self.stage_name}] Program {program.id}: Running validation against {self.data_to_validate_stage}"
        )

        prev_result = program.stage_results.get(self.data_to_validate_stage)
        if not prev_result or not prev_result.is_completed():
            raise StageError(
                f"Previous stage '{self.data_to_validate_stage}' did not complete successfully",
                stage_name=self.stage_name,
                stage_type="validation",
            )

        self.input_obj = prev_result.output

        if self.context_stage:
            context_result = program.stage_results.get(self.context_stage)
            if not context_result or not context_result.is_completed():
                raise StageError(
                    f"Context stage '{self.context_stage}' did not complete successfully",
                    stage_name=self.stage_name,
                    stage_type="validation",
                )
            self.input_obj = (context_result.output, prev_result.output)

        return await super()._execute_stage(program, started_at)
