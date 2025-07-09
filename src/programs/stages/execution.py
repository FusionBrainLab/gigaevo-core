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
    ensure_not_none,
    ensure_positive,
)
from src.programs.program import Program, ProgramStageResult, StageState
from src.programs.utils import (
    build_stage_result,
    construct_exec_code,
    dedent_code,
    run_python_snippet,
)

from .base import Stage
from .decorators import semaphore, retry

# Threshold for using temporary file vs embedding in code (in bytes)
# This is approximately 8KB of base64 data (6KB of pickle data)
INPUT_SIZE_THRESHOLD = 8 * 1024


@semaphore(limit=6)
class RunPythonCode(Stage):
    """Bulletproof Python code execution stage with sandboxing and monitoring."""

    def __init__(
        self,
        code: str,
        function_name: str = "run_code",
        python_path: Optional[List[Path]] = None,
        input_obj: Optional[Any] = None,
        enable_sandboxing: bool = False,
        max_output_size: int = 1024 * 1024 * 64,  # 64MB
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.code = ensure_not_none(code, "code")
        self.function_name = ensure_not_none(function_name, "function_name")
        self.python_path = python_path or []
        self.input_obj = input_obj
        self.enable_sandboxing = enable_sandboxing
        self.max_output_size = ensure_positive(
            max_output_size, "max_output_size"
        )
        self._requires_code = False  # This stage provides its own code

    async def _execute_stage(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        logger.debug(
            f"[{self.stage_name}] Program {program.id}: Executing Python code"
        )

        try:
            # Prepare execution environment
            user_code = dedent_code(self.code)
            input_b64 = None
            input_file_path = None
            temp_file = None

            if self.input_obj is not None:
                try:
                    # Serialize input object
                    input_pickle = pickle.dumps(self.input_obj)
                    input_b64_bytes = base64.b64encode(input_pickle)
                    
                    # If input is large, use temporary file instead of embedding in code
                    if len(input_b64_bytes) > INPUT_SIZE_THRESHOLD:
                        temp_file = tempfile.NamedTemporaryFile(mode='wb', delete=False)
                        temp_file.write(input_pickle)
                        temp_file.close()
                        input_file_path = temp_file.name
                        input_b64 = None  # Don't embed in code
                        
                        logger.debug(
                            f"[{self.stage_name}] Using temporary file for large input: {len(input_b64_bytes)} bytes"
                        )
                    else:
                        input_b64 = input_b64_bytes.decode("utf-8")
                    
                except Exception as e:
                    raise ProgramExecutionError(
                        f"Failed to serialize input object: {e}",
                        program_id=program.id,
                        cause=e,
                    )

            try:
                # Construct execution code with safety measures
                exec_code = construct_exec_code(
                    user_code=user_code,
                    function_name=self.function_name,
                    input_b64=input_b64,
                    input_file_path=input_file_path,
                    python_path=self.python_path,
                )

                # Execute with comprehensive monitoring
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
                if result.output is not None and (size := len(pickle.dumps(result.output))) > self.max_output_size:
                    return build_stage_result(
                        status=StageState.FAILED,
                        started_at=started_at,
                        error=f"Output too large: {size} > {self.max_output_size}",
                        stage_name=self.stage_name,
                        context=f"Output size limit exceeded: {size} bytes",
                    )

                return result

            finally:
                # Clean up temporary file if used
                if temp_file and input_file_path:
                    try:
                        Path(input_file_path).unlink()
                    except Exception as e:
                        logger.warning(f"Failed to clean up temporary file {input_file_path}: {e}")

        except Exception as e:
            return build_stage_result(
                status=StageState.FAILED,
                started_at=started_at,
                error=e,
                stage_name=self.stage_name,
                context="Code execution failed with unexpected error",
            )

    async def _run_sandboxed(
        self, code: str, started_at: datetime, program_id: str
    ) -> ProgramStageResult:
        """Run code in a sandboxed environment."""
        # Create temporary directory for sandboxing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                # Set resource limits
                await self._set_resource_limits()

                # Execute with timeout and monitoring
                result = await run_python_snippet(
                    code, started_at, timeout=self.timeout, cwd=temp_path
                )

                return result

            except Exception as e:
                raise ProgramExecutionError(
                    f"Sandboxed execution failed: {e}",
                    program_id=program_id,
                    cause=e,
                )

    async def _set_resource_limits(self) -> None:
        """Set resource limits for sandboxed execution."""
        try:
            # Set memory limit
            memory_limit = self.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

            # Set CPU time limit
            cpu_limit = int(self.timeout * 2)  # Allow some overhead
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))

            # Set file size limit
            file_limit = 10 * 1024 * 1024  # 10MB
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))

        except Exception as e:
            logger.warning(
                f"[{self.stage_name}] Failed to set resource limits: {e}"
            )


class RunCodeStage(RunPythonCode):
    """Stage that runs the program's own code."""

    def __init__(self, function_name: str = "run_code", context_stage: Optional[str] = None, **kwargs):
        # We'll set the code in run() method
        super().__init__(code="", function_name=function_name, **kwargs)
        self.context_stage = context_stage
        self._requires_code = True

    async def _execute_stage(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        # Set the code from the program
        self.code = program.code

        if self.context_stage:
            context_result: Optional[ProgramStageResult] = program.stage_results.get(
                self.context_stage
            )
            if not context_result or not context_result.is_completed():
                raise StageError(
                    f"Context stage '{self.context_stage}' did not complete successfully",
                    stage_name=self.stage_name,
                    stage_type="execution",
                )
            self.input_obj = context_result.output

        # Call parent run method
        return await super()._execute_stage(program, started_at)


@retry(times=2, backoff=0.1)
class RunValidationStage(RunPythonCode):
    """Stage that runs validation code against previous stage output."""

    def __init__(
        self,
        validator_path: Path,
        data_to_validate_stage: str,
        context_stage: Optional[str] = None,
        function_name: str = "validate",
        **kwargs,
    ):
        self.validator_path = Path(validator_path)
        self.data_to_validate_stage = ensure_not_none(
            data_to_validate_stage, "data_to_validate_stage"
        )
        self.context_stage = context_stage

        # Validate validator file exists
        if not self.validator_path.exists():
            raise ValidationError(
                f"Validator file not found: {self.validator_path}"
            )

        # Read validator code
        try:
            validator_code = self.validator_path.read_text()
        except Exception as e:
            raise ValidationError(f"Failed to read validator file: {e}")

        super().__init__(
            code=validator_code,
            function_name=function_name,
            python_path=[self.validator_path.parent],
            **kwargs,
        )

    async def _execute_stage(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        logger.debug(
            f"[{self.stage_name}] Program {program.id}: Running validation against {self.data_to_validate_stage}"
        )

        # Check previous stage result
        prev_result: Optional[ProgramStageResult] = program.stage_results.get(
            self.data_to_validate_stage
        )
        if not prev_result or not prev_result.is_completed():
            raise StageError(
                f"Previous stage '{self.data_to_validate_stage}' did not complete successfully",
                stage_name=self.stage_name,
                stage_type="validation",
            )
        
        self.input_obj = prev_result.output
        
        if self.context_stage:
            context_result: Optional[ProgramStageResult] = program.stage_results.get(
                self.context_stage
            )
            if not context_result or not context_result.is_completed():
                raise StageError(
                    f"Context stage '{self.context_stage}' did not complete successfully",
                    stage_name=self.stage_name,
                    stage_type="validation",
                )
            self.input_obj = (context_result.output, prev_result.output)

        # Run validation
        return await super()._execute_stage(program, started_at)
