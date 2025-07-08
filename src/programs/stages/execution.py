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

            if self.input_obj is not None:
                try:
                    input_b64 = base64.b64encode(
                        pickle.dumps(self.input_obj)
                    ).decode("utf-8")
                except Exception as e:
                    raise ProgramExecutionError(
                        f"Failed to serialize input object: {e}",
                        program_id=program.id,
                        cause=e,
                    )

            # Construct execution code with safety measures
            exec_code = construct_exec_code(
                user_code=user_code,
                function_name=self.function_name,
                input_b64=input_b64,
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
            if result.output and len(str(result.output)) > self.max_output_size:
                return build_stage_result(
                    status=StageState.FAILED,
                    started_at=started_at,
                    error=f"Output too large: {len(str(result.output))} > {self.max_output_size}",
                    stage_name=self.stage_name,
                    context=f"Output size limit exceeded: {len(str(result.output))} bytes",
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

    def __init__(self, function_name: str = "run_code", **kwargs):
        # We'll set the code in run() method
        super().__init__(code="", function_name=function_name, **kwargs)
        self._requires_code = True

    async def _execute_stage(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        # Set the code from the program
        self.code = program.code

        # Call parent run method
        return await super()._execute_stage(program, started_at)


@retry(times=2, backoff=0.1)
class RunValidationStage(RunPythonCode):
    """Stage that runs validation code against previous stage output."""

    def __init__(
        self,
        validator_path: Path,
        prev_stage_name: str,
        function_name: str = "validate",
        **kwargs,
    ):
        self.validator_path = Path(validator_path)
        self.prev_stage_name = ensure_not_none(
            prev_stage_name, "prev_stage_name"
        )

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
            f"[{self.stage_name}] Program {program.id}: Running validation against {self.prev_stage_name}"
        )

        # Check previous stage result
        prev_result: Optional[ProgramStageResult] = program.stage_results.get(
            self.prev_stage_name
        )
        if not prev_result or not prev_result.is_completed():
            raise StageError(
                f"Previous stage '{self.prev_stage_name}' did not complete successfully",
                stage_name=self.stage_name,
                stage_type="validation",
            )

        # Set input object from previous stage
        self.input_obj = prev_result.output

        # Run validation
        return await super()._execute_stage(program, started_at)
