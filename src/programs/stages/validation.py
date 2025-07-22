"""Code validation stage for MetaEvolve."""

import ast
from datetime import datetime
import re
from typing import List, Optional

from loguru import logger

from src.exceptions import (
    SecurityViolationError,
    ensure_positive,
)
from src.programs.constants import DANGEROUS_PATTERNS
from src.programs.program import Program, ProgramStageResult, StageState
from src.programs.utils import build_stage_result

from .base import Stage
from .decorators import semaphore


@semaphore(limit=4)
class ValidateCodeStage(Stage):

    def __init__(
        self,
        safe_mode: bool = False,
        custom_patterns: Optional[List[str]] = None,
        max_code_length: int = 10000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.safe_mode = safe_mode
        self.custom_patterns = custom_patterns or []
        self.max_code_length = ensure_positive(
            max_code_length, "max_code_length"
        )
        self._requires_code = True

        # Compile patterns for better performance
        self._compiled_patterns = []
        for pattern in DANGEROUS_PATTERNS + self.custom_patterns:
            try:
                self._compiled_patterns.append(
                    re.compile(pattern, re.IGNORECASE)
                )
            except re.error as e:
                logger.warning(
                    f"[{self.stage_name}] Invalid regex pattern '{pattern}': {e}"
                )

    async def _execute_stage(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        code = program.code

        logger.debug(
            f"[{self.stage_name}] Program {program.id}: Starting comprehensive code validation"
        )

        # Basic validation
        if not code or not code.strip():
            return build_stage_result(
                status=StageState.FAILED,
                started_at=started_at,
                error="Code cannot be empty",
                stage_name=self.stage_name,
                context="Code validation failed - empty code",
            )

        # Length validation
        if len(code) > self.max_code_length:
            return build_stage_result(
                status=StageState.FAILED,
                started_at=started_at,
                error=f"Code too long: {len(code)} > {self.max_code_length}",
                stage_name=self.stage_name,
                context=f"Code length validation failed: {len(code)} characters",
            )

        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            error_msg = (
                f"SyntaxError on line {e.lineno}, offset {e.offset}: {e.msg}"
            )
            code_line = e.text.strip() if e.text else "<source unavailable>"

            return build_stage_result(
                status=StageState.FAILED,
                started_at=started_at,
                error=error_msg,
                stage_name=self.stage_name,
                context=f"Python syntax validation failed on line {e.lineno}: `{code_line}`",
            )

        # Security validation
        try:
            await self._validate_security(code, program.id)
        except SecurityViolationError as e:
            return build_stage_result(
                status=StageState.FAILED,
                started_at=started_at,
                error=e,
                stage_name=self.stage_name,
                context="Security validation failed",
            )

        # AST validation for additional safety
        try:
            await self._validate_ast(code, program.id)
        except SecurityViolationError as e:
            return build_stage_result(
                status=StageState.FAILED,
                started_at=started_at,
                error=e,
                stage_name=self.stage_name,
                context="AST validation failed",
            )

        return build_stage_result(
            status=StageState.COMPLETED,
            started_at=started_at,
            output={
                "message": "Code validation passed",
                "code_length": len(code),
                "security_checks_passed": True,
                "syntax_valid": True,
            },
            stage_name=self.stage_name,
        )

    async def _validate_security(self, code: str, program_id: str) -> None:
        """Comprehensive security validation."""
        validation_mode = "safe" if self.safe_mode else "trusted"

        if validation_mode == "safe" or self.enable_security_checks:
            # Check against dangerous patterns
            for pattern in self._compiled_patterns:
                match = pattern.search(code)
                if match:
                    raise SecurityViolationError(
                        f"Potentially dangerous operation detected",
                        violation_type="dangerous_pattern",
                        detected_pattern=match.group(0),
                    )

            # Additional security checks
            await self._check_import_safety(code, program_id)
            await self._check_file_operations(code, program_id)
            await self._check_network_operations(code, program_id)

    async def _check_import_safety(self, code: str, program_id: str) -> None:
        """Check for unsafe imports."""
        dangerous_imports = [
            "os",
            "sys",
            "subprocess",
            "socket",
            "urllib",
            "requests",
            "multiprocessing",
            "threading",
            "ctypes",
            "__import__",
        ]

        for imp in dangerous_imports:
            patterns = [
                rf"\bimport\s+{imp}\b",
                rf"\bfrom\s+{imp}\s+import\b",
                rf"__import__\s*\(\s*[\'\"]{imp}[\'\"]\s*\)",
            ]

            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    raise SecurityViolationError(
                        f"Unsafe import detected: {imp}",
                        violation_type="unsafe_import",
                        detected_pattern=imp,
                    )

    async def _check_file_operations(self, code: str, program_id: str) -> None:
        """Check for file system operations."""
        file_patterns = [
            r"\bopen\s*\(",
            r"\bfile\s*\(",
            r"\.read\s*\(",
            r"\.write\s*\(",
            r"\.remove\s*\(",
            r"\.unlink\s*\(",
        ]

        for pattern in file_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                raise SecurityViolationError(
                    f"File operation detected",
                    violation_type="file_operation",
                    detected_pattern=pattern,
                )

    async def _check_network_operations(
        self, code: str, program_id: str
    ) -> None:
        """Check for network operations."""
        network_patterns = [
            r"socket\.",
            r"urllib\.",
            r"requests\.",
            r"http\.",
            r"ftp\.",
        ]

        for pattern in network_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                raise SecurityViolationError(
                    f"Network operation detected",
                    violation_type="network_operation",
                    detected_pattern=pattern,
                )

    async def _validate_ast(self, code: str, program_id: str) -> None:
        """Validate code using AST analysis."""
        try:
            tree = ast.parse(code)

            # Check for dangerous AST nodes
            for node in ast.walk(tree):
                # Note: ast.Exec was removed in Python 3
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ["os", "sys", "subprocess"]:
                            raise SecurityViolationError(
                                f"Import of {alias.name} not allowed",
                                violation_type="unsafe_import",
                                detected_pattern=alias.name,
                            )

        except SyntaxError:
            # Already handled in main validation
            pass
        except Exception as e:
            logger.warning(f"[{self.stage_name}] AST validation error: {e}")
