from __future__ import annotations

import ast
import re

from loguru import logger

from gigaevo.exceptions import SecurityViolationError
from gigaevo.programs.constants import DANGEROUS_PATTERNS
from gigaevo.programs.core_types import StageIO, VoidInput
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.stage_registry import StageRegistry


class CodeValidationOutput(StageIO):
    message: str
    code_length: int
    security_checks_passed: bool
    syntax_valid: bool


@StageRegistry.register(description="Validate program code for syntax and security")
class ValidateCodeStage(Stage):
    """
    Validates Program.code:
      - Non-empty & length bound
      - Python syntax (compile)
      - Optional "safe mode" checks (regex + simple AST heuristics)
    """

    InputsModel = VoidInput
    OutputModel = CodeValidationOutput

    def __init__(
        self,
        *,
        safe_mode: bool = False,
        custom_patterns: list[str] | None = None,
        max_code_length: int = 10_000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if max_code_length <= 0:
            raise ValueError("max_code_length must be positive")
        self.safe_mode = safe_mode
        self.custom_patterns = custom_patterns or []
        self.max_code_length = max_code_length

        self._compiled_patterns: list[re.Pattern[str]] = []
        for pattern in [*DANGEROUS_PATTERNS, *self.custom_patterns]:
            try:
                self._compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error as e:
                logger.warning(
                    "[{}] Invalid regex pattern '{}': {}",
                    type(self).__name__,
                    pattern,
                    e,
                )

    async def compute(self, program: Program) -> StageIO:
        code = program.code or ""
        clsname = type(self).__name__

        if not code.strip():
            raise ValueError("Code cannot be empty")

        if len(code) > self.max_code_length:
            raise ValueError(f"Code too long: {len(code)} > {self.max_code_length}")

        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            code_line = (e.text or "").strip() or "<source unavailable>"
            raise SyntaxError(
                f"SyntaxError at line {e.lineno}, offset {e.offset}: {e.msg}. Line: `{code_line}`"
            ) from e

        if self.safe_mode:
            self._validate_security_text(code)
            self._validate_security_ast_file_ops(code)
            self._validate_ast_imports(code)

        logger.debug("[{}] Code validation passed (len={})", clsname, len(code))
        return CodeValidationOutput(
            message="Code validation passed",
            code_length=len(code),
            security_checks_passed=bool(self.safe_mode),
            syntax_valid=True,
        )

    def _validate_security_text(self, code: str) -> None:
        """Regex-based screening for dangerous textual patterns."""
        for pattern in self._compiled_patterns:
            m = pattern.search(code)
            if m:
                raise SecurityViolationError(
                    f"Potentially dangerous pattern detected: {m.group(0)!r}"
                )

    def _validate_security_ast_file_ops(self, code: str) -> None:
        """AST scan for obvious file I/O calls (open/file/read/write/remove/unlink)."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in {"open", "file"}:
                    raise SecurityViolationError(
                        "File operation detected via AST (open/file)"
                    )
                if isinstance(node.func, ast.Attribute) and node.func.attr in {
                    "read",
                    "write",
                    "remove",
                    "unlink",
                }:
                    raise SecurityViolationError(
                        f"File operation detected via AST ({node.func.attr})"
                    )

    def _validate_ast_imports(self, code: str) -> None:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in {"os", "sys", "subprocess"}:
                        raise SecurityViolationError(
                            f"Import of '{alias.name}' not allowed in safe_mode"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module in {"os", "sys", "subprocess"}:
                    raise SecurityViolationError(
                        f"Import from '{node.module}' not allowed in safe_mode"
                    )
