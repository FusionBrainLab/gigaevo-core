"""Extended tests for gigaevo/programs/stages/validation.py

Targets uncovered branches:
  - Invalid regex pattern in custom_patterns (warning path, not crash)
  - AST detection of .remove() and .unlink() file operations
  - AST import checking with ImportFrom for non-forbidden modules (allowed)
  - safe_mode with code that has valid syntax but uses file attribute calls
  - Interaction between text-based and AST-based security checks
"""

from __future__ import annotations

import pytest

from gigaevo.exceptions import SecurityViolationError
from gigaevo.programs.program import Program
from gigaevo.programs.stages.validation import (
    CodeValidationOutput,
    ValidateCodeStage,
)


def _prog(code: str) -> Program:
    return Program(code=code)


# ═══════════════════════════════════════════════════════════════════════════
# Invalid regex pattern handling
# ═══════════════════════════════════════════════════════════════════════════


class TestInvalidRegexPattern:
    def test_invalid_custom_pattern_does_not_crash(self) -> None:
        """An invalid regex in custom_patterns should warn, not crash init."""
        # '[invalid' is an invalid regex — unclosed character class
        stage = ValidateCodeStage(
            timeout=30.0,
            safe_mode=True,
            custom_patterns=["[invalid"],
        )
        # Stage should still be usable — the invalid pattern is skipped
        assert stage is not None

    async def test_invalid_pattern_still_allows_valid_code(self) -> None:
        """Even with a bad regex, valid code passes."""
        stage = ValidateCodeStage(
            timeout=30.0,
            safe_mode=True,
            custom_patterns=["[bad_regex"],
        )
        result = await stage.compute(_prog("x = 42"))
        assert result.syntax_valid is True

    async def test_mixed_valid_and_invalid_patterns(self) -> None:
        """Valid patterns still work when mixed with invalid ones."""
        stage = ValidateCodeStage(
            timeout=30.0,
            safe_mode=True,
            custom_patterns=["[bad", r"forbidden_word"],
        )
        with pytest.raises(SecurityViolationError):
            await stage.compute(_prog("x = forbidden_word()"))


# ═══════════════════════════════════════════════════════════════════════════
# AST file operation detection (.remove, .unlink)
# ═══════════════════════════════════════════════════════════════════════════


class TestASTFileOps:
    async def test_remove_detected(self) -> None:
        """obj.remove() should be caught by AST scan."""
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        with pytest.raises(SecurityViolationError, match="remove"):
            await stage.compute(
                _prog("import pathlib\np = pathlib.Path('x')\np.remove()")
            )

    async def test_unlink_detected(self) -> None:
        """obj.unlink() should be caught by AST scan."""
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        with pytest.raises(SecurityViolationError, match="unlink"):
            await stage.compute(
                _prog("import pathlib\np = pathlib.Path('x')\np.unlink()")
            )

    async def test_read_detected(self) -> None:
        """obj.read() should be caught."""
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        with pytest.raises(SecurityViolationError, match="read"):
            await stage.compute(_prog("x = 1\nresult.read()"))

    async def test_write_detected(self) -> None:
        """obj.write('data') caught."""
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        with pytest.raises(SecurityViolationError, match="write"):
            await stage.compute(_prog("x = 1\nf.write('data')"))

    async def test_allowed_attribute_not_blocked(self) -> None:
        """Attribute calls not in the blocklist should pass."""
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        code = "x = [1, 2, 3]\nx.append(4)\ny = x.copy()"
        result = await stage.compute(_prog(code))
        assert result.syntax_valid is True

    async def test_open_as_function_call(self) -> None:
        """open() as a Name call (not Attribute) is caught."""
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        with pytest.raises(SecurityViolationError):
            await stage.compute(_prog("f = open('test.txt', 'w')"))

    async def test_file_as_function_call(self) -> None:
        """file() call is caught (legacy Python 2 style)."""
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        with pytest.raises(SecurityViolationError):
            await stage.compute(_prog("f = file('test.txt')"))


# ═══════════════════════════════════════════════════════════════════════════
# AST import checking edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestASTImportEdgeCases:
    async def test_import_math_allowed(self) -> None:
        """Non-forbidden imports should pass in safe_mode."""
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        result = await stage.compute(_prog("import math\nx = math.pi"))
        assert result.syntax_valid is True

    async def test_from_math_import_allowed(self) -> None:
        """from math import sqrt should pass."""
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        result = await stage.compute(_prog("from math import sqrt\nx = sqrt(4)"))
        assert result.syntax_valid is True

    async def test_import_os_blocked(self) -> None:
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        with pytest.raises(SecurityViolationError, match="os"):
            await stage.compute(_prog("import os"))

    async def test_from_sys_import_blocked(self) -> None:
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        with pytest.raises(SecurityViolationError, match="sys"):
            await stage.compute(_prog("from sys import argv"))

    async def test_from_subprocess_import_blocked(self) -> None:
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        with pytest.raises(SecurityViolationError, match="subprocess"):
            await stage.compute(_prog("from subprocess import run"))

    async def test_import_collections_allowed(self) -> None:
        """Non-forbidden standard library modules should pass."""
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        result = await stage.compute(_prog("from collections import defaultdict"))
        assert result.syntax_valid is True


# ═══════════════════════════════════════════════════════════════════════════
# Text-based vs AST-based interaction
# ═══════════════════════════════════════════════════════════════════════════


class TestSecurityCheckInteraction:
    async def test_text_check_fires_before_ast(self) -> None:
        """Text-based check catches 'eval(' before AST check runs."""
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        # eval() is caught by text regex first
        with pytest.raises(SecurityViolationError, match="eval"):
            await stage.compute(_prog("x = eval('1+1')"))

    async def test_ast_catches_what_text_misses(self) -> None:
        """A .remove() call is not in DANGEROUS_PATTERNS text list,
        but IS caught by AST scan."""
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        # 'os.path.remove()' — the word 'remove' is not in text patterns
        # but .remove() is caught by AST attribute scan
        with pytest.raises(SecurityViolationError):
            await stage.compute(_prog("x = 1\npath.remove()"))


# ═══════════════════════════════════════════════════════════════════════════
# Output model structure
# ═══════════════════════════════════════════════════════════════════════════


class TestCodeValidationOutput:
    def test_output_model_fields(self) -> None:
        out = CodeValidationOutput(
            message="ok",
            code_length=10,
            security_checks_passed=True,
            syntax_valid=True,
        )
        assert out.message == "ok"
        assert out.code_length == 10
        assert out.security_checks_passed is True
        assert out.syntax_valid is True

    async def test_safe_mode_off_security_checks_false(self) -> None:
        """When safe_mode=False, security_checks_passed should be False."""
        stage = ValidateCodeStage(timeout=30.0, safe_mode=False)
        result = await stage.compute(_prog("x = 42"))
        assert result.security_checks_passed is False

    async def test_code_length_accurate(self) -> None:
        code = "x = 42\ny = 100"
        stage = ValidateCodeStage(timeout=30.0)
        result = await stage.compute(_prog(code))
        assert result.code_length == len(code)


# ═══════════════════════════════════════════════════════════════════════════
# SyntaxError detail preservation
# ═══════════════════════════════════════════════════════════════════════════


class TestSyntaxErrorDetails:
    async def test_syntax_error_includes_line_info(self) -> None:
        """SyntaxError message should include line number and offset."""
        stage = ValidateCodeStage(timeout=30.0)
        with pytest.raises(SyntaxError, match="line"):
            await stage.compute(_prog("def f(\n    x = "))

    async def test_multiline_syntax_error(self) -> None:
        stage = ValidateCodeStage(timeout=30.0)
        code = "x = 1\ny = 2\nif True\n    pass"
        with pytest.raises(SyntaxError):
            await stage.compute(_prog(code))
