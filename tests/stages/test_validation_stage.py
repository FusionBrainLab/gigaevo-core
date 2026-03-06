"""Tests for gigaevo/programs/stages/validation.py"""

import pytest

from gigaevo.exceptions import SecurityViolationError
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.validation import CodeValidationOutput, ValidateCodeStage


def _prog(code="def solve(): return 42"):
    return Program(code=code, state=ProgramState.RUNNING)


class TestValidateCodeStageSyntax:
    async def test_valid_code(self):
        stage = ValidateCodeStage(timeout=30.0)
        result = await stage.compute(_prog("x = 1 + 2"))
        assert isinstance(result, CodeValidationOutput)
        assert result.syntax_valid is True
        assert result.code_length == len("x = 1 + 2")

    async def test_empty_code(self):
        stage = ValidateCodeStage(timeout=30.0)
        # Program model enforces min_length=1 on code, so bypass validation with model_construct
        p = Program.model_construct(code="", state=ProgramState.RUNNING)
        with pytest.raises(ValueError, match="empty"):
            await stage.compute(p)

    async def test_whitespace_only(self):
        stage = ValidateCodeStage(timeout=30.0)
        # Program strips whitespace, so bypass validation with model_construct
        p = Program.model_construct(code="   \n\t  ", state=ProgramState.RUNNING)
        with pytest.raises(ValueError, match="empty"):
            await stage.compute(p)

    async def test_syntax_error(self):
        stage = ValidateCodeStage(timeout=30.0)
        with pytest.raises(SyntaxError, match="SyntaxError"):
            await stage.compute(_prog("def foo("))

    async def test_too_long(self):
        stage = ValidateCodeStage(timeout=30.0, max_code_length=10)
        with pytest.raises(ValueError, match="too long"):
            await stage.compute(_prog("x = 1 + 2 + 3 + 4 + 5"))

    async def test_custom_max_length(self):
        stage = ValidateCodeStage(timeout=30.0, max_code_length=5)
        with pytest.raises(ValueError, match="too long"):
            await stage.compute(_prog("x = 123"))

    def test_max_code_length_must_be_positive(self):
        with pytest.raises(ValueError, match="max_code_length must be positive"):
            ValidateCodeStage(timeout=30.0, max_code_length=0)

    async def test_code_at_exact_max_length(self):
        """Code with len == max_code_length should pass."""
        code = "x = 1 + 2"  # 9 chars, no trailing whitespace (Program strips it)
        stage = ValidateCodeStage(timeout=30.0, max_code_length=len(code))
        result = await stage.compute(_prog(code))
        assert result.code_length == len(code)

    async def test_code_one_over_max_length(self):
        """Code with len == max_code_length + 1 should fail."""
        code = "x = 1 + 2"
        stage = ValidateCodeStage(timeout=30.0, max_code_length=len(code) - 1)
        with pytest.raises(ValueError, match="too long"):
            await stage.compute(_prog(code))


class TestValidateCodeStageSecurity:
    async def test_safe_mode_blocks_os(self):
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        with pytest.raises(SecurityViolationError):
            await stage.compute(_prog("import os\nos.listdir('.')"))

    async def test_safe_mode_blocks_sys(self):
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        with pytest.raises(SecurityViolationError):
            await stage.compute(_prog("import sys"))

    async def test_safe_mode_blocks_subprocess(self):
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        with pytest.raises(SecurityViolationError):
            await stage.compute(_prog("import subprocess"))

    async def test_safe_mode_blocks_eval(self):
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        with pytest.raises(SecurityViolationError):
            await stage.compute(_prog("result = eval('1+1')"))

    async def test_safe_mode_blocks_exec(self):
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        with pytest.raises(SecurityViolationError):
            await stage.compute(_prog("exec('x=1')"))

    async def test_safe_mode_blocks_open(self):
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        with pytest.raises(SecurityViolationError):
            await stage.compute(_prog("f = open('test.txt')"))

    async def test_safe_mode_allows_safe_code(self):
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        result = await stage.compute(_prog("import math\nx = math.sqrt(4)"))
        assert result.syntax_valid is True
        assert result.security_checks_passed is True

    async def test_custom_pattern(self):
        stage = ValidateCodeStage(
            timeout=30.0, safe_mode=True, custom_patterns=[r"forbidden_func"]
        )
        with pytest.raises(SecurityViolationError):
            await stage.compute(_prog("forbidden_func()"))

    async def test_ast_file_ops_write(self):
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        # AST check catches .write() attribute calls
        code = "x = 1\nresult.write('data')"
        with pytest.raises(SecurityViolationError, match="File operation"):
            await stage.compute(_prog(code))

    async def test_ast_imports_from(self):
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        with pytest.raises(SecurityViolationError):
            await stage.compute(_prog("from os import path"))

    async def test_safe_mode_blocks_from_import_aliased(self):
        """'from os import path as p' should be blocked."""
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        with pytest.raises(SecurityViolationError):
            await stage.compute(_prog("from os import path as p"))

    async def test_safe_mode_blocks_nested_import(self):
        """'import os.path' should be blocked."""
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        with pytest.raises(SecurityViolationError):
            await stage.compute(_prog("import os.path"))

    async def test_safe_mode_allows_math_operations(self):
        """Complex math code with numpy-style operations passes."""
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        code = "import math\nresult = [math.sqrt(i**2 + 1) for i in range(100)]"
        result = await stage.compute(_prog(code))
        assert result.syntax_valid is True
        assert result.security_checks_passed is True

    async def test_no_safe_mode_allows_dangerous(self):
        stage = ValidateCodeStage(timeout=30.0, safe_mode=False)
        result = await stage.compute(_prog("import os\nos.listdir('.')"))
        assert result.syntax_valid is True
        assert result.security_checks_passed is False


class TestValidateCodeStageOutput:
    async def test_output_fields(self):
        stage = ValidateCodeStage(timeout=30.0, safe_mode=True)
        code = "x = 42"
        result = await stage.compute(_prog(code))
        assert result.message == "Code validation passed"
        assert result.code_length == len(code)
        assert result.security_checks_passed is True
        assert result.syntax_valid is True
