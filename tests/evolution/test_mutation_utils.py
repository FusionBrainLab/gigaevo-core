"""Tests for gigaevo/evolution/mutation/utils.py — _DocstringRemover."""

from __future__ import annotations

import ast
import textwrap

from gigaevo.evolution.mutation.utils import _DocstringRemover


def _strip(code: str) -> str:
    """Remove docstrings from code and return the result."""
    tree = ast.parse(textwrap.dedent(code))
    tree = _DocstringRemover().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


class TestDocstringRemover:
    def test_removes_function_docstring(self) -> None:
        code = '''
def foo():
    """This is a docstring."""
    return 1
'''
        result = _strip(code)
        assert "This is a docstring" not in result
        assert "return 1" in result

    def test_removes_async_function_docstring(self) -> None:
        code = '''
async def bar():
    """Async docstring."""
    pass
'''
        result = _strip(code)
        assert "Async docstring" not in result

    def test_removes_class_docstring(self) -> None:
        code = '''
class MyClass:
    """Class docstring."""
    x = 1
'''
        result = _strip(code)
        assert "Class docstring" not in result
        assert "x = 1" in result

    def test_removes_module_docstring(self) -> None:
        code = '''"""Module docstring."""
x = 1
'''
        result = _strip(code)
        assert "Module docstring" not in result
        assert "x = 1" in result

    def test_removes_nested_docstrings(self) -> None:
        code = '''
class Outer:
    """Outer doc."""
    def method(self):
        """Method doc."""
        return 42
'''
        result = _strip(code)
        assert "Outer doc" not in result
        assert "Method doc" not in result
        assert "42" in result

    def test_preserves_non_docstring_strings(self) -> None:
        code = """
def foo():
    x = "not a docstring"
    return x
"""
        result = _strip(code)
        assert "not a docstring" in result

    def test_no_docstrings_unchanged(self) -> None:
        code = """
def foo():
    return 1
"""
        result = _strip(code)
        assert "return 1" in result

    def test_preserves_string_after_statement(self) -> None:
        code = """
def foo():
    x = 1
    "this is not a docstring"
    return x
"""
        result = _strip(code)
        # The string after a statement is not a docstring
        assert "this is not a docstring" in result
