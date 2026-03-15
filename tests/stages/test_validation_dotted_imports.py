"""Tests for Finding 1: AST import checker doesn't split dotted module names.

gigaevo/programs/stages/validation.py — _validate_ast_imports, line 155-158:

    elif isinstance(node, ast.ImportFrom):
        if node.module in self._BLOCKED_MODULES:
            raise SecurityViolationError(...)

For `from os.path import join`, the AST gives node.module = "os.path".
The blocked set contains "os" (not "os.path"), so "os.path" not in {"os", ...}
and the import slips through.

FIX NEEDED: The check should be:
    root = node.module.split(".")[0] if node.module else ""
    if root in self._BLOCKED_MODULES: ...

This file also covers `from subprocess.run import ...` and similar dotted forms
for all blocked modules.
"""

from __future__ import annotations

import pytest

from gigaevo.exceptions import SecurityViolationError
from gigaevo.programs.program import Program
from gigaevo.programs.stages.validation import ValidateCodeStage


def _stage() -> ValidateCodeStage:
    return ValidateCodeStage(timeout=30.0, safe_mode=True)


def _prog(code: str) -> Program:
    return Program(code=code)


# ---------------------------------------------------------------------------
# TestDottedFromImports — the main bug
# ---------------------------------------------------------------------------


class TestDottedFromImports:
    async def test_from_os_path_import_join_blocked(self) -> None:
        """from os.path import join — node.module='os.path' must be blocked.

        Bug: current code checks `node.module in _BLOCKED_MODULES` which is
        `'os.path' in {'os', ...}` → False. The import slips through.
        """
        stage = _stage()
        code = "from os.path import join\nresult = join('/a', 'b')"
        with pytest.raises(SecurityViolationError, match="os"):
            await stage.compute(_prog(code))

    async def test_from_os_path_import_exists_blocked(self) -> None:
        """from os.path import exists — same dotted-module bypass."""
        stage = _stage()
        code = "from os.path import exists\nflag = exists('/etc/passwd')"
        with pytest.raises(SecurityViolationError, match="os"):
            await stage.compute(_prog(code))

    async def test_from_subprocess_popen_blocked(self) -> None:
        """from subprocess import Popen — node.module='subprocess' — should already work."""
        stage = _stage()
        code = "from subprocess import Popen"
        with pytest.raises(SecurityViolationError, match="subprocess"):
            await stage.compute(_prog(code))

    async def test_from_os_environ_blocked(self) -> None:
        """from os import environ — node.module='os' — must be blocked."""
        stage = _stage()
        code = "from os import environ"
        with pytest.raises(SecurityViolationError, match="os"):
            await stage.compute(_prog(code))

    async def test_from_urllib_parse_blocked(self) -> None:
        """from urllib.parse import urlencode — node.module='urllib.parse' must be blocked."""
        stage = _stage()
        code = "from urllib.parse import urlencode\nresult = urlencode({'q': 'test'})"
        with pytest.raises(SecurityViolationError, match="urllib"):
            await stage.compute(_prog(code))

    async def test_from_urllib_request_blocked(self) -> None:
        """from urllib.request import urlopen — must be blocked."""
        stage = _stage()
        code = "from urllib.request import urlopen"
        with pytest.raises(SecurityViolationError, match="urllib"):
            await stage.compute(_prog(code))

    async def test_from_socket_blocked(self) -> None:
        """from socket import create_connection — node.module='socket' must be blocked."""
        stage = _stage()
        code = "from socket import create_connection"
        with pytest.raises(SecurityViolationError, match="socket"):
            await stage.compute(_prog(code))

    async def test_from_shutil_blocked(self) -> None:
        """from shutil import rmtree — node.module='shutil' must be blocked."""
        stage = _stage()
        code = "from shutil import rmtree"
        with pytest.raises(SecurityViolationError, match="shutil"):
            await stage.compute(_prog(code))

    async def test_from_glob_blocked(self) -> None:
        """from glob import glob — node.module='glob' must be blocked."""
        stage = _stage()
        code = "from glob import glob"
        with pytest.raises(SecurityViolationError, match="glob"):
            await stage.compute(_prog(code))

    async def test_from_pickle_blocked(self) -> None:
        """from pickle import loads — node.module='pickle' must be blocked."""
        stage = _stage()
        code = "from pickle import loads"
        with pytest.raises(SecurityViolationError, match="pickle"):
            await stage.compute(_prog(code))

    async def test_from_importlib_util_blocked(self) -> None:
        """from importlib.util import find_spec — node.module='importlib.util' must be blocked."""
        stage = _stage()
        code = "from importlib.util import find_spec\nfind_spec('os')"
        with pytest.raises(SecurityViolationError, match="importlib"):
            await stage.compute(_prog(code))

    async def test_from_sys_path_blocked(self) -> None:
        """from sys import path — node.module='sys' must be blocked."""
        stage = _stage()
        code = "from sys import path"
        with pytest.raises(SecurityViolationError, match="sys"):
            await stage.compute(_prog(code))


# ---------------------------------------------------------------------------
# Regression: non-dotted blocked imports still work after any fix
# ---------------------------------------------------------------------------


class TestNonDottedBlockedImportsStillWork:
    async def test_import_os_still_blocked(self) -> None:
        """import os — regression guard, must remain blocked."""
        stage = _stage()
        with pytest.raises(SecurityViolationError, match="os"):
            await stage.compute(_prog("import os"))

    async def test_import_subprocess_still_blocked(self) -> None:
        stage = _stage()
        with pytest.raises(SecurityViolationError):
            await stage.compute(_prog("import subprocess"))

    async def test_import_sys_still_blocked(self) -> None:
        stage = _stage()
        with pytest.raises(SecurityViolationError):
            await stage.compute(_prog("import sys"))

    async def test_from_sys_import_blocked(self) -> None:
        stage = _stage()
        with pytest.raises(SecurityViolationError, match="sys"):
            await stage.compute(_prog("from sys import argv"))


# ---------------------------------------------------------------------------
# Allowed imports remain unblocked
# ---------------------------------------------------------------------------


class TestAllowedImportsUnaffected:
    async def test_from_math_import_allowed(self) -> None:
        stage = _stage()
        result = await stage.compute(_prog("from math import sqrt\nx = sqrt(4)"))
        assert result.syntax_valid is True

    async def test_from_collections_import_allowed(self) -> None:
        stage = _stage()
        result = await stage.compute(_prog("from collections import defaultdict"))
        assert result.syntax_valid is True

    async def test_from_typing_import_allowed(self) -> None:
        stage = _stage()
        result = await stage.compute(_prog("from typing import Optional, List"))
        assert result.syntax_valid is True

    async def test_from_pathlib_import_blocked_only_by_file_ops(self) -> None:
        """pathlib itself is not in the blocked module list — only file ops are caught by AST."""
        stage = _stage()
        # Importing pathlib is OK (it's not in _BLOCKED_MODULES), but calling .unlink() is not
        with pytest.raises(SecurityViolationError, match="unlink"):
            await stage.compute(_prog("from pathlib import Path\nPath('x').unlink()"))

    async def test_import_math_dot_submodule_allowed(self) -> None:
        """Dotted imports of safe modules pass."""
        stage = _stage()
        # math is not blocked — dotted form should pass too
        result = await stage.compute(_prog("import math\nx = math.pi"))
        assert result.syntax_valid is True
