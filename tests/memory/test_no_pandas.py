"""The memory subsystem must stay pandas-free (task #93 purity contract)."""

from __future__ import annotations

import ast
from pathlib import Path

import gigaevo.memory

MEMORY_ROOT = Path(gigaevo.memory.__file__).parent


def _pandas_imports(tree: ast.AST) -> list[str]:
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits += [a.name for a in node.names if a.name.split(".")[0] == "pandas"]
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] == "pandas":
                hits.append(node.module)
    return hits


def test_no_pandas_imports_under_memory():
    offenders = {}
    for path in MEMORY_ROOT.rglob("*.py"):
        if "_vendor" in path.parts:
            continue
        hits = _pandas_imports(ast.parse(path.read_text(encoding="utf-8")))
        if hits:
            offenders[str(path.relative_to(MEMORY_ROOT))] = hits
    assert not offenders, f"pandas imports found in gigaevo/memory: {offenders}"
