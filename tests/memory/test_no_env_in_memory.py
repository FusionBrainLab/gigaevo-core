"""Guard: the memory subsystem takes all configuration through Hydra.

``os.getenv`` / ``os.environ`` / ``dotenv`` must not appear under
``gigaevo/memory/`` or ``gigaevo/memory_platform/`` outside the explicit
allowlist (CLI entrypoint defaults and HF/Langfuse runtime fallbacks)."""

from __future__ import annotations

import ast
from pathlib import Path

GIGAEVO_ROOT = Path(__file__).parent.parent.parent / "gigaevo"

ENV_ALLOWLIST = {
    "memory/ideas_tracker/cli.py",
    "memory/ideas_tracker/ideas_tracker.py",
    "memory/shared_memory/agentic_runtime.py",
    "memory/shared_memory/amem_gam_retriever.py",
}


def _iter_memory_files():
    for pkg in ("memory", "memory_platform"):
        for path in sorted((GIGAEVO_ROOT / pkg).rglob("*.py")):
            if "_vendor" not in path.parts:
                yield path


def _env_reads(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in {"getenv", "environ"}
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
        ):
            lines.append(node.lineno)
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "os"
            and any(alias.name in {"getenv", "environ"} for alias in node.names)
        ):
            lines.append(node.lineno)
    return lines


def _dotenv_imports(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "dotenv"
        ):
            lines.append(node.lineno)
        if isinstance(node, ast.Import) and any(
            alias.name.startswith("dotenv") for alias in node.names
        ):
            lines.append(node.lineno)
    return lines


def test_no_env_access_outside_allowlist():
    offenders = []
    for path in _iter_memory_files():
        rel = str(path.relative_to(GIGAEVO_ROOT))
        if rel in ENV_ALLOWLIST:
            continue
        for lineno in _env_reads(ast.parse(path.read_text(encoding="utf-8"))):
            offenders.append(f"{rel}:{lineno}")
    assert not offenders, f"env access outside allowlist: {offenders}"


def test_no_dotenv_anywhere():
    offenders = []
    for path in _iter_memory_files():
        rel = str(path.relative_to(GIGAEVO_ROOT))
        for lineno in _dotenv_imports(ast.parse(path.read_text(encoding="utf-8"))):
            offenders.append(f"{rel}:{lineno}")
    assert not offenders, f"dotenv usage in memory subsystem: {offenders}"
