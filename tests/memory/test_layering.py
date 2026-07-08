"""AST-enforced layering of ``gigaevo.memory``.

Layer order: ``cards < events < storage < {read | write} < provider``.
``read/`` and ``write/`` never import each other (eviction's ``CardScorer`` /
``CardValueScorer`` Protocols live in ``write/``; ``read/reputation`` implements
them; config wires them). Chroma is confined to ``storage/index.py``; LLM handles (routers,
agents, langgraph/langchain) are confined to the research agent and the
write-side authoring modules.
"""

from __future__ import annotations

import ast
from pathlib import Path

import gigaevo.memory

MEMORY_ROOT = Path(gigaevo.memory.__file__).parent
PACKAGE = "gigaevo.memory"

LAYER_ALLOWED = {
    "cards": frozenset(),
    "events": frozenset({"cards"}),
    "storage": frozenset({"cards", "events", "storage"}),
    "read": frozenset({"cards", "events", "storage", "read"}),
    "write": frozenset({"cards", "events", "storage", "write"}),
    "provider": frozenset({"cards", "events", "storage", "read", "write", "provider"}),
    "live_memory_hook": frozenset(
        {"cards", "events", "storage", "read", "write", "provider"}
    ),
}

CHROMA_ALLOWED = {"storage/index.py"}
# The embedding stack (Chroma + the sentence-transformers model it loads, plus
# the transformer/runtime libs underneath) is confined to the one index module.
EMBEDDING_PREFIXES = (
    "chromadb",
    "sentence_transformers",
    "transformers",
    "fastembed",
    "onnxruntime",
)
LLM_ALLOWED = {
    "storage/research.py",
    "write/librarian.py",
    "write/consolidation.py",
    "write/writer.py",
}
LLM_PREFIXES = ("gigaevo.llm", "langgraph", "langchain", "langchain_core")


def _memory_modules() -> list[Path]:
    return sorted(p for p in MEMORY_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _module_layer(rel: Path) -> str:
    if len(rel.parts) > 1:
        return rel.parts[0]
    # The package-root __init__ re-exports the integration surface — treat it as
    # the top (provider) layer so it may import from any layer below.
    return "provider" if rel.stem == "__init__" else rel.stem


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    rel = path.relative_to(MEMORY_ROOT)
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = f"{PACKAGE}." + ".".join(
                    rel.parts[: len(rel.parts) - node.level]
                )
                module = f"{base}.{node.module}" if node.module else base
                modules.append(module.rstrip("."))
            elif node.module:
                modules.append(node.module)
    return modules


def _memory_target(module: str) -> str | None:
    """Layer name a ``gigaevo.memory``-internal import resolves to, else None."""
    if module == PACKAGE:
        return "provider"  # package __init__ re-exports the integration surface
    if not module.startswith(PACKAGE + "."):
        return None
    return module.removeprefix(PACKAGE + ".").split(".")[0]


def test_layer_imports_respect_order() -> None:
    violations: list[str] = []
    for path in _memory_modules():
        rel = path.relative_to(MEMORY_ROOT)
        allowed = LAYER_ALLOWED[_module_layer(rel)]
        for module in _imported_modules(path):
            target = _memory_target(module)
            if target is not None and target not in allowed:
                violations.append(f"{rel}: imports {module}")
    assert not violations, "layering violations:\n" + "\n".join(violations)


def test_embedding_stack_confined_to_index() -> None:
    violations: list[str] = []
    for path in _memory_modules():
        rel = path.relative_to(MEMORY_ROOT).as_posix()
        if rel in CHROMA_ALLOWED:
            continue
        for module in _imported_modules(path):
            if any(
                module == prefix or module.startswith(prefix + ".")
                for prefix in EMBEDDING_PREFIXES
            ):
                violations.append(f"{rel}: imports {module}")
    assert not violations, (
        "embedding stack leaked outside storage/index.py:\n" + "\n".join(violations)
    )


def test_llm_handles_confined() -> None:
    violations: list[str] = []
    for path in _memory_modules():
        rel = path.relative_to(MEMORY_ROOT).as_posix()
        if rel in LLM_ALLOWED:
            continue
        for module in _imported_modules(path):
            if any(
                module == prefix or module.startswith(prefix + ".")
                for prefix in LLM_PREFIXES
            ):
                violations.append(f"{rel}: imports {module}")
    assert not violations, "LLM handle outside allowed modules:\n" + "\n".join(
        violations
    )
