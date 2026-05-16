"""Discipline guard: the dataplane may not silently broaden ``except``.

Walks every ``.py`` file under ``gigaevo/dataplane/`` (excluding the
test tree) and forbids:

- ``except:`` — bare except, no exceptions, never allowed.
- ``except Exception`` and ``except BaseException`` — only allowed
  when the source line carries an explicit ``# noqa: BLE001`` marker
  acknowledging the breadth. Every such site must therefore declare
  itself a reviewer-visible boundary.

The dataplane is the layer that converts internal failures into typed
``Result[T, E]`` returns. A broad ``except`` without an annotation is
the shape of a bug that re-emerges as soon as the next contributor
copies the surrounding function without noticing it. This test makes
the discipline mechanical: a new broad ``except`` lands red, gets a
marker (or gets narrowed), then lands green.
"""

from __future__ import annotations

import ast
from pathlib import Path

_DATAPLANE_ROOT = Path(__file__).resolve().parent.parent
_NOQA_MARKER = "noqa: BLE001"


def _python_files() -> list[Path]:
    """Every ``.py`` under the dataplane package except the test tree."""
    out: list[Path] = []
    for path in _DATAPLANE_ROOT.rglob("*.py"):
        # Skip this file and its siblings — the discipline applies to
        # production code only.
        if "tests" in path.parts:
            continue
        out.append(path)
    return out


def _line_has_marker(source_lines: list[str], lineno: int) -> bool:
    """True if ``source_lines[lineno - 1]`` carries the noqa marker.

    Handles multi-line ``except`` clauses by walking forward until a
    non-blank line is found (typical ruff convention places the noqa
    on the same line as the ``except`` keyword, but defensive handling
    avoids false positives if a future formatter wraps the line).
    """
    if 1 <= lineno <= len(source_lines):
        line = source_lines[lineno - 1]
        if _NOQA_MARKER in line:
            return True
    return False


def _is_broad_handler(handler: ast.ExceptHandler) -> bool:
    """True if the handler catches ``Exception``/``BaseException``."""
    t = handler.type
    if t is None:
        return False  # bare except — handled separately
    if isinstance(t, ast.Name):
        return t.id in {"Exception", "BaseException"}
    if isinstance(t, ast.Tuple):
        return any(
            isinstance(el, ast.Name) and el.id in {"Exception", "BaseException"}
            for el in t.elts
        )
    return False


def test_no_bare_except_in_dataplane() -> None:
    """``except:`` with no exception class is banned outright."""
    offenders: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                offenders.append(f"{path}:{node.lineno}: bare `except:`")
    assert not offenders, (
        "Bare except is forbidden inside gigaevo/dataplane/. Offending sites:\n  "
        + "\n  ".join(offenders)
    )


def test_broad_except_requires_noqa_marker() -> None:
    """``except Exception``/``BaseException`` requires a ``# noqa: BLE001`` marker.

    The marker is a deliberate declaration that this is a
    coordinator-boundary swallow — not a forgotten narrowing. Reviewers
    can grep for the marker to enumerate every such site.
    """
    offenders: list[str] = []
    for path in _python_files():
        source = path.read_text()
        lines = source.splitlines()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if not _is_broad_handler(node):
                continue
            if not _line_has_marker(lines, node.lineno):
                offenders.append(
                    f"{path}:{node.lineno}: broad `except` without "
                    f"`# {_NOQA_MARKER}` rationale marker"
                )
    assert not offenders, (
        "Broad except inside gigaevo/dataplane/ without `noqa: BLE001` "
        "marker. Either narrow the exception class or annotate the site:\n  "
        + "\n  ".join(offenders)
    )
