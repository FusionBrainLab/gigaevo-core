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


def _line_has_marker(
    source_lines: list[str], lineno: int, end_lineno: int | None = None
) -> bool:
    """True if any line in the ``except`` clause carries the noqa marker.

    A single-line clause has ``end_lineno == lineno`` and only one line
    to inspect. A wrapped clause such as ``except (A, Exception):`` spans
    several physical lines; ruff convention places the noqa on the line
    closing the parenthesised tuple, so the full span ``lineno
    .. end_lineno`` is checked.
    """
    if end_lineno is None or end_lineno < lineno:
        end_lineno = lineno
    if not 1 <= lineno <= len(source_lines):
        return False
    upper = min(end_lineno, len(source_lines))
    return any(_NOQA_MARKER in source_lines[i - 1] for i in range(lineno, upper + 1))


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
            if not _line_has_marker(lines, node.lineno, node.end_lineno):
                offenders.append(
                    f"{path}:{node.lineno}: broad `except` without "
                    f"`# {_NOQA_MARKER}` rationale marker"
                )
    assert not offenders, (
        "Broad except inside gigaevo/dataplane/ without `noqa: BLE001` "
        "marker. Either narrow the exception class or annotate the site:\n  "
        + "\n  ".join(offenders)
    )


def _scan_source(source: str) -> list[tuple[int, str]]:
    """Run the same checks against an in-memory source; return offenders.

    Used by the self-tests to verify the discipline fires on known-bad
    constructs without writing temporary files to the dataplane tree.
    """
    out: list[tuple[int, str]] = []
    lines = source.splitlines()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.type is None:
            out.append((node.lineno, "bare"))
            continue
        if _is_broad_handler(node) and not _line_has_marker(
            lines, node.lineno, node.end_lineno
        ):
            out.append((node.lineno, "broad"))
    return out


class TestSelfChecks:
    """The guard catches the violations it claims to catch."""

    def test_bare_except_caught(self) -> None:
        src = "try:\n    pass\nexcept:\n    pass\n"
        assert _scan_source(src) == [(3, "bare")]

    def test_broad_except_without_marker_caught(self) -> None:
        src = "try:\n    pass\nexcept Exception:\n    pass\n"
        assert _scan_source(src) == [(3, "broad")]

    def test_broad_except_with_marker_accepted(self) -> None:
        src = "try:\n    pass\nexcept Exception:  # noqa: BLE001\n    pass\n"
        assert _scan_source(src) == []

    def test_broad_except_tuple_form_caught(self) -> None:
        src = "try:\n    pass\nexcept (ValueError, Exception):\n    pass\n"
        assert _scan_source(src) == [(3, "broad")]

    def test_multiline_tuple_marker_on_closing_paren_accepted(self) -> None:
        src = (
            "try:\n"
            "    pass\n"
            "except (\n"
            "    ValueError,\n"
            "    Exception,\n"
            "):  # noqa: BLE001\n"
            "    pass\n"
        )
        assert _scan_source(src) == []

    def test_baseexception_caught(self) -> None:
        src = "try:\n    pass\nexcept BaseException:\n    pass\n"
        assert _scan_source(src) == [(3, "broad")]

    def test_narrow_except_unaffected(self) -> None:
        src = "try:\n    pass\nexcept ValueError:\n    pass\n"
        assert _scan_source(src) == []
