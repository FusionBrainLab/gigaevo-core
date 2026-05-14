"""Single-emission-site invariant for canonical events.

Each canonical event has exactly one architectural seam that emits it. Other
code must never bypass that seam by (a) calling `emit()` directly, (b) calling
the adversarial `emit_*` helpers, or (c) forging a canonical-event line with
`logger.info("[EVENT_NAME] ...")`.

This test walks the source tree and flags violations. If a new seam is
legitimately needed, add the file to `ALLOWED_SEAMS` with a comment explaining
which event(s) it emits.
"""

from __future__ import annotations

from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "gigaevo"

# Files allowed to call emit() / emit_*(...).  Each entry pairs a relative
# path with the event(s) it may emit.
ALLOWED_SEAMS: dict[str, tuple[str, ...]] = {
    # General emit() infrastructure
    "monitoring/emit.py": ("*",),  # defines emit()
    "monitoring/exception_sink.py": ("EXCEPTION",),
    # General seams (one event each)
    "programs/stages/base.py": ("STAGE_EXEC",),
    "llm/agents/base.py": ("LLM_CALL",),
    "evolution/engine/backpressure_sampler.py": ("BACKPRESSURE_SAMPLE",),
    # Adversarial helpers + their one caller each
    "adversarial/structured_logging.py": (
        "TRACKER_WRITE",
        "HOF_FETCH",
        "HOF_ROTATE",
        "CELL_PICK",
        "METRIC_EMIT",
    ),
    "adversarial/dg_tracker.py": ("TRACKER_WRITE",),
    "adversarial/opponent_provider.py": ("HOF_FETCH", "HOF_ROTATE", "CELL_PICK"),
    "adversarial/tracker_coverage_stages.py": ("METRIC_EMIT",),
}

# Canonical event names the registry defines (used to scan for forged
# `[EVENT_NAME] ...` log lines).
CANONICAL_EVENT_NAMES = {
    "GENERATION_BOUNDARY",
    "EXCEPTION",
    "STAGE_EXEC",
    "LLM_CALL",
    "BACKPRESSURE_SAMPLE",
    "METRIC_EMIT",
    "TRACKER_WRITE",
    "HOF_FETCH",
    "HOF_ROTATE",
    "CELL_PICK",
}

EMIT_CALL_RE = re.compile(r"\bemit\s*\(")
EMIT_HELPER_RE = re.compile(r"\bemit_[a-z_]+\s*\(")
FORGED_EVENT_RE = re.compile(r"\[([A-Z_]+)\]\s*\{")


def _source_files() -> list[Path]:
    return [p for p in SOURCE_ROOT.rglob("*.py") if p.is_file()]


def _rel(path: Path) -> str:
    return str(path.relative_to(SOURCE_ROOT)).replace("\\", "/")


def test_no_unapproved_emit_calls() -> None:
    """Only whitelisted seam files may call emit() / emit_*()."""
    violations: list[str] = []
    for path in _source_files():
        rel = _rel(path)
        if rel in ALLOWED_SEAMS:
            continue
        text = path.read_text(encoding="utf-8")
        # Strip comments and docstrings — false positives are not a goal
        # here, but skip obvious non-code contexts cheaply.
        for m in EMIT_CALL_RE.finditer(text):
            # Skip imports / attribute access like `.emit(` (methods named
            # emit on other classes are unrelated).
            start = max(0, m.start() - 1)
            if start >= 0 and text[start] == ".":
                continue
            line = text[: m.start()].count("\n") + 1
            violations.append(f"{rel}:{line}: bare `emit(` call outside seams")
        for m in EMIT_HELPER_RE.finditer(text):
            # Skip imports / attribute access
            start = max(0, m.start() - 1)
            if start >= 0 and text[start] == ".":
                continue
            snippet = text[m.start() : m.start() + 40]
            # Imports are fine — they are where helpers travel to seams.
            # The `emit_*(` pattern only flags real call sites; import lines
            # look like `from X import emit_foo` (no `(`), so they won't match
            # EMIT_HELPER_RE at all.
            line = text[: m.start()].count("\n") + 1
            violations.append(f"{rel}:{line}: `{snippet.split('(')[0]}(` outside seams")
    assert not violations, (
        "Canonical-event emission occurred outside whitelisted seams:\n"
        + "\n".join(violations)
    )


def test_no_forged_canonical_event_log_lines() -> None:
    """Nobody may forge a `[EVENT_NAME] {json}` log line via logger.info."""
    violations: list[str] = []
    for path in _source_files():
        rel = _rel(path)
        # Seam helpers construct the string — allowed.
        if rel in ALLOWED_SEAMS:
            continue
        text = path.read_text(encoding="utf-8")
        for m in FORGED_EVENT_RE.finditer(text):
            name = m.group(1)
            if name not in CANONICAL_EVENT_NAMES:
                continue
            line = text[: m.start()].count("\n") + 1
            violations.append(
                f"{rel}:{line}: forged [{name}] log line outside seams "
                f"— use emit() / emit_{name.lower()}() instead"
            )
    assert not violations, "Forged canonical-event lines:\n" + "\n".join(violations)
