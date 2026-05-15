"""Hostile-input integration tests for the sanitizer wiring in
``gigaevo/programs/dag/dag.py`` and ``gigaevo/runner/dag_runner.py``.

We exercise the exact log lines wrapped with ``sanitize_for_log`` by
constructing exceptions whose ``__str__`` carries ANSI / NUL / BIDI and
asserting the captured loguru output contains no raw control bytes.
"""

from __future__ import annotations

import re

from loguru import logger

from gigaevo.programs.core_types import StageError

# Patterns the sanitizer must strip.
_ANSI_RE = re.compile(r"\x1b\[")
_BIDI_RE = re.compile(r"[‪-‮⁦-⁩]")
_C0_RAW_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _capture():
    """Attach a list-sink to loguru. Returns (list, sink_id)."""
    captured: list[str] = []
    sink_id = logger.add(captured.append, format="{message}", level="TRACE")
    return captured, sink_id


def _assert_clean(captured: list[str]) -> None:
    joined = "\n".join(captured)
    assert not _ANSI_RE.search(joined), f"ANSI escape leaked: {joined!r}"
    assert not _BIDI_RE.search(joined), f"BIDI override leaked: {joined!r}"
    assert not _C0_RAW_RE.search(joined), f"Raw C0 control leaked: {joined!r}"


class TestDagRunnerErrorLogSanitization:
    """The ``DagRunner`` interpolates the ``__str__`` of caught
    exceptions into ``logger.error`` lines. After the sanitizer wiring,
    a hostile exception message must arrive at the sink in escaped form.
    """

    def test_sanitize_for_log_module_is_wired(self):
        """Smoke test: the dag_runner module imports the sanitizer."""
        from gigaevo.runner import dag_runner

        assert hasattr(dag_runner, "sanitize_for_log")

    def test_hostile_exception_string_is_escaped(self):
        """Simulate the wrapped log call directly. This is the exact
        idiom every error path in dag_runner.py now uses."""
        from gigaevo.runner.dag_runner import sanitize_for_log

        captured, sink_id = _capture()
        try:
            exc = RuntimeError("\x1b[31mCUDA OOM\x1b[0m\nat \x00addr ‮swap")
            logger.error(
                "[DagScheduler] program {} failed: {}",
                "deadbeef",
                sanitize_for_log(str(exc)),
            )
            _assert_clean(captured)
            # The visible text "CUDA OOM" must survive — sanitizer
            # preserves printable content.
            assert "CUDA OOM" in "\n".join(captured)
        finally:
            logger.remove(sink_id)

    def test_traceback_format_exc_is_escaped(self):
        """``traceback.format_exc()`` from a thrown hostile exception
        is also wrapped by the new logging path."""
        from gigaevo.runner.dag_runner import sanitize_for_log

        captured, sink_id = _capture()
        try:
            try:
                raise ValueError("\x1b[31mbad\x1b[0m\x00")
            except ValueError:
                import traceback as tb

                logger.error(
                    "[DagScheduler] Traceback:\n{}",
                    sanitize_for_log(tb.format_exc()),
                )
            _assert_clean(captured)
        finally:
            logger.remove(sink_id)


class TestDagStageErrorPretty:
    """``DAG._process_finished_task`` now wraps ``error.pretty()``.
    Verify the StageError model + the wrap together yield clean text
    on hostile input."""

    def test_stage_error_pretty_is_safe(self):
        # StageError validators scrub ``type`` / ``message`` /
        # ``traceback`` at construction. ``stage`` is supplied
        # internally (canonical class names — no LLM influence), so it
        # is not scrubbed. The dag-side ``sanitize_for_log`` wrap is
        # the defensive belt for the whole ``pretty()`` string.
        from gigaevo.utils.text_sanitize import sanitize_for_log

        err = StageError(
            type="\x1b[31mError\x1b[0m",
            message="\x00invalid\x1b[0m",
            stage="HostileStage",
            traceback="line1\n\x1b[31mline2\x1b[0m\n‮line3",
        )
        rendered = sanitize_for_log(err.pretty(include_traceback=True))
        assert "\x1b[" not in rendered
        assert "\x00" not in rendered
        assert "‮" not in rendered

    def test_dag_module_imports_sanitizer(self):
        from gigaevo.programs.dag import dag as dag_mod

        assert hasattr(dag_mod, "sanitize_for_log")
