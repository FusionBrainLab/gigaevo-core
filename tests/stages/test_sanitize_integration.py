"""Hostile-input integration tests for the sanitizer wiring in
``gigaevo/programs/stages/`` and ``gigaevo/programs/dag/``.

Each test feeds ANSI escape sequences, NUL bytes, BIDI overrides, or
lone UTF-16 surrogates into one of the call sites surgically wired with
``sanitize_for_log`` / ``clean_identifier`` and asserts that the
relevant downstream surface (loguru sink contents, optuna trial keys,
SyntaxError args) is free of the offending bytes — without disturbing
the surrounding logic.
"""

from __future__ import annotations

from pathlib import Path
import re

from loguru import logger
import pytest

from gigaevo.programs.core_types import StageError
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.optimization.optuna.models import (
    CodeModification,
    OptunaSearchSpace,
    ParamSpec,
)
from gigaevo.programs.stages.optimization.optuna.stage import OptunaOptimizationStage
from gigaevo.programs.stages.optimization.utils import evaluate_single
from gigaevo.programs.stages.python_executors.execution import PythonCodeExecutor
from gigaevo.programs.stages.python_executors.wrapper import ExecRunnerError
from gigaevo.programs.stages.validation import ValidateCodeStage

# Byte patterns we never want to see on log sinks or as optuna trial keys.
_ANSI_RE = re.compile(r"\x1b\[")
_BIDI_RE = re.compile(r"[‪-‮⁦-⁩]")
_C0_RAW_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _attach_sink() -> list[str]:
    """Add a memory loguru sink for the duration of a single test.

    Returns the underlying list that captures every emitted log message;
    each test removes the sink at the end via ``logger.remove``.
    """
    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}", level="TRACE")
    # Stash the sink id on the list so the caller can remove it.
    messages.append(f"__sink_id__={sink_id}")
    return messages


def _detach_sink(messages: list[str]) -> None:
    for entry in messages:
        if isinstance(entry, str) and entry.startswith("__sink_id__="):
            sink_id = int(entry.split("=", 1)[1])
            logger.remove(sink_id)
            return


def _assert_sink_clean(messages: list[str]) -> None:
    for line in messages:
        if line.startswith("__sink_id__="):
            continue
        assert not _ANSI_RE.search(line), f"ANSI escape leaked into log: {line!r}"
        assert not _BIDI_RE.search(line), f"BIDI override leaked into log: {line!r}"
        assert not _C0_RAW_RE.search(line), f"Raw C0 control leaked into log: {line!r}"


# ---------------------------------------------------------------------------
# validation.py — SyntaxError text scrubbing
# ---------------------------------------------------------------------------


class TestValidationSyntaxErrorSanitized:
    """``ValidateCodeStage`` interpolates ``e.msg`` / ``e.text`` into a
    re-raised ``SyntaxError``. With a hostile compiler message that
    text would propagate verbatim into every downstream consumer."""

    async def test_syntax_error_message_strips_ansi(self):
        # Inject a raw \x1b sequence into the code as a comment so the
        # parser quotes it back in e.text.
        code = "def foo(\x1b[31m  # bad\n"
        stage = ValidateCodeStage(timeout=30.0)
        prog = Program(code=code, state=ProgramState.RUNNING)
        with pytest.raises(SyntaxError) as ei:
            await stage.compute(prog)
        rendered = str(ei.value)
        assert "\x1b[" not in rendered
        assert "\x1b" not in rendered or "\\x1b" in rendered

    async def test_syntax_error_message_escapes_nul(self):
        code = "def bar(\x00\n"
        stage = ValidateCodeStage(timeout=30.0)
        prog = Program(code=code, state=ProgramState.RUNNING)
        with pytest.raises(SyntaxError) as ei:
            await stage.compute(prog)
        rendered = str(ei.value)
        assert "\x00" not in rendered


# ---------------------------------------------------------------------------
# optuna/stage.py — ParamSpec.name as an Optuna trial key
# ---------------------------------------------------------------------------


class TestOptunaParamNameCleaning:
    """``OptunaOptimizationStage._apply_modifications`` must scrub
    ``ParamSpec.name`` before the name is handed to ``trial.suggest_*``
    where it becomes an optuna storage key."""

    @staticmethod
    def _make_stage(tmp_path: Path) -> OptunaOptimizationStage:
        # Minimal validator file so __init__ succeeds.
        validator = tmp_path / "validator.py"
        validator.write_text(
            "def validate(_):\n    return {'score': 0.0}\n",
            encoding="utf-8",
        )
        return OptunaOptimizationStage(
            llm=None,  # type: ignore[arg-type]  # not used here
            validator_path=validator,
            score_key="score",
            timeout=30.0,
        )

    def test_nul_in_paramspec_name_cleaned(self, tmp_path: Path):
        stage = self._make_stage(tmp_path)
        param = ParamSpec(
            name="\x00../etc/passwd",
            initial_value=1.0,
            param_type="float",
            low=0.0,
            high=10.0,
            reason="hostile",
        )
        mod = CodeModification(
            start_line=1,
            end_line=1,
            parameterized_snippet="x = _optuna_params['\x00../etc/passwd']",
        )
        ss = OptunaSearchSpace(
            parameters=[param],
            modifications=[mod],
            reasoning="hostile",
        )
        # _apply_modifications mutates ParamSpec.name in place.
        try:
            stage._apply_modifications("x = 1\n", ss)
        except Exception:
            # We don't care whether AST parse succeeds — we only care
            # that the param name was scrubbed before any optuna call.
            pass
        assert "\x00" not in param.name
        assert ".." in param.name or "etc" in param.name  # printable survived
        # And the name is now safe for use as an optuna key (no control bytes).
        assert _C0_RAW_RE.search(param.name) is None

    def test_ansi_in_paramspec_name_cleaned(self, tmp_path: Path):
        stage = self._make_stage(tmp_path)
        param = ParamSpec(
            name="\x1b[31mred_param\x1b[0m",
            initial_value=1,
            param_type="int",
            low=0,
            high=10,
            reason="hostile",
        )
        ss = OptunaSearchSpace(
            parameters=[param],
            modifications=[
                CodeModification(
                    start_line=1, end_line=1, parameterized_snippet="x = 1"
                )
            ],
            reasoning="hostile",
        )
        try:
            stage._apply_modifications("x = 1\n", ss)
        except Exception:
            pass
        assert "\x1b" not in param.name
        assert "red_param" in param.name

    def test_only_bad_chars_falls_back_to_positional(self, tmp_path: Path):
        stage = self._make_stage(tmp_path)
        # Every character outside the identifier charset.
        param = ParamSpec(
            name="\x00\x1b‮",
            initial_value=1.0,
            param_type="float",
            low=0.0,
            high=1.0,
            reason="hostile",
        )
        ss = OptunaSearchSpace(
            parameters=[param],
            modifications=[
                CodeModification(
                    start_line=1, end_line=1, parameterized_snippet="x = 1"
                )
            ],
            reasoning="hostile",
        )
        try:
            stage._apply_modifications("x = 1\n", ss)
        except Exception:
            pass
        assert param.name == "param_0"


# ---------------------------------------------------------------------------
# execution.py — subprocess stderr through the warning log
# ---------------------------------------------------------------------------


class _HostilePythonCodeExecutor(PythonCodeExecutor):
    """Bypass the subprocess pool entirely; raise a precomposed
    ``ExecRunnerError`` so we can drive the ``except`` branch
    deterministically without spinning a real worker."""

    def __init__(self, hostile_stderr: str, **kwargs):
        super().__init__(**kwargs)
        self._hostile_stderr = hostile_stderr

    async def compute(self, program):  # noqa: D401 — override
        # Re-run the parent implementation but force the error path.
        # Easiest: call the parent's exception block directly by invoking
        # the same logger pattern via the parent's compute, but inject the
        # error. We mimic the structure manually for stability.
        raise ExecRunnerError(
            returncode=1,
            stderr=self._hostile_stderr,
            stdout_bytes=b"",
        )


class TestExecutionWarningLogSanitized:
    async def test_hostile_stderr_does_not_leak_to_loguru(self):
        # Build a real PythonCodeExecutor with a stub that raises
        # ExecRunnerError on the inner await — exercises the exact
        # logger.warning line we wrapped.
        hostile = "\x1b[31mCUDA error\x1b[0m: \x00 invalid value ‮malicious"
        stage = PythonCodeExecutor(timeout=30.0)

        async def _fake_runner(**_kw):
            raise ExecRunnerError(
                returncode=1, stderr=hostile, stdout_bytes=b""
            )

        # Monkey-patch the bound name imported inside execution.py.
        from gigaevo.programs.stages.python_executors import execution as exec_mod

        original = exec_mod.run_exec_runner
        exec_mod.run_exec_runner = _fake_runner  # type: ignore[assignment]
        messages = _attach_sink()
        try:
            prog = Program(code="def run_code(): return 1", state=ProgramState.RUNNING)
            result = await stage.compute(prog)
            # The stage should return a FAILED result, not raise.
            assert hasattr(result, "status")
            _assert_sink_clean(messages)
        finally:
            _detach_sink(messages)
            exec_mod.run_exec_runner = original  # type: ignore[assignment]

    async def test_stage_error_traceback_scrubbed_by_validator(self):
        # The StageError validator must convert any control bytes in
        # the constructed StageError.traceback into escaped form.
        hostile = "Traceback:\n\x1b[31mline\x1b[0m\nFinal\x00"
        err = StageError(type="X", message="m", traceback=hostile)
        assert "\x1b[" not in (err.traceback or "")
        assert "\x00" not in (err.traceback or "")


# ---------------------------------------------------------------------------
# optimization/utils.py — evaluate_single ExecRunnerError path
# ---------------------------------------------------------------------------


class TestEvaluateSingleSanitization:
    async def test_exec_runner_error_returned_message_is_sanitized(
        self, monkeypatch
    ):
        hostile_stderr = "\x1b[31mcompile error\x1b[0m\nlast: \x00bad"

        async def _fake_runner(**_kw):
            raise ExecRunnerError(
                returncode=1, stderr=hostile_stderr, stdout_bytes=b""
            )

        from gigaevo.programs.stages.optimization import utils as utils_mod

        monkeypatch.setattr(utils_mod, "run_exec_runner", _fake_runner)
        messages = _attach_sink()
        try:
            scores, err = await evaluate_single(
                eval_code="def _opt(): return 1",
                eval_fn_name="_opt",
                context=None,
                score_key="score",
                python_path=[],
                timeout=5,
                max_memory_mb=None,
                log_tag="Unit",
            )
            assert scores is None
            assert err is not None
            assert "\x1b" not in err
            assert "\x00" not in err
            _assert_sink_clean(messages)
        finally:
            _detach_sink(messages)
