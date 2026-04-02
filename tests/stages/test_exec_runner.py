"""Tests for exec_runner.py — the subprocess worker that executes Python code snippets."""

from __future__ import annotations

import io
import linecache
import struct
import sys
import types
from typing import Any
from unittest.mock import patch

import cloudpickle
import pytest

from gigaevo.programs.stages.python_executors.exec_runner import (
    _CODE_FILENAME,
    _format_syntax_error,
    _load_module_from_code,
    _register_source,
    _run_one,
    _worker_loop,
    _write_code_context,
    main,
)

# ---------------------------------------------------------------------------
# _register_source
# ---------------------------------------------------------------------------


class TestRegisterSource:
    def test_single_line_populates_linecache(self) -> None:
        """Single-line source is stored in linecache with correct metadata."""
        source = "x = 1\n"
        filename = "__test_single_line__.py"
        _register_source(filename, source)
        assert filename in linecache.cache
        size, _, lines, stored_fn = linecache.cache[filename]
        assert size == len(source)
        assert lines == ["x = 1\n"]
        assert stored_fn == filename
        # cleanup
        linecache.cache.pop(filename, None)

    def test_multiline_populates_linecache(self) -> None:
        """Multi-line source is split into individual lines."""
        source = "a = 1\nb = 2\nc = 3\n"
        filename = "__test_multiline__.py"
        _register_source(filename, source)
        _, _, lines, _ = linecache.cache[filename]
        assert lines == ["a = 1\n", "b = 2\n", "c = 3\n"]
        linecache.cache.pop(filename, None)


# ---------------------------------------------------------------------------
# _load_module_from_code
# ---------------------------------------------------------------------------


class TestLoadModuleFromCode:
    def test_valid_code_returns_module(self) -> None:
        """Valid code produces a module with the expected attribute."""
        mod = _load_module_from_code("x = 42\n")
        assert isinstance(mod, types.ModuleType)
        assert mod.x == 42
        sys.modules.pop("user_code", None)
        linecache.cache.pop(_CODE_FILENAME, None)

    def test_module_registered_in_sys_modules(self) -> None:
        """The loaded module is inserted into sys.modules."""
        _load_module_from_code("y = 99\n")
        assert "user_code" in sys.modules
        assert sys.modules["user_code"].y == 99
        sys.modules.pop("user_code", None)
        linecache.cache.pop(_CODE_FILENAME, None)

    def test_custom_mod_name(self) -> None:
        """A custom module name is used when specified."""
        mod = _load_module_from_code("z = 7\n", mod_name="custom_mod")
        assert mod.__name__ == "custom_mod"
        assert "custom_mod" in sys.modules
        sys.modules.pop("custom_mod", None)
        linecache.cache.pop(_CODE_FILENAME, None)

    def test_syntax_error_raises(self) -> None:
        """Code with a syntax error raises SyntaxError."""
        with pytest.raises(SyntaxError):
            _load_module_from_code("def f(\n  return 1")
        sys.modules.pop("user_code", None)
        linecache.cache.pop(_CODE_FILENAME, None)


# ---------------------------------------------------------------------------
# _write_code_context
# ---------------------------------------------------------------------------


class TestWriteCodeContext:
    def test_user_frame_formatting(self) -> None:
        """When the traceback has a user_code frame, context lines are written."""
        code = "def boom():\n    x = 1\n    raise ValueError('oops')\n"
        _register_source(_CODE_FILENAME, code)
        try:
            mod = _load_module_from_code(code)
            mod.boom()
        except ValueError as exc:
            buf = io.StringIO()
            _write_code_context(exc, out=buf)
            output = buf.getvalue()
            assert "Code context" in output
            assert ">>" in output  # the offending line marker
        finally:
            sys.modules.pop("user_code", None)
            linecache.cache.pop(_CODE_FILENAME, None)

    def test_no_user_frames_writes_nothing(self) -> None:
        """When the traceback has no user_code frames, nothing is written."""
        try:
            raise RuntimeError("not from user code")
        except RuntimeError as exc:
            buf = io.StringIO()
            _write_code_context(exc, out=buf)
            assert buf.getvalue() == ""

    def test_internal_error_in_write(self) -> None:
        """If an exception occurs inside _write_code_context, it writes the error."""
        # Create an exception with a bogus __traceback__ that triggers
        # an error inside extract_tb
        exc = RuntimeError("test")
        exc.__traceback__ = None  # no traceback at all
        buf = io.StringIO()
        # With __traceback__=None, extract_tb returns [] => no user frames => empty
        _write_code_context(exc, out=buf)
        # No user frames, so nothing written (graceful handling)
        assert buf.getvalue() == ""


# ---------------------------------------------------------------------------
# _format_syntax_error
# ---------------------------------------------------------------------------


class TestFormatSyntaxError:
    def test_text_and_offset(self) -> None:
        """SyntaxError with text and offset produces caret pointer."""
        err = SyntaxError("invalid syntax")
        err.filename = "test.py"
        err.lineno = 5
        err.text = "x = 1 +\n"
        err.offset = 5
        result = _format_syntax_error(err)
        assert "Traceback (most recent call last):" in result
        assert '"test.py", line 5' in result
        assert "x = 1 +" in result
        assert "^" in result
        assert "SyntaxError: invalid syntax" in result

    def test_no_text(self) -> None:
        """SyntaxError without text skips the line and caret."""
        err = SyntaxError("unexpected EOF")
        err.filename = "test.py"
        err.lineno = 1
        err.text = None
        err.offset = None
        result = _format_syntax_error(err)
        assert '"test.py", line 1' in result
        assert "SyntaxError: unexpected EOF" in result
        assert "^" not in result

    def test_no_offset(self) -> None:
        """SyntaxError with text but no offset shows line without caret."""
        err = SyntaxError("bad token")
        err.filename = "test.py"
        err.lineno = 3
        err.text = "foo bar\n"
        err.offset = None
        result = _format_syntax_error(err)
        assert "foo bar" in result
        assert "^" not in result
        assert "SyntaxError: bad token" in result


# ---------------------------------------------------------------------------
# _run_one
# ---------------------------------------------------------------------------


class TestRunOne:
    def test_success(self) -> None:
        """Successful execution returns (result, None)."""
        payload: dict[str, Any] = {
            "code": "def solve(a, b): return a + b",
            "function_name": "solve",
            "args": [3, 4],
            "kwargs": {},
        }
        result, error = _run_one(payload)
        assert result == 7
        assert error is None
        sys.modules.pop("user_code", None)
        linecache.cache.pop(_CODE_FILENAME, None)

    def test_missing_function(self) -> None:
        """Calling a nonexistent function returns an error dict."""
        payload: dict[str, Any] = {
            "code": "def other(): return 1",
            "function_name": "nonexistent",
        }
        result, error = _run_one(payload)
        assert result is None
        assert error is not None
        assert error["_error"] is True
        assert "not found" in error["stderr"] or "not callable" in error["stderr"]
        sys.modules.pop("user_code", None)
        linecache.cache.pop(_CODE_FILENAME, None)

    def test_runtime_error(self) -> None:
        """A runtime error returns an error dict with traceback."""
        payload: dict[str, Any] = {
            "code": "def boom(): raise ValueError('kaboom')",
            "function_name": "boom",
        }
        result, error = _run_one(payload)
        assert result is None
        assert error is not None
        assert error["returncode"] == 1
        assert "ValueError" in error["stderr"]
        assert "kaboom" in error["stderr"]
        sys.modules.pop("user_code", None)
        linecache.cache.pop(_CODE_FILENAME, None)

    def test_syntax_error(self) -> None:
        """Code with a syntax error returns an error dict."""
        payload: dict[str, Any] = {
            "code": "def f(\n  return 1",
            "function_name": "f",
        }
        result, error = _run_one(payload)
        assert result is None
        assert error is not None
        assert error["_error"] is True
        assert "SyntaxError" in error["stderr"]
        sys.modules.pop("user_code", None)
        linecache.cache.pop(_CODE_FILENAME, None)

    def test_stdout_capture(self) -> None:
        """Print statements inside user code are captured."""
        payload: dict[str, Any] = {
            "code": "def speak():\n    print('hello')\n    return 42\n",
            "function_name": "speak",
        }
        result, error = _run_one(payload)
        assert result == 42
        assert error is None
        sys.modules.pop("user_code", None)
        linecache.cache.pop(_CODE_FILENAME, None)

    def test_python_path_replaces_shadowed_top_level_module(self, tmp_path) -> None:
        """Problem-local modules should override stale cached modules in workers."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / "helper.py").write_text("VALUE = 'repo'\n", encoding="utf-8")

        problem_dir = tmp_path / "problem"
        problem_dir.mkdir()
        (problem_dir / "helper.py").write_text("VALUE = 'problem'\n", encoding="utf-8")

        sys.path.insert(0, str(repo_dir))
        try:
            import helper

            assert helper.VALUE == "repo"

            payload: dict[str, Any] = {
                "code": "from helper import VALUE\n\ndef read_value(): return VALUE\n",
                "function_name": "read_value",
                "python_path": [str(problem_dir)],
            }
            result, error = _run_one(payload)
            assert error is None
            assert result == "problem"
        finally:
            sys.modules.pop("helper", None)
            sys.modules.pop("user_code", None)
            linecache.cache.pop(_CODE_FILENAME, None)
            sys.path = [p for p in sys.path if p != str(repo_dir)]

    def test_python_path_is_moved_ahead_of_repo_root(self, tmp_path) -> None:
        """A reused worker must reorder python_path entries to the front."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / "helper.py").write_text("VALUE = 'repo'\n", encoding="utf-8")

        problem_dir = tmp_path / "problem"
        problem_dir.mkdir()
        (problem_dir / "helper.py").write_text("VALUE = 'problem'\n", encoding="utf-8")

        sys.path.insert(0, str(repo_dir))
        sys.path.append(str(problem_dir))
        try:
            payload: dict[str, Any] = {
                "code": "from helper import VALUE\n\ndef read_value(): return VALUE\n",
                "function_name": "read_value",
                "python_path": [str(problem_dir)],
            }
            result, error = _run_one(payload)
            assert error is None
            assert result == "problem"
        finally:
            sys.modules.pop("helper", None)
            sys.modules.pop("user_code", None)
            linecache.cache.pop(_CODE_FILENAME, None)
            sys.path = [
                p for p in sys.path if p not in {str(repo_dir), str(problem_dir)}
            ]


# ---------------------------------------------------------------------------
# _worker_loop
# ---------------------------------------------------------------------------


class TestWorkerLoop:
    def test_single_message_roundtrip(self) -> None:
        """Send one payload via the length-prefixed protocol and get a result back."""
        payload: dict[str, Any] = {
            "code": "def add(a, b): return a + b",
            "function_name": "add",
            "args": [10, 20],
            "kwargs": {},
        }
        payload_bytes = cloudpickle.dumps(payload)
        # Build stdin: 4-byte length prefix + payload + 4-byte zero (exit signal)
        stdin_buf = io.BytesIO()
        stdin_buf.write(struct.pack(">I", len(payload_bytes)))
        stdin_buf.write(payload_bytes)
        stdin_buf.write(struct.pack(">I", 0))  # zero-length sentinel to exit
        stdin_buf.seek(0)

        stdout_buf = io.BytesIO()

        with patch.object(
            sys, "stdin", new=type("FakeStdin", (), {"buffer": stdin_buf})()
        ):
            with patch.object(
                sys, "stdout", new=type("FakeStdout", (), {"buffer": stdout_buf})()
            ):
                _worker_loop()

        stdout_buf.seek(0)
        resp_len_bytes = stdout_buf.read(4)
        assert len(resp_len_bytes) == 4
        (resp_len,) = struct.unpack(">I", resp_len_bytes)
        resp_body = stdout_buf.read(resp_len)
        result = cloudpickle.loads(resp_body)
        assert result == 30

        sys.modules.pop("user_code", None)
        linecache.cache.pop(_CODE_FILENAME, None)

    def test_empty_input_exits(self) -> None:
        """An empty stdin causes the loop to exit immediately."""
        stdin_buf = io.BytesIO(b"")
        stdout_buf = io.BytesIO()

        with patch.object(
            sys, "stdin", new=type("FakeStdin", (), {"buffer": stdin_buf})()
        ):
            with patch.object(
                sys, "stdout", new=type("FakeStdout", (), {"buffer": stdout_buf})()
            ):
                _worker_loop()

        # No output should have been written
        assert stdout_buf.tell() == 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


class TestMain:
    def test_worker_flag_dispatches_to_worker_loop(self) -> None:
        """When --worker is in sys.argv, main() calls _worker_loop."""
        with patch(
            "gigaevo.programs.stages.python_executors.exec_runner._worker_loop"
        ) as mock_loop:
            with patch.object(sys, "argv", ["exec_runner.py", "--worker"]):
                main()
            mock_loop.assert_called_once()

    def test_main_non_worker_success(self) -> None:
        """main() without --worker reads stdin, executes payload, writes result."""
        payload = {
            "code": "def run(a): return a * 2",
            "function_name": "run",
            "args": [21],
            "kwargs": {},
        }
        stdin_buf = io.BytesIO(cloudpickle.dumps(payload))
        stdout_buf = io.BytesIO()
        with patch.object(sys, "argv", ["exec_runner.py"]):
            with patch.object(sys, "stdin", new=type("S", (), {"buffer": stdin_buf})()):
                with patch.object(
                    sys, "stdout", new=type("S", (), {"buffer": stdout_buf})()
                ):
                    main()
        stdout_buf.seek(0)
        result = cloudpickle.load(stdout_buf)
        assert result == 42
        sys.modules.pop("user_code", None)
        linecache.cache.pop(_CODE_FILENAME, None)

    def test_main_non_worker_error_exits_nonzero(self) -> None:
        """main() without --worker exits with code 1 on user code error."""
        payload = {
            "code": "def run(): raise RuntimeError('boom')",
            "function_name": "run",
        }
        stdin_buf = io.BytesIO(cloudpickle.dumps(payload))
        stderr_buf = io.StringIO()
        with patch.object(sys, "argv", ["exec_runner.py"]):
            with patch.object(sys, "stdin", new=type("S", (), {"buffer": stdin_buf})()):
                with patch.object(sys, "stderr", new=stderr_buf):
                    with pytest.raises(SystemExit) as exc_info:
                        main()
        assert exc_info.value.code == 1
        assert "RuntimeError" in stderr_buf.getvalue()
        sys.modules.pop("user_code", None)
        linecache.cache.pop(_CODE_FILENAME, None)


# ---------------------------------------------------------------------------
# Additional tests from audit
# ---------------------------------------------------------------------------


class TestRegisterSourceOverwrite:
    def test_overwrite_existing_entry(self) -> None:
        filename = "__test_overwrite__.py"
        _register_source(filename, "x = 1\n")
        _register_source(filename, "x = 999\n")
        _, _, lines, _ = linecache.cache[filename]
        assert lines == ["x = 999\n"]
        linecache.cache.pop(filename, None)


class TestLoadModuleLinecache:
    def test_load_module_registers_source_in_linecache(self) -> None:
        code = "result = 123\n"
        _load_module_from_code(code)
        assert _CODE_FILENAME in linecache.cache
        _, _, lines, _ = linecache.cache[_CODE_FILENAME]
        assert lines == ["result = 123\n"]
        sys.modules.pop("user_code", None)
        linecache.cache.pop(_CODE_FILENAME, None)


class TestRunOneEdgeCases:
    def test_runtime_error_includes_code_context(self) -> None:
        """Error stderr includes the code-context annotation with '>>' marker."""
        payload: dict[str, Any] = {
            "code": "def boom():\n    x = 1\n    raise ValueError('kaboom')\n",
            "function_name": "boom",
        }
        result, error = _run_one(payload)
        assert result is None
        assert ">>" in error["stderr"]
        sys.modules.pop("user_code", None)
        linecache.cache.pop(_CODE_FILENAME, None)

    def test_stdout_not_leaked_to_real_stdout(self) -> None:
        """User print() output must not appear on real process stdout."""
        payload: dict[str, Any] = {
            "code": "def speak():\n    print('should_not_appear')\n    return 42\n",
            "function_name": "speak",
        }
        real_stdout = io.StringIO()
        with patch.object(sys, "stdout", real_stdout):
            result, error = _run_one(payload)
        assert result == 42
        assert "should_not_appear" not in real_stdout.getvalue()
        sys.modules.pop("user_code", None)
        linecache.cache.pop(_CODE_FILENAME, None)

    def test_runtime_error_with_prior_print_includes_captured(self) -> None:
        """Captured stdout before error is included in the error stderr."""
        payload: dict[str, Any] = {
            "code": (
                "def boom():\n"
                "    print('before_error_output')\n"
                "    raise ValueError('after')\n"
            ),
            "function_name": "boom",
        }
        result, error = _run_one(payload)
        assert result is None
        assert "before_error_output" in error["stderr"]
        sys.modules.pop("user_code", None)
        linecache.cache.pop(_CODE_FILENAME, None)


class TestFormatSyntaxErrorEdgeCases:
    def test_out_of_range_offset_no_caret(self) -> None:
        err = SyntaxError("bad")
        err.filename = "f.py"
        err.lineno = 1
        err.text = "x\n"
        err.offset = 999
        result = _format_syntax_error(err)
        assert "x" in result
        assert "^" not in result

    def test_zero_offset_no_caret(self) -> None:
        err = SyntaxError("bad")
        err.filename = "f.py"
        err.lineno = 1
        err.text = "x\n"
        err.offset = 0
        result = _format_syntax_error(err)
        assert "^" not in result


class TestWorkerLoopEdgeCases:
    def test_multiple_messages_processed(self) -> None:
        """Worker loop processes all messages until the zero sentinel."""
        payloads = [
            {"code": "def f(): return 1", "function_name": "f"},
            {"code": "def f(): return 2", "function_name": "f"},
            {"code": "def f(): return 3", "function_name": "f"},
        ]
        stdin_buf = io.BytesIO()
        for p in payloads:
            b = cloudpickle.dumps(p)
            stdin_buf.write(struct.pack(">I", len(b)))
            stdin_buf.write(b)
        stdin_buf.write(struct.pack(">I", 0))
        stdin_buf.seek(0)

        stdout_buf = io.BytesIO()
        with patch.object(sys, "stdin", new=type("S", (), {"buffer": stdin_buf})()):
            with patch.object(
                sys, "stdout", new=type("S", (), {"buffer": stdout_buf})()
            ):
                _worker_loop()

        stdout_buf.seek(0)
        results = []
        for _ in payloads:
            (n,) = struct.unpack(">I", stdout_buf.read(4))
            results.append(cloudpickle.loads(stdout_buf.read(n)))
        assert results == [1, 2, 3]
        sys.modules.pop("user_code", None)
        linecache.cache.pop(_CODE_FILENAME, None)

    def test_truncated_payload_exits_cleanly(self) -> None:
        """If stdin closes before delivering all bytes, loop exits without writing."""
        stdin_buf = io.BytesIO(struct.pack(">I", 100) + b"truncated!")
        stdout_buf = io.BytesIO()
        with patch.object(sys, "stdin", new=type("S", (), {"buffer": stdin_buf})()):
            with patch.object(
                sys, "stdout", new=type("S", (), {"buffer": stdout_buf})()
            ):
                _worker_loop()
        assert stdout_buf.tell() == 0

    def test_error_payload_returned_as_error_dict(self) -> None:
        """Worker writes back error dict when user code raises."""
        payload = {
            "code": "def f(): raise ValueError('worker_err')",
            "function_name": "f",
        }
        b = cloudpickle.dumps(payload)
        stdin_buf = io.BytesIO(struct.pack(">I", len(b)) + b + struct.pack(">I", 0))
        stdin_buf.seek(0)
        stdout_buf = io.BytesIO()

        with patch.object(sys, "stdin", new=type("S", (), {"buffer": stdin_buf})()):
            with patch.object(
                sys, "stdout", new=type("S", (), {"buffer": stdout_buf})()
            ):
                _worker_loop()

        stdout_buf.seek(0)
        (n,) = struct.unpack(">I", stdout_buf.read(4))
        response = cloudpickle.loads(stdout_buf.read(n))
        assert isinstance(response, dict)
        assert response["_error"] is True
        assert "ValueError" in response["stderr"]
        assert response["returncode"] == 1
        sys.modules.pop("user_code", None)
        linecache.cache.pop(_CODE_FILENAME, None)
