"""Downstream-consumer fixture tests for ``gigaevo.utils.text_sanitize``.

The unit tests in ``test_text_sanitize.py`` assert per-pattern strip /
preserve / escape behavior. This file asserts the next layer of contract:
that real-world hostile LLM error text, after passing through the
appropriate sanitizer, is accepted by every downstream surface the
project actually uses --- and that the *un*-sanitized form would fail.

Consumers exercised:

* ``json.dumps`` with both ``ensure_ascii`` settings, including the
  often-overlooked post-encode ``str.encode('utf-8')`` step that is
  where lone surrogates actually break.
* ``pydantic.BaseModel.model_dump_json`` on a small frozen model.
* ``pydantic.TypeAdapter(str).dump_python`` and ``dump_json``.
* ``text.encode('utf-8')`` --- the asyncpg / file-write path.
* ``loguru`` file-sink with a timestamp format, including line-forgery
  attempts.
* ``subprocess.run`` argv (NUL is rejected by the kernel's ``exec``).
* ``redis.asyncio`` / ``fakeredis`` with key and value containing
  protocol bytes (``\\r\\n``).
* ``sqlite3`` TEXT column round-trip (NUL handling differs from
  Postgres --- see notes inline).
* ``csv.writer`` round-trip after sanitization.
* Path-traversal residual risk in ``clean_identifier`` output.
* HTML / Markdown / SQL-LIKE / URL passthrough (documented non-goals).
* Composability, idempotence under iteration, JSON round-trip identity.
* Realistic provider-error string: ANSI from nvcc, CUTLASS template
  syntax, embedded NUL, Greek-letter Mojo identifier.
* Performance sanity (1 MB hostile input under 1 second per sanitizer).

Every test that exercises an "un-sanitized output would break" path uses
``pytest.raises`` to pin the exact failure mode; a regression that makes
that failure silently disappear (for instance, a future stdlib release
that starts silently replacing lone surrogates) would correctly fail
this test and prompt review rather than a silent semantic change.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from gigaevo.utils.text_sanitize import (
    clean_identifier,
    sanitize_for_dbtext,
    sanitize_for_json,
    sanitize_for_log,
)

# ---------------------------------------------------------------------------
# Shared hostile fixtures
# ---------------------------------------------------------------------------

# An astral lone high surrogate that breaks UTF-8 encoders and pydantic
# JSON serialization. Used in many tests.
LONE_HIGH = "\ud83d"
LONE_LOW = "\udc00"

# A minimal but realistic Triton MLIR diagnostic body, with ANSI from gcc
# colorization, embedded NUL (driver buffer artifact), CR (terminal width
# trick), C0 bytes, a BIDI override, and a lone surrogate.
HOSTILE_COMPILER_ERROR = (
    "\x1b[1m\x1b[31merror:\x1b[0m cannot convert value of type 'Layout<Shape<_32,_128>>'"
    "\n  at "
    "\x1b[33m/tmp/kernel.mlir:42:13\x1b[0m"
    " in template instantiation"
    "\nstack:\x00offset=0x1234"
    "\rtail-overwrite"
    f"\x07{LONE_HIGH} (lone surrogate)"
    "\n‮ direction-spoofed continuation"  # RLO
)

# A realistic OpenAI-2.x-style ``BadRequestError.message`` body simulating
# a remote tool-use call that returned an embedded compiler trace.
HOSTILE_PROVIDER_ERROR = (
    "tool 'compile_kernel' failed with:\n"
    "\x1b[1;31mnvcc fatal\x1b[0m   : Unsupported gpu architecture 'compute_120'\n"
    "  while expanding "
    "\x1b[36mcutlass::gemm::device::Gemm<cutlass::half_t, "
    "cutlass::layout::RowMajor, cutlass::half_t, cutlass::layout::ColumnMajor, "
    "float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm120>\x1b[0m\n"
    "  identifier: αβγ_residual_kernel\x00\n"
    "  return code: 1\r\n"
)


@pytest.fixture
def lone_surrogate_text() -> str:
    """A string Python permits but UTF-8 / JSON encoders reject."""
    return f"prefix{LONE_HIGH}suffix"


@pytest.fixture
def nul_text() -> str:
    return "before\x00after"


@pytest.fixture
def crlf_forgery_text() -> str:
    """Single CR-only forgery attempt (no LF)."""
    return "real-entry\rFAKE-overwrite"


# ---------------------------------------------------------------------------
# json.dumps --- ensure_ascii=True (default)
# ---------------------------------------------------------------------------


class TestJsonDumpsAscii:
    """``json.dumps`` with default ``ensure_ascii=True``.

    Important nuance: stdlib ``json.dumps(..., ensure_ascii=True)`` does
    NOT raise on lone surrogates --- it escapes them as ``\\uXXXX``
    literals. The actual breakage occurs only when the resulting str is
    UTF-8 encoded (which is what file / socket writers do).
    """

    def test_lone_surrogate_unsanitized_serializes_but_unencodable(
        self, lone_surrogate_text: str
    ) -> None:
        # The serializer itself succeeds...
        serialized = json.dumps(lone_surrogate_text)
        assert "\\ud83d" in serialized
        # ...but the result is unsendable: a lone ``\\uXXXX`` decodes back
        # to a lone surrogate on json.loads, which then refuses UTF-8.
        loaded = json.loads(serialized)
        with pytest.raises(UnicodeEncodeError):
            loaded.encode("utf-8")

    def test_lone_surrogate_sanitized_round_trips(
        self, lone_surrogate_text: str
    ) -> None:
        cleaned = sanitize_for_json(lone_surrogate_text)
        serialized = json.dumps(cleaned)
        loaded = json.loads(serialized)
        # Encoding must succeed end-to-end.
        loaded.encode("utf-8")
        assert "�" in loaded

    def test_nul_passes_through_json_ascii_escape(self, nul_text: str) -> None:
        # NUL is permitted in JSON strings (escapes to ``\\x00``). Round-
        # trip must preserve byte exactly. The sanitizer does NOT need to
        # touch NUL for the JSON destination.
        serialized = json.dumps(nul_text)
        assert "\\u0000" in serialized
        assert json.loads(serialized) == nul_text


# ---------------------------------------------------------------------------
# json.dumps --- ensure_ascii=False
# ---------------------------------------------------------------------------


class TestJsonDumpsNonAscii:
    """``ensure_ascii=False`` keeps multibyte text verbatim.

    NUL becomes the two-char literal ``\\u0000`` (safe to write). Lone
    surrogates remain as raw lone surrogates in the resulting Python str
    --- this is where UTF-8 encoding raises.
    """

    def test_lone_surrogate_unsanitized_encode_raises(
        self, lone_surrogate_text: str
    ) -> None:
        # json.dumps does NOT raise; it returns a str containing the lone
        # surrogate. The encoder rejects it.
        serialized = json.dumps(lone_surrogate_text, ensure_ascii=False)
        assert LONE_HIGH in serialized
        with pytest.raises(UnicodeEncodeError):
            serialized.encode("utf-8")

    def test_lone_surrogate_sanitized_encodes_cleanly(
        self, lone_surrogate_text: str
    ) -> None:
        cleaned = sanitize_for_json(lone_surrogate_text)
        json.dumps(cleaned, ensure_ascii=False).encode("utf-8")

    def test_nul_passes_through_safely(self, nul_text: str) -> None:
        # NUL becomes the literal 6-char escape ``\\u0000`` even with
        # ensure_ascii=False --- it is the one C0 byte JSON refuses raw.
        serialized = json.dumps(nul_text, ensure_ascii=False)
        assert "\\u0000" in serialized
        serialized.encode("utf-8")  # writable

    def test_compiler_error_with_log_sanitizer_then_json(self) -> None:
        cleaned = sanitize_for_log(HOSTILE_COMPILER_ERROR)
        serialized = json.dumps(cleaned, ensure_ascii=False)
        # End-to-end byte path must succeed.
        encoded = serialized.encode("utf-8")
        # ANSI bytes never reach the encoder.
        assert b"\x1b" not in encoded
        # Legitimate technical content survives.
        assert "Layout<Shape<_32,_128>>" in cleaned


# ---------------------------------------------------------------------------
# pydantic.BaseModel.model_dump_json
# ---------------------------------------------------------------------------


class TestPydanticBaseModel:
    """The most common production path: a pydantic v2 model with a single
    str field, serialized via ``model_dump_json``. Pydantic raises a
    ``PydanticSerializationError`` wrapping the underlying
    ``UnicodeEncodeError`` for lone surrogates.
    """

    def _model_cls(self) -> Any:
        from pydantic import BaseModel, ConfigDict

        class ErrMsg(BaseModel):
            model_config = ConfigDict(frozen=True)
            payload: str

        return ErrMsg

    def test_unsanitized_lone_surrogate_dump_raises(
        self, lone_surrogate_text: str
    ) -> None:
        from pydantic import ValidationError  # noqa: F401  (sanity import)
        from pydantic_core import PydanticSerializationError

        Model = self._model_cls()
        # Construction is fine --- the surrogate is a legal Python str char.
        m = Model(payload=lone_surrogate_text)
        with pytest.raises(PydanticSerializationError):
            m.model_dump_json()

    def test_sanitized_lone_surrogate_dump_ok(
        self, lone_surrogate_text: str
    ) -> None:
        Model = self._model_cls()
        m = Model(payload=sanitize_for_json(lone_surrogate_text))
        data = m.model_dump_json()
        assert json.loads(data)["payload"].endswith("suffix")

    def test_dump_json_round_trip_through_loads(self) -> None:
        Model = self._model_cls()
        cleaned = sanitize_for_log(HOSTILE_COMPILER_ERROR)
        m = Model(payload=cleaned)
        revived = Model.model_validate_json(m.model_dump_json())
        assert revived.payload == cleaned

    def test_nul_safe_through_pydantic(self, nul_text: str) -> None:
        # NUL travels through pydantic JSON serialization as ``\\x00``.
        # No sanitizer required for this single-byte case, BUT downstream
        # consumers like Postgres still reject NUL, so the project policy
        # is to sanitize at construction time. Verify both paths.
        Model = self._model_cls()
        Model(payload=nul_text).model_dump_json()
        Model(payload=sanitize_for_dbtext(nul_text)).model_dump_json()


# ---------------------------------------------------------------------------
# pydantic.TypeAdapter
# ---------------------------------------------------------------------------


class TestPydanticTypeAdapter:
    def test_dump_python_unsanitized_passes_str_through(
        self, lone_surrogate_text: str
    ) -> None:
        # dump_python returns the str unchanged --- pydantic does not
        # validate-encode for the Python target. This is correct stdlib
        # behavior but means the breakage only surfaces at dump_json /
        # network time.
        from pydantic import TypeAdapter

        ta = TypeAdapter(str)
        assert ta.dump_python(lone_surrogate_text) == lone_surrogate_text

    def test_dump_json_unsanitized_raises(self, lone_surrogate_text: str) -> None:
        from pydantic import TypeAdapter
        from pydantic_core import PydanticSerializationError

        ta = TypeAdapter(str)
        with pytest.raises(PydanticSerializationError):
            ta.dump_json(lone_surrogate_text)

    def test_dump_json_sanitized_ok(self, lone_surrogate_text: str) -> None:
        from pydantic import TypeAdapter

        ta = TypeAdapter(str)
        ta.dump_json(sanitize_for_json(lone_surrogate_text))

    def test_dump_python_sanitized_drops_surrogate(
        self, lone_surrogate_text: str
    ) -> None:
        from pydantic import TypeAdapter

        ta = TypeAdapter(str)
        cleaned = sanitize_for_json(lone_surrogate_text)
        assert LONE_HIGH not in ta.dump_python(cleaned)


# ---------------------------------------------------------------------------
# text.encode('utf-8') --- the asyncpg / file write path
# ---------------------------------------------------------------------------


class TestUtf8Encode:
    """asyncpg always UTF-8-encodes string values before sending them on
    the wire. The same Python encoder is what file writers (``open(path,
    'w').write(s)`` if the locale is UTF-8) use. Lone surrogates are the
    failure mode that ``sanitize_for_json`` / ``_log`` / ``_dbtext`` all
    fix; NUL is rejected by asyncpg at a higher layer (string-literal
    pre-check) and so is fixed by ``_dbtext`` and ``_log`` specifically.
    """

    def test_lone_surrogate_unsanitized_raises(
        self, lone_surrogate_text: str
    ) -> None:
        with pytest.raises(UnicodeEncodeError):
            lone_surrogate_text.encode("utf-8")

    def test_lone_surrogate_after_log_ok(self, lone_surrogate_text: str) -> None:
        sanitize_for_log(lone_surrogate_text).encode("utf-8")

    def test_lone_surrogate_after_json_ok(self, lone_surrogate_text: str) -> None:
        sanitize_for_json(lone_surrogate_text).encode("utf-8")

    def test_lone_surrogate_after_dbtext_ok(self, lone_surrogate_text: str) -> None:
        sanitize_for_dbtext(lone_surrogate_text).encode("utf-8")

    def test_nul_after_log_no_raw_nul(self, nul_text: str) -> None:
        # ``sanitize_for_log`` escapes NUL to the literal ``\\x00``,
        # which is safe to UTF-8-encode but no longer contains the actual
        # NUL byte. This is what makes the output safe for line-oriented
        # log sinks AND for Postgres TEXT columns simultaneously.
        cleaned = sanitize_for_log(nul_text)
        encoded = cleaned.encode("utf-8")
        assert b"\x00" not in encoded

    def test_compiler_error_after_log_encodes(self) -> None:
        sanitize_for_log(HOSTILE_COMPILER_ERROR).encode("utf-8")

    def test_compiler_error_after_dbtext_encodes(self) -> None:
        sanitize_for_dbtext(HOSTILE_COMPILER_ERROR).encode("utf-8")


# ---------------------------------------------------------------------------
# loguru file sink with line-forgery
# ---------------------------------------------------------------------------


class TestLoguruFileSink:
    """Loguru produces one record per ``logger.info`` call, but its file
    sink writes the formatted message verbatim --- so an LF inside the
    message produces a physical second line. The sanitizer documents
    this as a residual risk (``\\n`` is preserved so genuine tracebacks
    survive); a forged-line attempt that includes a fake timestamp can
    masquerade as a separate entry. CR-only forgeries are fully defused
    because ``\\r`` is escaped to ``\\x0d``.
    """

    def _setup_sink(self, fmt: str = "{time:YYYY-MM-DD HH:mm:ss} {level} {message}"):
        from loguru import logger

        logger.remove()
        tf = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".log", encoding="utf-8"
        )
        tf.close()
        sink_path = Path(tf.name)
        sink_id = logger.add(sink_path, format=fmt, level="INFO")
        return logger, sink_path, sink_id

    def _teardown(self, logger: Any, sink_path: Path, sink_id: int) -> None:
        logger.remove(sink_id)
        if sink_path.exists():
            sink_path.unlink()

    def test_unsanitized_lf_forgery_produces_extra_line(self) -> None:
        # Documented residual risk: a hostile string with ``\n`` plus a
        # plausible timestamp prefix shows up in the sink as if it were
        # a real log entry. The mitigation is a higher-level invariant:
        # genuine entries always begin with the configured timestamp
        # prefix that the application's formatter emits, which the
        # forged line cannot match exactly without server-side knowledge.
        logger, path, sid = self._setup_sink()
        try:
            cleaned = sanitize_for_log("legit\n2026-01-01 00:00:00 INFO FORGED")
            logger.info(cleaned)
            content = path.read_text(encoding="utf-8")
        finally:
            self._teardown(logger, path, sid)
        # We document the residual risk by asserting it: two lines, one
        # of which looks like a forged entry.
        lines = content.rstrip("\n").splitlines()
        assert len(lines) == 2, "LF in message produces physical second line"
        # Useful diagnostic: the second line is the forgery.
        assert "FORGED" in lines[1]

    def test_cr_only_forgery_is_defused(self, crlf_forgery_text: str) -> None:
        logger, path, sid = self._setup_sink()
        try:
            cleaned = sanitize_for_log(crlf_forgery_text)
            logger.info(cleaned)
            content = path.read_text(encoding="utf-8")
        finally:
            self._teardown(logger, path, sid)
        # CR was escaped, so the single logger.info call still produces
        # one and only one physical line in the file.
        lines = content.rstrip("\n").splitlines()
        assert len(lines) == 1
        assert "\\x0d" in lines[0]
        assert "FAKE-overwrite" in lines[0]  # visible but inert

    def test_ansi_alone_does_not_forge(self) -> None:
        logger, path, sid = self._setup_sink()
        try:
            cleaned = sanitize_for_log("a\x1b[31mb\x1b[0mc")
            logger.info(cleaned)
            content = path.read_text(encoding="utf-8")
        finally:
            self._teardown(logger, path, sid)
        lines = content.rstrip("\n").splitlines()
        assert len(lines) == 1
        # ANSI bytes do not reach the sink.
        assert "\x1b" not in content
        assert "abc" in lines[0]

    def test_lone_surrogate_in_message_does_not_crash_sink(
        self, lone_surrogate_text: str
    ) -> None:
        logger, path, sid = self._setup_sink()
        try:
            logger.info(sanitize_for_log(lone_surrogate_text))
            content = path.read_text(encoding="utf-8")
        finally:
            self._teardown(logger, path, sid)
        assert "prefix" in content and "suffix" in content
        assert LONE_HIGH not in content  # replaced with U+FFFD

    def test_unsanitized_lone_surrogate_would_break_sink(
        self, lone_surrogate_text: str
    ) -> None:
        # Loguru queues writes; we trigger the failure path by directly
        # exercising the file write the sink would otherwise perform.
        # This is the equivalent of "what happens when the sink flushes".
        with pytest.raises(UnicodeEncodeError):
            (lone_surrogate_text + "\n").encode("utf-8")


# ---------------------------------------------------------------------------
# subprocess argv
# ---------------------------------------------------------------------------


class TestSubprocessArgv:
    """The kernel's ``execve`` accepts argv via a NUL-terminated C array,
    so Python pre-validates: any argv element containing NUL raises
    ``ValueError`` before fork. ``clean_identifier`` is the right
    sanitizer for argv because it produces strict-ASCII output with no
    control bytes whatsoever.
    """

    def test_unsanitized_nul_in_argv_rejected(self) -> None:
        with pytest.raises(ValueError, match="null"):
            subprocess.run(
                [sys.executable, "-c", "pass", "model\x00admin"],
                capture_output=True,
                check=False,
                timeout=10,
            )

    def test_clean_identifier_output_is_exec_safe(self) -> None:
        ident = clean_identifier("model\x00admin\nworker\x07")
        # Result is the conservative identifier charset only.
        assert "\x00" not in ident
        assert "\n" not in ident
        assert "\x07" not in ident
        r = subprocess.run(
            [sys.executable, "-c", "import sys; print(sys.argv[1])", ident],
            capture_output=True,
            check=True,
            timeout=10,
            text=True,
        )
        assert r.stdout.strip() == ident

    def test_subprocess_stdin_accepts_nul_bytes(self) -> None:
        # Pipes accept NUL --- the kernel rejection is argv-specific.
        # This guards against a future "over-sanitize stdin" mistake.
        r = subprocess.run(
            [sys.executable, "-c", "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())"],
            input=b"a\x00b",
            capture_output=True,
            check=True,
            timeout=10,
        )
        assert r.stdout == b"a\x00b"


# ---------------------------------------------------------------------------
# Redis RESP protocol
# ---------------------------------------------------------------------------


class TestRedisProtocol:
    """RESP delimits frames with ``\\r\\n`` and length-prefixes bulk
    strings, so modern clients are not vulnerable to in-band injection.
    We use fakeredis as a behavioral oracle to confirm that keys and
    values containing CRLF round-trip cleanly --- the sanitizer is not
    required here, but applying it to keys avoids weird debug output.
    """

    def test_crlf_in_key_round_trips(self) -> None:
        import fakeredis

        r = fakeredis.FakeRedis()
        try:
            r.set("a\r\nFLUSHDB", b"value")
            assert r.get("a\r\nFLUSHDB") == b"value"
        finally:
            r.close()

    def test_crlf_in_value_round_trips(self) -> None:
        import fakeredis

        r = fakeredis.FakeRedis()
        try:
            forged = b"val\r\n*1\r\n$8\r\nFLUSHALL\r\n"
            r.set("k", forged)
            assert r.get("k") == forged
        finally:
            r.close()

    def test_sanitized_key_works(self) -> None:
        import fakeredis

        r = fakeredis.FakeRedis()
        try:
            key = clean_identifier("user:42@host\r\nproblem")
            r.set(key, b"v")
            assert r.get(key) == b"v"
        finally:
            r.close()


# ---------------------------------------------------------------------------
# SQLite TEXT column round-trip
# ---------------------------------------------------------------------------


class TestSqliteTextColumn:
    """SQLite's TEXT affinity stores arbitrary bytes including NUL ---
    unlike Postgres' TEXT, which rejects NUL at the wire layer. This is
    a place where the dbtext sanitizer's NUL-replacement policy is
    motivated by the *Postgres* contract, not SQLite's. We still verify
    that sanitized output round-trips cleanly through SQLite.
    """

    @pytest.fixture
    def conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(":memory:")
        c.execute("CREATE TABLE t (v TEXT NOT NULL)")
        yield c
        c.close()

    def test_sqlite_does_not_truncate_at_nul(
        self, conn: sqlite3.Connection, nul_text: str
    ) -> None:
        # Documented behavior: SQLite stores NUL byte intact. This is
        # different from Postgres TEXT and is the reason sanitize_for_dbtext
        # is needed only on the Postgres path. We pin the behavior so a
        # future SQLite Python binding change does not silently affect
        # callers who relied on round-trip.
        conn.execute("INSERT INTO t VALUES (?)", (nul_text,))
        (got,) = conn.execute("SELECT v FROM t").fetchone()
        assert got == nul_text  # full string preserved

    def test_dbtext_sanitized_round_trips(self, conn: sqlite3.Connection) -> None:
        cleaned = sanitize_for_dbtext(f"a\x00b{LONE_HIGH}c")
        conn.execute("INSERT INTO t VALUES (?)", (cleaned,))
        (got,) = conn.execute("SELECT v FROM t").fetchone()
        assert got == cleaned
        # And the round-tripped value is still UTF-8-encodable for the
        # consumer that will eventually read this row.
        got.encode("utf-8")

    def test_log_sanitized_round_trips(self, conn: sqlite3.Connection) -> None:
        cleaned = sanitize_for_log(HOSTILE_COMPILER_ERROR)
        conn.execute("INSERT INTO t VALUES (?)", (cleaned,))
        (got,) = conn.execute("SELECT v FROM t").fetchone()
        assert got == cleaned

    def test_unsanitized_lone_surrogate_breaks_text_factory(
        self, conn: sqlite3.Connection
    ) -> None:
        # If the connection text_factory is the default ``str``, the C
        # layer stores the lone-surrogate-bearing str by encoding it as
        # UTF-8 internally; this raises at insert time, not at fetch.
        # We assert the failure path so a regression of the dbtext
        # sanitizer's surrogate handling is caught here.
        with pytest.raises((UnicodeEncodeError, sqlite3.ProgrammingError)):
            conn.execute("INSERT INTO t VALUES (?)", (f"x{LONE_HIGH}y",))


# ---------------------------------------------------------------------------
# CSV writer round-trip
# ---------------------------------------------------------------------------


class TestCsvWriter:
    def test_hostile_row_round_trips_after_log_sanitize(self) -> None:
        cleaned = sanitize_for_log(HOSTILE_COMPILER_ERROR)
        out = io.StringIO()
        csv.writer(out).writerow(["id-1", cleaned, "tail"])
        # Read back; csv.reader correctly handles embedded newlines from
        # quoted fields. ANSI is already stripped, so quote rules apply
        # cleanly.
        rows = list(csv.reader(io.StringIO(out.getvalue())))
        assert len(rows) == 1
        assert rows[0][0] == "id-1"
        assert rows[0][1] == cleaned
        assert rows[0][2] == "tail"

    def test_unsanitized_lone_surrogate_breaks_utf8_write(
        self, lone_surrogate_text: str
    ) -> None:
        out = io.StringIO()
        csv.writer(out).writerow([lone_surrogate_text])
        with pytest.raises(UnicodeEncodeError):
            out.getvalue().encode("utf-8")

    def test_sanitized_writes_to_real_file(self, tmp_path: Path) -> None:
        cleaned = sanitize_for_log(HOSTILE_COMPILER_ERROR)
        target = tmp_path / "out.csv"
        with target.open("w", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerow(["k", cleaned])
        # File write must not raise.
        assert target.read_text(encoding="utf-8").count("\n") >= 1

    def test_commas_and_quotes_in_message_handled(self) -> None:
        msg = sanitize_for_log('value with, comma and "quote" inside')
        out = io.StringIO()
        csv.writer(out).writerow([msg])
        rows = list(csv.reader(io.StringIO(out.getvalue())))
        assert rows[0][0] == msg


# ---------------------------------------------------------------------------
# clean_identifier path-traversal residual risk
# ---------------------------------------------------------------------------


class TestCleanIdentifierPathTraversal:
    """``clean_identifier`` keeps ``/`` because the project genuinely
    uses path-style model identifiers (``meta-llama/Llama-3.1-70B``).
    A side-effect is that ``..`` and trailing slashes also survive, so
    callers that pipe the result to filesystem APIs MUST normalize /
    confine before joining. These tests pin the residual risk.
    """

    def test_dot_dot_survives(self) -> None:
        assert ".." in clean_identifier("../../etc/passwd")

    def test_full_path_traversal_string_survives(self) -> None:
        assert clean_identifier("../../etc/passwd") == "../../etc/passwd"

    def test_slash_survives(self) -> None:
        assert "/" in clean_identifier("a/b/c")

    def test_null_byte_stripped_from_path(self) -> None:
        assert "\x00" not in clean_identifier("foo\x00../bar")

    def test_filesystem_join_is_callers_problem(self, tmp_path: Path) -> None:
        # Demonstrate the trap: if a caller blindly does
        # ``tmp_path / clean_identifier(user_input)`` the resolved path
        # can escape ``tmp_path``. Callers must use Path.resolve() +
        # is_relative_to() or similar.
        evil = clean_identifier("../../etc/passwd")
        joined = (tmp_path / evil).resolve()
        # The resolved path is OUTSIDE tmp_path --- this is the failure
        # mode callers must guard against. We assert it to document.
        assert not str(joined).startswith(str(tmp_path.resolve()))


# ---------------------------------------------------------------------------
# Pass-through hazards (documented non-goals)
# ---------------------------------------------------------------------------


class TestPassThroughHazards:
    """The sanitizers deliberately do NOT escape HTML / Markdown / SQL
    LIKE wildcards / URL-unsafe characters. These tests pin the policy.
    """

    def test_html_special_chars_pass_through_log(self) -> None:
        src = "<script>alert(1)</script> & &amp;"
        assert sanitize_for_log(src) == src

    def test_html_consumer_must_escape_itself(self) -> None:
        import html

        cleaned = sanitize_for_log("<script>alert(1)</script>")
        # The HTML consumer adds its own layer.
        assert "&lt;" in html.escape(cleaned)

    def test_markdown_special_chars_pass_through_log(self) -> None:
        src = "`backtick` *italic* **bold** [link](url)"
        assert sanitize_for_log(src) == src

    def test_sql_like_wildcards_survive_clean_identifier(self) -> None:
        # ``%`` and ``_`` --- ``_`` is in the identifier charset, ``%``
        # is not. Both are non-issues at this layer; SQL LIKE callers
        # must escape themselves.
        assert "_" in clean_identifier("a_b")
        assert "%" not in clean_identifier("a%b")

    def test_url_unsafe_chars_largely_absent_from_clean_identifier(self) -> None:
        # Side benefit of the conservative charset: ``?``, ``#``, ``&``
        # are absent. ``/``, ``:``, ``@``, ``.`` survive (legitimate in
        # path / scheme / userinfo positions).
        ident = clean_identifier("a?b#c&d/e:f@g.h")
        for unsafe in "?#&":
            assert unsafe not in ident


# ---------------------------------------------------------------------------
# Composability
# ---------------------------------------------------------------------------


class TestComposability:
    """Order of application matters for output bytes (an earlier escape
    can hide a NUL behind ``\\x00``) but never for safety: regardless of
    order, the final output is safe for every consumer."""

    SRC = f"a\x00b{LONE_HIGH}c\x1b[2Jd"

    def test_log_then_dbtext_safe(self) -> None:
        out = sanitize_for_dbtext(sanitize_for_log(self.SRC))
        out.encode("utf-8")
        json.dumps(out)

    def test_dbtext_then_log_safe(self) -> None:
        out = sanitize_for_log(sanitize_for_dbtext(self.SRC))
        out.encode("utf-8")
        json.dumps(out)

    def test_orders_differ_in_form_but_both_safe(self) -> None:
        a = sanitize_for_log(sanitize_for_dbtext(self.SRC))
        b = sanitize_for_dbtext(sanitize_for_log(self.SRC))
        # Inner-dbtext converts NUL to U+FFFD, so outer-log sees no NUL
        # to escape: ``\\x00`` literal is absent in (a).
        assert "\\x00" not in a
        # Outer-dbtext sees ``\\x00`` (literal backslash-x-zero-zero) and
        # leaves it alone --- no real NUL to replace.
        assert "\\x00" in b
        # Both are safe at every downstream target.
        for v in (a, b):
            v.encode("utf-8")
            json.dumps(v)


# ---------------------------------------------------------------------------
# Idempotence under iteration
# ---------------------------------------------------------------------------


class TestIdempotenceStress:
    def test_log_idempotent_100_times(self) -> None:
        once = sanitize_for_log(HOSTILE_COMPILER_ERROR)
        v = once
        for _ in range(100):
            v = sanitize_for_log(v)
        assert v == once

    def test_dbtext_idempotent_100_times(self) -> None:
        once = sanitize_for_dbtext(HOSTILE_COMPILER_ERROR)
        v = once
        for _ in range(100):
            v = sanitize_for_dbtext(v)
        assert v == once

    def test_json_idempotent_100_times(self) -> None:
        once = sanitize_for_json(HOSTILE_COMPILER_ERROR)
        v = once
        for _ in range(100):
            v = sanitize_for_json(v)
        assert v == once

    def test_identifier_idempotent_100_times(self) -> None:
        once = clean_identifier("model\x00 with\n stuff αβγ")
        v = once
        for _ in range(100):
            v = clean_identifier(v)
        assert v == once


# ---------------------------------------------------------------------------
# JSON round-trip identity from second pass onwards
# ---------------------------------------------------------------------------


class TestJsonRoundTripIdentity:
    def test_sanitize_dump_load_sanitize_is_identity(self) -> None:
        once = sanitize_for_json(HOSTILE_COMPILER_ERROR)
        revived = json.loads(json.dumps(once))
        twice = sanitize_for_json(revived)
        assert twice == once

    def test_log_sanitize_dump_load_log_sanitize_is_identity(self) -> None:
        once = sanitize_for_log(HOSTILE_COMPILER_ERROR)
        revived = json.loads(json.dumps(once))
        twice = sanitize_for_log(revived)
        assert twice == once

    def test_pydantic_round_trip_identity(self) -> None:
        from pydantic import BaseModel

        class M(BaseModel):
            payload: str

        once = sanitize_for_log(HOSTILE_COMPILER_ERROR)
        m1 = M(payload=once)
        m2 = M.model_validate_json(m1.model_dump_json())
        assert m2.payload == once


# ---------------------------------------------------------------------------
# Realistic provider-error fixture
# ---------------------------------------------------------------------------


class TestRealisticProviderError:
    def test_provider_error_through_log_then_json(self) -> None:
        cleaned = sanitize_for_log(HOSTILE_PROVIDER_ERROR)
        # JSON serialization must succeed end-to-end.
        json.dumps(cleaned).encode("utf-8")
        # ANSI bytes do not survive.
        assert "\x1b" not in cleaned
        # NUL is escaped to literal.
        assert "\x00" not in cleaned
        # Greek identifier survives (Mojo-style identifier preservation).
        assert "αβγ_residual_kernel" in cleaned
        # CUTLASS template syntax survives.
        assert "cutlass::gemm::device::Gemm" in cleaned
        # The compiler diagnostic text itself survives.
        assert "Unsupported gpu architecture" in cleaned

    def test_provider_error_through_dbtext_then_log(self) -> None:
        # Composition order used in production: dbtext first (for the
        # DB write), log on the way to display.
        once = sanitize_for_dbtext(HOSTILE_PROVIDER_ERROR)
        once.encode("utf-8")
        twice = sanitize_for_log(once)
        twice.encode("utf-8")
        # CR was preserved by dbtext, escaped by log.
        assert "\r" not in twice
        assert "\\x0d" in twice

    def test_provider_error_into_pydantic(self) -> None:
        from pydantic import BaseModel

        class ProviderErr(BaseModel):
            message: str

        m = ProviderErr(message=sanitize_for_log(HOSTILE_PROVIDER_ERROR))
        # Full pydantic JSON serialization path must succeed.
        revived = ProviderErr.model_validate_json(m.model_dump_json())
        assert "αβγ_residual_kernel" in revived.message

    def test_provider_error_clean_identifier_extraction(self) -> None:
        # A common project pattern: extract a model name from a free-form
        # provider error. clean_identifier on a slice must survive even
        # if the slice contains hostile bytes.
        slice_ = HOSTILE_PROVIDER_ERROR.split("identifier: ", 1)[1].split("\n")[0]
        ident = clean_identifier(slice_)
        # ASCII-only, no controls, no NUL.
        assert ident.isascii()
        assert "\x00" not in ident


# ---------------------------------------------------------------------------
# Performance sanity --- 1 MB input under 1 second
# ---------------------------------------------------------------------------


class TestPerformanceSanity:
    """Pure regex throughput sanity. The sanitizers are linear in input
    length; 1 MB through any of them should complete well under 1 s on
    typical CI hardware. A regression that introduced quadratic
    behavior (for instance, repeated full-string passes inside a loop)
    would fail this guard."""

    @pytest.fixture(scope="class")
    def one_mb_hostile(self) -> str:
        # Repeat a 1 KiB hostile block 1024 times. Includes ANSI, NUL,
        # BIDI, surrogate, C1, CR.
        chunk = (
            "\x1b[31mErr\x1b[0m \x00 ‮ \ud83d \x9b \r tail "
            + ("padding " * 100)
        )[:1024]
        return chunk * 1024  # ~1 MiB

    def test_sanitize_for_log_under_one_second(self, one_mb_hostile: str) -> None:
        t0 = time.perf_counter()
        sanitize_for_log(one_mb_hostile)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"sanitize_for_log too slow: {elapsed:.3f}s"

    def test_sanitize_for_json_under_one_second(self, one_mb_hostile: str) -> None:
        t0 = time.perf_counter()
        sanitize_for_json(one_mb_hostile)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"sanitize_for_json too slow: {elapsed:.3f}s"

    def test_sanitize_for_dbtext_under_one_second(self, one_mb_hostile: str) -> None:
        t0 = time.perf_counter()
        sanitize_for_dbtext(one_mb_hostile)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"sanitize_for_dbtext too slow: {elapsed:.3f}s"

    def test_clean_identifier_under_one_second(self, one_mb_hostile: str) -> None:
        t0 = time.perf_counter()
        clean_identifier(one_mb_hostile)
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"clean_identifier too slow: {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# Final invariant: every sanitizer's output is byte-encodable
# ---------------------------------------------------------------------------


class TestUniversalEncodability:
    """The single most important contract: regardless of input, every
    sanitizer's primary output (the function that targets a specific
    destination) must produce a string that destination accepts.
    """

    INPUTS = [
        "",
        "plain ASCII",
        "αβγ Greek",
        "配置错误",
        "a\x00b",
        f"a{LONE_HIGH}b",
        f"a{LONE_LOW}b",
        "a\x1b[2Jb",
        "a‮b",
        HOSTILE_COMPILER_ERROR,
        HOSTILE_PROVIDER_ERROR,
        # Maximum-entropy mix.
        "".join(chr(i) for i in range(0, 0x80)),
    ]

    @pytest.mark.parametrize("src", INPUTS)
    def test_log_output_utf8_encodable(self, src: str) -> None:
        sanitize_for_log(src).encode("utf-8")

    @pytest.mark.parametrize("src", INPUTS)
    def test_json_output_json_dumpable_and_loadable(self, src: str) -> None:
        cleaned = sanitize_for_json(src)
        # ensure_ascii=False is the strict path; default also works but
        # the strict path is what surfaces lone-surrogate bugs.
        encoded = json.dumps(cleaned, ensure_ascii=False).encode("utf-8")
        assert json.loads(encoded.decode("utf-8")) == cleaned

    @pytest.mark.parametrize("src", INPUTS)
    def test_dbtext_output_utf8_encodable_and_nul_free(self, src: str) -> None:
        cleaned = sanitize_for_dbtext(src)
        assert "\x00" not in cleaned
        cleaned.encode("utf-8")

    @pytest.mark.parametrize("src", INPUTS)
    def test_identifier_output_ascii_and_no_controls(self, src: str) -> None:
        ident = clean_identifier(src)
        # Strict ASCII subset only.
        assert ident.isascii()
        # No control bytes.
        for ch in ident:
            assert ch.isprintable() or ch == ""
