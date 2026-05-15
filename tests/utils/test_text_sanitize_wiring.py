"""Integration tests proving the sanitizer is wired into call sites.

Each test feeds a hostile string (ANSI escape, NUL, lone surrogate, BIDI
override, CR carriage-return) through a real call site, captures the
resulting log line or stored value via loguru/fakeredis, and asserts that
the destination saw a sanitized form rather than the raw hostile bytes.

The tests deliberately use the production logging path (loguru sink) and
production fakeredis (already a dev dep) so that a regression which silently
removes a ``sanitize_for_log`` wrap will surface as a hostile byte
re-appearing in the captured output.
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Any

import fakeredis
from loguru import logger
import pytest


# Shared hostile fixtures
LONE_HIGH = "\ud83d"
HOSTILE = (
    "\x1b[31merr\x1b[0m"  # ANSI red
    "\x00NUL"  # NUL
    "\rCR"  # CR forgery
    "\x07BEL"  # bell
    f"{LONE_HIGH}LS"  # lone surrogate
    "‮RLO"  # RLO BIDI override
)
# A variant without a lone surrogate, for paths that flow through pydantic
# string fields whose validators reject lone surrogates upstream of our
# sanitizer wrapping (e.g. ``Program.code``).
HOSTILE_NO_SURROGATE = (
    "\x1b[31merr\x1b[0m"  # ANSI red
    "\x00NUL"  # NUL
    "\rCR"  # CR forgery
    "\x07BEL"  # bell
    "‮RLO"  # RLO BIDI override
)


def _assert_no_raw_hostile_non_surrogate(captured: str) -> None:
    """Variant for paths where the lone-surrogate has already been filtered
    upstream by a pydantic validator before reaching the sanitizer.
    """
    assert "\x1b" not in captured, "raw ANSI ESC survived"
    assert "\x00" not in captured, "raw NUL survived"
    assert "\x07" not in captured, "raw BEL survived"
    assert "‮" not in captured, "BIDI RLO survived"
    captured.encode("utf-8")


@pytest.fixture
def loguru_sink():
    """Add a string-buffer loguru sink, yield (buffer, sink_id), tear down."""
    buf = io.StringIO()
    sink_id = logger.add(buf, format="{message}", level="DEBUG")
    yield buf
    logger.remove(sink_id)


def _assert_sanitized(captured: str) -> None:
    """Assert no raw hostile bytes survived; sanitized escapes are OK."""
    assert "\x1b" not in captured, "raw ANSI ESC survived"
    assert "\x00" not in captured, "raw NUL survived"
    assert "\x07" not in captured, "raw BEL survived"
    assert LONE_HIGH not in captured, "lone surrogate survived"
    assert "‮" not in captured, "BIDI RLO survived"
    # Captured string must encode cleanly as UTF-8 (the loguru sink path).
    captured.encode("utf-8")


# ---------------------------------------------------------------------------
# gigaevo/database/redis_program_storage.py
# ---------------------------------------------------------------------------


class TestRedisProgramStorageCorruptDataLog:
    def test_corrupt_data_log_sanitized(self, loguru_sink) -> None:
        from gigaevo.database.redis_program_storage import RedisProgramStorage

        # Trigger _safe_deserialize with a corrupt JSON blob whose error
        # message includes hostile bytes. We construct an exception by
        # raising one with a hostile message, which from_dict will surface.
        bad_blob = '{"id": "x", "code": "' + HOSTILE + '"}'
        # The exception message produced by Program.from_dict will mention
        # missing fields; the JSON value itself isn't echoed back, so we
        # also exercise the path where the error str contains hostile bytes.
        # Simplest: monkeypatch Program.from_dict to raise with hostile msg.
        from gigaevo.programs import program as program_mod

        original = program_mod.Program.from_dict

        def boom(_data: Any, *, exclude: Any = None) -> None:  # noqa: ANN401
            raise ValueError(f"parse failed: {HOSTILE}")

        program_mod.Program.from_dict = staticmethod(boom)  # type: ignore[assignment]
        try:
            result = RedisProgramStorage._safe_deserialize(bad_blob, ctx="test")
        finally:
            program_mod.Program.from_dict = original  # type: ignore[assignment]

        assert result is None
        captured = loguru_sink.getvalue()
        assert "[RedisProgramStorage] Corrupt data in test:" in captured
        _assert_sanitized(captured)


# ---------------------------------------------------------------------------
# gigaevo/database/state_manager.py
# ---------------------------------------------------------------------------


class TestStateManagerInvalidTransitionLog:
    async def test_invalid_state_transition_log_sanitized(
        self, state_manager, make_program, loguru_sink, monkeypatch
    ) -> None:
        from gigaevo.programs import program_state as ps_mod

        prog = make_program()
        await state_manager.storage.add(prog)

        def boom(_old: Any, _new: Any) -> None:  # noqa: ANN401
            raise ValueError(f"bad transition: {HOSTILE}")

        monkeypatch.setattr(ps_mod, "validate_transition", boom)
        # Re-import inside state_manager too (it imports by name).
        from gigaevo.database import state_manager as sm_mod

        monkeypatch.setattr(sm_mod, "validate_transition", boom)

        with pytest.raises(ValueError):
            await state_manager.set_program_state(prog, ps_mod.ProgramState.DONE)

        captured = loguru_sink.getvalue()
        assert "Invalid state transition for" in captured
        _assert_sanitized(captured)


# ---------------------------------------------------------------------------
# gigaevo/evolution/mutation/mutation_operator.py
# ---------------------------------------------------------------------------


class TestMutationOperatorCanonicalizeLog:
    def test_syntax_error_log_sanitized(self, loguru_sink, monkeypatch) -> None:
        from gigaevo.evolution.mutation import mutation_operator as mop

        # Force ast.parse to raise a SyntaxError whose msg has hostile bytes.
        def boom(_src: str) -> None:
            raise SyntaxError(f"bad syntax: {HOSTILE}")

        monkeypatch.setattr(mop.ast, "parse", boom)
        out = mop.LLMMutationOperator._canonicalize_code("x = 1")
        # Falls back to original code on failure.
        assert out == "x = 1"
        captured = loguru_sink.getvalue()
        assert "Failed to canonicalize code" in captured
        _assert_sanitized(captured)


# ---------------------------------------------------------------------------
# gigaevo/prompts/coevolution/stages.py
# ---------------------------------------------------------------------------


class TestPromptExecutionStageErrorMessages:
    """The ValueError messages produced by PromptExecutionStage carry the
    LLM-derived code snippet or exception text. The sanitizer is applied
    before f-string interpolation so the exception args themselves are
    UTF-8-encodable and do not break downstream loguru / asyncpg writers
    that surface ``str(exc)``.
    """

    async def test_non_python_content_error_sanitized(self) -> None:
        from gigaevo.prompts.coevolution.stages import PromptExecutionStage

        stage = PromptExecutionStage()
        # No "def entrypoint" so the first branch fires. Use a hostile
        # string that does NOT include a lone UTF-16 surrogate — those are
        # already rejected by Program.code's pydantic validator upstream,
        # so this is the realistic threat surface for stages.py.
        bad_code = HOSTILE_NO_SURROGATE + "garbage content here that exceeds 80 chars" * 3
        from gigaevo.programs.program import Program

        prog = Program(code=bad_code)
        with pytest.raises(ValueError) as ei:
            await stage.compute(prog)
        msg = str(ei.value)
        _assert_no_raw_hostile_non_surrogate(msg)
        # The marker phrase survives.
        assert "Code starts with" in msg

    async def test_syntax_error_message_sanitized(self, monkeypatch) -> None:
        from gigaevo.prompts.coevolution import stages as st

        stage = st.PromptExecutionStage()
        # Genuinely malformed Python whose SyntaxError str() will not
        # already contain hostile bytes; we then monkeypatch the
        # ``compile`` builtin in the module's globals so the str(exc) we
        # interpolate carries the hostile payload.
        code = "def entrypoint():\n    pass\n"
        from gigaevo.programs.program import Program

        prog = Program(code=code)

        # Patch the builtin module's compile via stages.__builtins__.
        import builtins

        original = builtins.compile

        def boom_compile(*_a: Any, **_k: Any) -> None:  # noqa: ANN401
            raise SyntaxError(f"bad: {HOSTILE_NO_SURROGATE}")

        monkeypatch.setattr(builtins, "compile", boom_compile)
        try:
            with pytest.raises(ValueError) as ei:
                await stage.compute(prog)
            _assert_no_raw_hostile_non_surrogate(str(ei.value))
        finally:
            monkeypatch.setattr(builtins, "compile", original)

    async def test_entrypoint_exception_message_sanitized(self) -> None:
        from gigaevo.prompts.coevolution.stages import PromptExecutionStage
        from gigaevo.programs.program import Program

        # entrypoint() raises with hostile bytes in the exception message.
        # No lone surrogate because Program.code's validator rejects it.
        code = (
            "def entrypoint():\n"
            f"    raise RuntimeError({HOSTILE_NO_SURROGATE!r})\n"
        )
        prog = Program(code=code)
        stage = PromptExecutionStage()
        with pytest.raises(ValueError) as ei:
            await stage.compute(prog)
        _assert_no_raw_hostile_non_surrogate(str(ei.value))


# ---------------------------------------------------------------------------
# gigaevo/prompts/coevolution/stats.py
# ---------------------------------------------------------------------------


class TestRedisPromptStatsProviderErrorLog:
    async def test_redis_error_log_sanitized(self, loguru_sink, monkeypatch) -> None:
        from gigaevo.prompts.coevolution import stats as st

        provider = st.RedisPromptStatsProvider(
            host="localhost", port=6379, db=0, prefix="test"
        )

        # Make the Redis client GET raise with hostile error text.
        class BoomRedis:
            async def get(self, _key: str) -> None:
                raise RuntimeError(f"redis error: {HOSTILE}")

        monkeypatch.setattr(provider, "_get_redis", lambda _db: BoomRedis())

        # Prompt ID with hostile bytes too (rare but possible in tests).
        result = await provider.get_stats(prompt_id=f"pid-{HOSTILE}")
        # No data found -> default zero stats.
        assert result.trials == 0
        captured = loguru_sink.getvalue()
        assert "Error reading stats from" in captured
        _assert_sanitized(captured)


# ---------------------------------------------------------------------------
# gigaevo/prompts/fetcher.py
# ---------------------------------------------------------------------------


class TestGigaEvoArchivePromptFetcherLogs:
    def _make_fetcher(self, tmp_path: Path) -> Any:
        from gigaevo.prompts.fetcher import GigaEvoArchivePromptFetcher

        return GigaEvoArchivePromptFetcher(
            prompt_redis_db=0,
            main_redis_prefix="main",
            main_redis_db=None,
            fallback_prompts_dir=tmp_path,
        )

    def test_archive_read_error_log_sanitized(
        self, tmp_path: Path, loguru_sink, monkeypatch
    ) -> None:
        fetcher = self._make_fetcher(tmp_path)

        class BoomRedis:
            def hvals(self, _k: str) -> None:
                raise RuntimeError(f"hvals failed: {HOSTILE}")

        monkeypatch.setattr(fetcher, "_get_sync_redis", lambda: BoomRedis())
        result = fetcher._refresh_candidates()
        assert result is None
        captured = loguru_sink.getvalue()
        assert "Archive read error" in captured
        _assert_sanitized(captured)

    def test_entrypoint_execution_error_log_sanitized(
        self, tmp_path: Path, loguru_sink
    ) -> None:
        fetcher = self._make_fetcher(tmp_path)
        # Code whose entrypoint() raises with hostile message.
        code = f"def entrypoint():\n    raise RuntimeError({HOSTILE!r})\n"
        out = fetcher._execute_entrypoint(code)
        assert out is None
        captured = loguru_sink.getvalue()
        assert "entrypoint() execution error" in captured
        _assert_sanitized(captured)

    def test_sampled_prompt_preview_sanitized(
        self, tmp_path: Path, loguru_sink
    ) -> None:
        fetcher = self._make_fetcher(tmp_path)
        # Inject candidates list directly so _sample_prompt fires the
        # preview log line containing system[:300] from the LLM code output.
        # The system text must be UTF-8-encodable so exec() can run the
        # Python source; no lone surrogates here, but ANSI/BIDI/NUL all
        # exercise the sanitizer.
        hostile_prompt = HOSTILE_NO_SURROGATE + " legitimate prompt body content"
        code = (
            "def entrypoint():\n"
            f"    return {hostile_prompt!r}\n"
        )
        fetcher._cached_candidates = [("pid-abc12345", 0.7, code)]
        pack = fetcher._sample_prompt()
        assert pack is not None
        captured = loguru_sink.getvalue()
        assert "Sampled:" in captured
        _assert_no_raw_hostile_non_surrogate(captured)

    def test_parse_program_error_log_sanitized(
        self, tmp_path: Path, loguru_sink, monkeypatch
    ) -> None:
        fetcher = self._make_fetcher(tmp_path)

        class FakeRedis:
            def hvals(self, _k: str) -> list[str]:
                return [f"prog-{HOSTILE}"]

            def get(self, _k: str) -> str:
                # Not valid JSON -> triggers the inner except branch.
                return f"not-json {HOSTILE}"

        monkeypatch.setattr(fetcher, "_get_sync_redis", lambda: FakeRedis())
        out = fetcher._refresh_candidates()
        # No valid candidates produced; return is None.
        assert out is None
        captured = loguru_sink.getvalue()
        assert "Error parsing program" in captured
        _assert_sanitized(captured)


# ---------------------------------------------------------------------------
# gigaevo/utils/trackers/backends/redis.py
# ---------------------------------------------------------------------------


class TestRedisMetricsBackendSanitization:
    def _make_backend(self) -> Any:
        from gigaevo.utils.trackers.backends.redis import RedisMetricsBackend
        from gigaevo.utils.trackers.configs import RedisMetricsConfig

        cfg = RedisMetricsConfig(
            redis_url="redis://localhost:6379/0",
            key_prefix="test_metrics",
            store_history=True,
        )
        backend = RedisMetricsBackend(cfg)
        # Substitute a fakeredis client so flush can actually run.
        backend._client = fakeredis.FakeRedis(decode_responses=True)
        return backend

    def test_clean_scalar_tag_round_trips_latest_and_history(self) -> None:
        backend = self._make_backend()

        backend.write_scalar("loss/train", 0.25, step=3, wall_time=1.5)
        backend.flush()

        assert backend.get_latest("loss/train") == {"loss/train": 0.25}
        assert backend.list_metrics() == ["loss/train"]
        assert backend.get_history("loss/train") == [
            {"s": 3, "t": 1.5, "v": 0.25, "k": "scalar"}
        ]

    def test_hostile_scalar_tag_round_trips_via_sanitized_field(self) -> None:
        backend = self._make_backend()
        tag = f"loss {HOSTILE}/train"
        safe_tag = backend._field_tag(tag)

        backend.write_scalar(tag, 0.5, step=4, wall_time=2.0)
        backend.flush()

        assert backend.get_latest(tag) == {safe_tag: 0.5}
        assert backend.list_metrics() == [safe_tag]
        assert backend.get_history(tag) == [
            {"s": 4, "t": 2.0, "v": 0.5, "k": "scalar"}
        ]
        _assert_sanitized(safe_tag)

    def test_history_key_strips_hostile_bytes_from_tag(self) -> None:
        backend = self._make_backend()
        # Tag with ANSI/BIDI/NUL/CR — clean_identifier strips them all.
        tag = f"loss/train {HOSTILE} step"
        key = backend._k_history(tag)
        # The conservative charset survives; everything hostile is gone.
        for hostile_ch in ("\x1b", "\x00", "\x07", "\r", LONE_HIGH, "‮"):
            assert hostile_ch not in key
        # Prefix is intact.
        assert key.startswith("test_metrics:history:")

    def test_history_key_caps_length(self) -> None:
        backend = self._make_backend()
        long_tag = "x" * 500
        key = backend._k_history(long_tag)
        # 128-char cap from clean_identifier plus the static prefix.
        # "test_metrics:history:" + 128 chars = 149.
        assert len(key) <= len("test_metrics:history:") + 128

    def test_write_text_sanitizes_value_for_dbtext(self) -> None:
        backend = self._make_backend()
        # Hostile payload: NUL + lone surrogate. sanitize_for_dbtext drops
        # both, replacing each with U+FFFD; the buffered entry's value
        # must already be free of those bytes.
        backend.write_text("tag1", f"text{HOSTILE}", step=1, wall_time=0.0)
        stored = backend._buffer[0]["value"]
        assert "\x00" not in stored
        assert LONE_HIGH not in stored
        # The original NUL/lone-surrogate become U+FFFD.
        assert "�" in stored

    def test_flush_writes_hostile_history_without_raising(self) -> None:
        backend = self._make_backend()
        # Histogram entry containing a list with a lone surrogate.
        backend.write_hist(
            f"loss{HOSTILE}/train",
            ["value-with-" + HOSTILE, 1.0, 2.0],
            step=1,
            wall_time=0.0,
        )
        backend.write_text(f"text{HOSTILE}", "payload " + HOSTILE, step=1, wall_time=0.0)
        # flush() invokes json.dumps via deep_sanitize_for_json -> must not
        # raise UnicodeEncodeError. The previous implementation would have.
        backend.flush()
        # Confirm something was actually written to fakeredis.
        history_keys = backend._client.keys("test_metrics:history:*")
        assert len(history_keys) >= 2

    def test_flush_history_payload_is_json_loadable_and_utf8(self) -> None:
        backend = self._make_backend()
        backend.write_text("tag2", f"payload {HOSTILE}", step=5, wall_time=10.0)
        backend.flush()
        # Pull back the history entry and confirm it is a clean JSON record.
        keys = backend._client.keys("test_metrics:history:*")
        assert keys
        raw = backend._client.lrange(keys[0], 0, -1)
        assert raw
        for entry in raw:
            # Already-decoded by decode_responses=True.
            entry.encode("utf-8")  # safe round-trip
            data = json.loads(entry)
            assert data["s"] == 5
            # The stored value never contains the raw hostile bytes.
            v = data["v"]
            if isinstance(v, str):
                assert "\x00" not in v
                assert LONE_HIGH not in v


# ---------------------------------------------------------------------------
# End-to-end: every sanitized log line through a real loguru file sink
# remains UTF-8 encodable.
# ---------------------------------------------------------------------------


class TestEndToEndLoguruFileSinkUtf8():
    def test_real_file_sink_accepts_all_sanitized_messages(self) -> None:
        tf = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".log", encoding="utf-8"
        )
        tf.close()
        path = Path(tf.name)
        sink_id = logger.add(path, format="{message}", level="DEBUG")
        try:
            from gigaevo.utils.text_sanitize import sanitize_for_log

            # Emit each hostile variant through the sink; loguru would
            # raise on a lone surrogate without the sanitizer.
            for src in (HOSTILE, HOSTILE * 3, "\x00" * 10, LONE_HIGH * 5):
                logger.info(sanitize_for_log(src))
            content = path.read_text(encoding="utf-8")
            assert "\x1b" not in content
            assert "\x00" not in content
            assert LONE_HIGH not in content
        finally:
            logger.remove(sink_id)
            if path.exists():
                path.unlink()
