"""Integration tests that prove the sanitizer is wired into LLM call sites.

Each test drives a hostile string (ANSI escape, NUL, lone surrogate, BIDI
override, CR carriage-return) through the production call site and asserts
the destination — a real loguru sink, a JSON dump, or pydantic field state —
never sees the raw hostile bytes.

The tests intentionally use the production logging path (``loguru.logger``
with a captured ``StringIO`` sink) so that a regression that silently drops
a ``sanitize_for_log`` wrap surfaces as a hostile byte reappearing in the
captured output. They are organized one class per modified file under
``gigaevo/llm/``.
"""

from __future__ import annotations

import io
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from loguru import logger
import pytest

from gigaevo.llm.agents.mutation import (
    MutationAgent,
    MutationChange,
    MutationState,
    MutationStructuredOutput,
)
from gigaevo.llm.bandit import BanditModelRouter, MutationOutcome
from gigaevo.llm.models import MultiModelRouter, _redact_url
from gigaevo.llm.token_tracking import TokenTracker
from gigaevo.programs.program import Program
from tests.conftest import NullWriter


# ---------------------------------------------------------------------------
# Shared hostile-input fixtures (kept consistent with tests/utils/...)
# ---------------------------------------------------------------------------

LONE_HIGH = "\ud83d"
HOSTILE = (
    "\x1b[31merr\x1b[0m"  # ANSI red
    "\x00NUL"  # NUL
    "\rCR"  # CR forgery
    "\x07BEL"  # bell
    f"{LONE_HIGH}LS"  # lone surrogate
    "‮RLO"  # RLO BIDI override
)


@pytest.fixture
def loguru_sink():
    """Add a string-buffer loguru sink, yield it, tear down."""
    buf = io.StringIO()
    sink_id = logger.add(buf, format="{message}", level="DEBUG")
    yield buf
    logger.remove(sink_id)


def _assert_no_raw_hostile(captured: str) -> None:
    assert "\x1b" not in captured, "raw ANSI ESC survived"
    assert "\x00" not in captured, "raw NUL survived"
    assert "\x07" not in captured, "raw BEL survived"
    assert LONE_HIGH not in captured, "lone surrogate survived"
    assert "‮" not in captured, "BIDI RLO survived"
    # Captured string must encode cleanly as UTF-8 (loguru already wrote it).
    captured.encode("utf-8")


def _mock_model(name: str) -> MagicMock:
    m = MagicMock()
    m.model_name = name
    m.with_structured_output = MagicMock(return_value=MagicMock())
    return m


# ---------------------------------------------------------------------------
# gigaevo/llm/models.py — MultiModelRouter init + _verify_models
# ---------------------------------------------------------------------------


class TestModelRouterLogSanitization:
    """Init banner and _verify_models warnings must never emit hostile bytes."""

    def test_init_log_with_hostile_model_name(
        self, loguru_sink, monkeypatch
    ) -> None:
        # Hostile bytes in model_name should be stripped by _safe_model_name
        # before reaching the init INFO line. Patch _verify_models out — this
        # test is about the init banner, not the server probe; the real probe
        # would otherwise spend ~10s timing out against the fake host.
        monkeypatch.setattr(
            MultiModelRouter, "_verify_models", lambda self: None
        )
        models = [_mock_model(f"gpt-4{HOSTILE}"), _mock_model("gpt-3.5-turbo")]
        # base_url must be present so the second loop also fires; we use one
        # that contains userinfo, exercising _redact_url alongside sanitizing.
        models[0].base_url = "http://user:pwd@host:8000/v1"
        models[1].base_url = "http://host:8000/v1"
        MultiModelRouter(models, [0.5, 0.5], writer=NullWriter(), name="san")
        captured = loguru_sink.getvalue()
        _assert_no_raw_hostile(captured)
        # Cleaned form survives — sanity-check the prefix is recognizable.
        assert "[MultiModelRouter:san]" in captured
        # Userinfo from base_url must be redacted in the log.
        assert "user:pwd" not in captured
        assert "pwd@" not in captured

    def test_verify_models_failure_log_sanitized(
        self, loguru_sink, monkeypatch
    ) -> None:
        """When the server probe raises, the exception message is sanitized."""
        import urllib.request

        def boom(*_a, **_kw):
            raise OSError(f"connect failed: {HOSTILE}")

        monkeypatch.setattr(urllib.request, "urlopen", boom)
        models = [_mock_model("gpt-4")]
        models[0].base_url = "http://host:8000/v1"
        MultiModelRouter(models, [1.0], writer=NullWriter(), name="probe")
        captured = loguru_sink.getvalue()
        assert "Cannot verify models" in captured
        _assert_no_raw_hostile(captured)

    def test_verify_models_not_found_log_sanitized(
        self, loguru_sink, monkeypatch
    ) -> None:
        """Server-returned model ids with hostile bytes are sanitized in WARN."""
        import urllib.request

        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                # The server claims to host a different model than the
                # configured one, with hostile bytes in its id.
                return json.dumps(
                    {"data": [{"id": f"other-model{HOSTILE}"}]}
                ).encode("utf-8")

        def fake_urlopen(*_a, **_kw):
            return FakeResp()

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        models = [_mock_model("gpt-4")]
        models[0].base_url = "http://host:8000/v1"
        MultiModelRouter(models, [1.0], writer=NullWriter(), name="probe2")
        captured = loguru_sink.getvalue()
        assert "NOT FOUND" in captured
        _assert_no_raw_hostile(captured)


class TestRedactUrl:
    """Strip userinfo, keep everything else."""

    def test_userinfo_stripped(self) -> None:
        assert _redact_url("http://u:p@h:8000/x") == "http://h:8000/x"

    def test_no_userinfo_preserved(self) -> None:
        assert _redact_url("http://h:8000/v1") == "http://h:8000/v1"

    def test_parse_failure_returns_input(self) -> None:
        # http://[ is an unparseable URL on stricter parsers — at minimum
        # the helper must not raise and must yield a str.
        result = _redact_url("not a url at all")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# gigaevo/llm/agents/mutation.py — structured-output validators + log calls
# ---------------------------------------------------------------------------


class TestMutationStructuredOutputValidators:
    """Field validators must scrub LLM-supplied text on construction."""

    def test_hostile_archetype_scrubbed_at_validation(self) -> None:
        out = MutationStructuredOutput(
            archetype=f"clever{HOSTILE}archetype",
            justification=f"because{HOSTILE}",
            insights_used=[f"insight{HOSTILE}1", "clean"],
            changes=[
                MutationChange(
                    description=f"swap loop{HOSTILE}",
                    explanation=f"why{HOSTILE}",
                )
            ],
            code=f"def f():\n    return 1{HOSTILE}",
        )
        # No raw hostile bytes survive in any string field.
        assert "\x1b" not in out.archetype
        assert "\x00" not in out.archetype
        assert LONE_HIGH not in out.archetype
        assert "\x00" not in out.justification
        assert all("\x00" not in s for s in out.insights_used)
        assert "\x1b" not in out.changes[0].description
        assert "\x07" not in out.changes[0].explanation
        # The code field is also scrubbed (ANSI/BIDI/C0-non-LF have no
        # legitimate place in Python source). LF must still survive.
        assert "\x00" not in out.code
        assert "\n" in out.code  # legitimate newline preserved

    def test_model_dump_json_succeeds_after_validation(self) -> None:
        """Lone surrogate inside any field would otherwise abort orjson; the
        validator pre-scrubs so JSON serialization is total."""
        out = MutationStructuredOutput(
            archetype=f"a{LONE_HIGH}",
            justification="j",
            insights_used=[f"i{LONE_HIGH}"],
            changes=[],
            code=f"x = 1{LONE_HIGH}",
        )
        blob = out.model_dump_json()
        # Round-trip parse must succeed — no encoder failure.
        json.loads(blob)


class TestMutationAgentLogSanitization:
    """Direct logger calls inside MutationAgent must scrub LLM-derived text."""

    def _make_agent(self) -> MutationAgent:
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=MagicMock())
        return MutationAgent(
            llm=mock_llm,
            system_prompt="sys",
            user_prompt_template="Mutate {count}:\n{parent_blocks}",
            mutation_mode="rewrite",
        )

    async def test_acall_llm_failure_log_sanitized(self, loguru_sink) -> None:
        agent = self._make_agent()
        agent.structured_llm = MagicMock()
        agent.structured_llm.ainvoke = AsyncMock(
            side_effect=RuntimeError(f"oops {HOSTILE}")
        )
        state: MutationState = {
            "input": [],
            "mutation_mode": "rewrite",
            "messages": [],
            "llm_response": None,
            "final_code": "",
            "mutation_label": "",
        }
        await agent.acall_llm(state)
        captured = loguru_sink.getvalue()
        assert "Structured LLM call failed" in captured
        _assert_no_raw_hostile(captured)
        # State["error"] must also be scrubbed so callers can stash it.
        assert "\x00" not in state.get("error", "")

    def test_parse_response_no_output_log_sanitized(self, loguru_sink) -> None:
        agent = self._make_agent()
        state: MutationState = {
            "input": [],
            "mutation_mode": "rewrite",
            "messages": [],
            "llm_response": None,
            "final_code": "",
            "mutation_label": "",
            "error": f"upstream said: {HOSTILE}",
        }
        agent.parse_response(state)
        captured = loguru_sink.getvalue()
        assert "No structured output" in captured
        _assert_no_raw_hostile(captured)


# ---------------------------------------------------------------------------
# gigaevo/llm/token_tracking.py — track() error path
# ---------------------------------------------------------------------------


class TestTokenTrackerWiring:
    def test_validation_error_caught_and_logged_sanitized(
        self, loguru_sink
    ) -> None:
        """A provider returning garbage token-usage types must not raise out
        of TokenTracker.track; the failure is logged with sanitized text."""
        tracker = TokenTracker(name="t", writer=NullWriter())

        class BadResponse:
            @property
            def response_metadata(self):
                # ``prompt_tokens`` is a string — pydantic coerces it, but if
                # we use a clearly non-coercible value we exercise the
                # try/except path.
                return {"token_usage": {"prompt_tokens": object()}}

        # Should not raise even though TokenUsage.from_response will hit
        # validation/type errors.
        tracker.track(BadResponse(), model_name=f"model{HOSTILE}")
        captured = loguru_sink.getvalue()
        _assert_no_raw_hostile(captured)

    def test_model_name_cleaned_in_no_usage_branch(self, loguru_sink) -> None:
        """When response carries no usage, the debug log must show a cleaned
        model name (control chars in ``model_name`` would otherwise reach
        loguru via the ``{}`` slot)."""
        tracker = TokenTracker(name="t", writer=NullWriter())

        class EmptyResponse:
            response_metadata: dict = {}

        tracker.track(EmptyResponse(), model_name=f"model{HOSTILE}")
        captured = loguru_sink.getvalue()
        assert "No token usage" in captured
        _assert_no_raw_hostile(captured)


# ---------------------------------------------------------------------------
# gigaevo/llm/agents/memory_selector.py — backend init + search log paths
# ---------------------------------------------------------------------------


class TestMemorySelectorLogSanitization:
    """The memory selector logs over backend errors; those strings are
    backend-supplied and must be scrubbed before reaching loguru."""

    async def test_search_unavailable_log_sanitized(self, loguru_sink) -> None:
        """When ``self.memory`` is None, ``select()`` logs the cached backend
        error verbatim — that path must run the value through sanitize."""
        from gigaevo.llm.agents.memory_selector import (
            MemorySelection,
            MemorySelectorAgent,
        )

        agent = MemorySelectorAgent.__new__(MemorySelectorAgent)
        agent.memory = None
        agent._backend_error = f"backend died: {HOSTILE}"
        import asyncio as _aio

        agent._search_lock = _aio.Lock()

        result = await agent.select(
            input=[],
            mutation_mode="rewrite",
            task_description="t",
            metrics_description="m",
            memory_text="",
            max_cards=1,
        )
        assert isinstance(result, MemorySelection)
        assert result.cards == []
        captured = loguru_sink.getvalue()
        assert "Memory backend unavailable" in captured
        _assert_no_raw_hostile(captured)

    async def test_search_failure_log_sanitized(self, loguru_sink) -> None:
        """When the underlying GAM search raises with hostile bytes in the
        exception message, the WARN line must show the scrubbed form."""
        from gigaevo.llm.agents.memory_selector import (
            MemorySelection,
            MemorySelectorAgent,
        )

        agent = MemorySelectorAgent.__new__(MemorySelectorAgent)

        # Memory backend that raises from research and from search.
        class BadMem:
            research_agent = None

            def search(self, query: str) -> str:
                raise RuntimeError(f"search exploded {HOSTILE}")

        agent.memory = BadMem()
        agent._backend_error = None
        import asyncio as _aio

        agent._search_lock = _aio.Lock()

        result = await agent.select(
            input=[],
            mutation_mode="rewrite",
            task_description="t",
            metrics_description="m",
            memory_text="",
            max_cards=1,
        )
        assert isinstance(result, MemorySelection)
        assert result.cards == []
        captured = loguru_sink.getvalue()
        assert "Red memory search failed" in captured
        _assert_no_raw_hostile(captured)


# ---------------------------------------------------------------------------
# gigaevo/llm/bandit.py — on_mutation_outcome debug log
# ---------------------------------------------------------------------------


class TestBanditRouterLogSanitization:
    def _make_router(self) -> BanditModelRouter:
        models = [_mock_model("m1"), _mock_model("m2")]
        return BanditModelRouter(
            models,
            [0.5, 0.5],
            writer=NullWriter(),
            name="bandit",
            fitness_key="fitness",
            higher_is_better=True,
        )

    def test_outcome_log_with_hostile_model_metadata(self, loguru_sink) -> None:
        """``program.get_metadata("mutation_model")`` could carry hostile text
        if any upstream stashes it raw; the bandit log must sanitize."""
        router = self._make_router()
        # Patch the bandit update so its KeyError on the hostile key (a
        # separate pre-existing concern queued as follow-up) does not mask
        # the wiring assertion we care about here.
        router._bandit.update_reward = lambda *a, **kw: None  # type: ignore[method-assign]
        program = Program(code="x = 1")
        program.metadata["mutation_model"] = f"m1{HOSTILE}"
        # No parent metrics → REJECTED_ACCEPTOR path that logs raw=0.0 line.
        router.on_mutation_outcome(
            program,
            parents=[],
            outcome=MutationOutcome.REJECTED_ACCEPTOR,
        )
        captured = loguru_sink.getvalue()
        assert "Reward for" in captured
        _assert_no_raw_hostile(captured)
