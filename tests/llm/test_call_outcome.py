"""Tests for the LLM call outcome classifier.

Every classification case below was verified against the actually-installed
provider sources (openai 2.36.0, httpx 0.28.1, langchain-openai 1.2.1,
langchain-core 1.4.0) at the time of writing. If a future SDK release
renames a class, the offending case will fail with a clear message
pointing at the missing fingerprint.
"""

from __future__ import annotations

from typing import Final

import httpx
from langchain_core.exceptions import OutputParserException
from langchain_openai.chat_models._client_utils import StreamChunkTimeoutError
from langchain_openai.chat_models.base import (
    OpenAIAPIContextOverflowError,
    OpenAIContextOverflowError,
    OpenAIRefusalError,
)
from langchain_openai.middleware.openai_moderation import OpenAIModerationError
import openai
from pydantic import ValidationError
import pytest

from gigaevo.llm.call_outcome import (
    INVARIANTS,
    OUTCOME_ACTION,
    ZERO_REWARD_OUTCOMES,
    BanditAction,
    LLMCallOutcome,
    LLMCallResult,
    classify_call_result,
)

_REQ: Final[httpx.Request] = httpx.Request(
    "POST", "https://api.openai.com/v1/chat/completions"
)


def _resp(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, headers=headers or {}, request=_REQ)


# ---------------------------------------------------------------------------
# Module-load invariants
# ---------------------------------------------------------------------------


class TestModuleInvariants:
    def test_outcome_action_covers_every_outcome(self) -> None:
        assert set(OUTCOME_ACTION) == set(LLMCallOutcome)

    def test_success_is_the_only_defer(self) -> None:
        assert OUTCOME_ACTION[LLMCallOutcome.SUCCESS] is BanditAction.DEFER_TO_OUTCOME
        for outcome in LLMCallOutcome:
            if outcome is LLMCallOutcome.SUCCESS:
                continue
            assert OUTCOME_ACTION[outcome] is BanditAction.INJECT_ZERO_REWARD, (
                f"{outcome} unexpectedly defers"
            )

    def test_zero_reward_set_excludes_success(self) -> None:
        assert LLMCallOutcome.SUCCESS not in ZERO_REWARD_OUTCOMES
        assert len(ZERO_REWARD_OUTCOMES) == len(LLMCallOutcome) - 1

    def test_invariants_documentation_present(self) -> None:
        assert len(INVARIANTS) >= 6
        for line in INVARIANTS:
            assert isinstance(line, str) and line


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


class TestSuccess:
    def test_none_is_success(self) -> None:
        result = classify_call_result(None)
        assert result.outcome is LLMCallOutcome.SUCCESS
        assert result.is_failure is False
        assert result.action is BanditAction.DEFER_TO_OUTCOME
        assert result.exception_class is None
        assert result.http_status is None
        assert result.cause_chain == ()

    def test_none_carries_model_name(self) -> None:
        result = classify_call_result(None, model_name="gpt-4")
        assert result.model_name == "gpt-4"
        assert result.outcome is LLMCallOutcome.SUCCESS


# ---------------------------------------------------------------------------
# openai 2.x APIStatusError subclasses — status_code authoritative
# ---------------------------------------------------------------------------


class TestOpenAIStatusErrors:
    def test_bad_request_400(self) -> None:
        result = classify_call_result(
            openai.BadRequestError("x", response=_resp(400), body=None)
        )
        assert result.outcome is LLMCallOutcome.BAD_REQUEST
        assert result.http_status == 400

    def test_authentication_error_401(self) -> None:
        result = classify_call_result(
            openai.AuthenticationError("x", response=_resp(401), body=None)
        )
        assert result.outcome is LLMCallOutcome.AUTH_FAILED
        assert result.http_status == 401

    def test_permission_denied_403(self) -> None:
        result = classify_call_result(
            openai.PermissionDeniedError("x", response=_resp(403), body=None)
        )
        assert result.outcome is LLMCallOutcome.AUTH_FAILED
        assert result.http_status == 403

    def test_not_found_404_falls_to_bad_request(self) -> None:
        result = classify_call_result(
            openai.NotFoundError("x", response=_resp(404), body=None)
        )
        assert result.outcome is LLMCallOutcome.BAD_REQUEST

    def test_conflict_409_falls_to_bad_request(self) -> None:
        result = classify_call_result(
            openai.ConflictError("x", response=_resp(409), body=None)
        )
        assert result.outcome is LLMCallOutcome.BAD_REQUEST

    def test_unprocessable_entity_422_falls_to_bad_request(self) -> None:
        result = classify_call_result(
            openai.UnprocessableEntityError("x", response=_resp(422), body=None)
        )
        assert result.outcome is LLMCallOutcome.BAD_REQUEST

    def test_rate_limit_429(self) -> None:
        result = classify_call_result(
            openai.RateLimitError("x", response=_resp(429), body=None)
        )
        assert result.outcome is LLMCallOutcome.RATE_LIMITED
        assert result.http_status == 429

    def test_internal_server_error_500(self) -> None:
        result = classify_call_result(
            openai.InternalServerError("x", response=_resp(500), body=None)
        )
        assert result.outcome is LLMCallOutcome.SERVER_5XX

    def test_unknown_5xx_classifies_as_server(self) -> None:
        result = classify_call_result(
            openai.APIStatusError("x", response=_resp(503), body=None)
        )
        assert result.outcome is LLMCallOutcome.SERVER_5XX


# ---------------------------------------------------------------------------
# openai connection/timeout (no status_code)
# ---------------------------------------------------------------------------


class TestOpenAIConnectionAndTimeout:
    def test_api_connection_error(self) -> None:
        result = classify_call_result(openai.APIConnectionError(request=_REQ))
        assert result.outcome is LLMCallOutcome.NETWORK_ERROR
        assert result.http_status is None

    def test_api_timeout_error(self) -> None:
        result = classify_call_result(openai.APITimeoutError(request=_REQ))
        assert result.outcome is LLMCallOutcome.TIMEOUT
        assert result.http_status is None


# ---------------------------------------------------------------------------
# httpx layer
# ---------------------------------------------------------------------------


class TestHttpxExceptions:
    @pytest.mark.parametrize(
        "exc_factory",
        [
            lambda: httpx.ConnectTimeout("boom", request=_REQ),
            lambda: httpx.ReadTimeout("boom", request=_REQ),
            lambda: httpx.WriteTimeout("boom", request=_REQ),
            lambda: httpx.PoolTimeout("boom", request=_REQ),
        ],
    )
    def test_timeouts(self, exc_factory) -> None:
        result = classify_call_result(exc_factory())
        assert result.outcome is LLMCallOutcome.TIMEOUT

    @pytest.mark.parametrize(
        "exc_factory",
        [
            lambda: httpx.ConnectError("x", request=_REQ),
            lambda: httpx.ReadError("x", request=_REQ),
            lambda: httpx.WriteError("x", request=_REQ),
            lambda: httpx.CloseError("x", request=_REQ),
            lambda: httpx.RemoteProtocolError("x", request=_REQ),
            lambda: httpx.LocalProtocolError("x", request=_REQ),
            lambda: httpx.ProxyError("x", request=_REQ),
            lambda: httpx.UnsupportedProtocol("x", request=_REQ),
        ],
    )
    def test_network_errors(self, exc_factory) -> None:
        result = classify_call_result(exc_factory())
        assert result.outcome is LLMCallOutcome.NETWORK_ERROR

    def test_decoding_error_is_parse_failed(self) -> None:
        result = classify_call_result(httpx.DecodingError("bad json", request=_REQ))
        assert result.outcome is LLMCallOutcome.PARSE_FAILED


# ---------------------------------------------------------------------------
# Context overflow (langchain-core base + langchain-openai specializations)
# ---------------------------------------------------------------------------


class TestContextOverflow:
    def test_openai_context_overflow_error(self) -> None:
        # OpenAIContextOverflowError inherits from BadRequestError(400) AND
        # ContextOverflowError; MRO match must override the 400→BAD_REQUEST
        # classification.
        result = classify_call_result(
            OpenAIContextOverflowError("overflow", response=_resp(400), body=None)
        )
        assert result.outcome is LLMCallOutcome.CONTEXT_OVERFLOW

    def test_openai_api_context_overflow_error(self) -> None:
        class _Fake(OpenAIAPIContextOverflowError):
            def __init__(self) -> None:
                pass

        assert classify_call_result(_Fake()).outcome is LLMCallOutcome.CONTEXT_OVERFLOW

    def test_text_marker_fallback(self) -> None:
        # A generic ValueError whose message advertises context overflow
        # still classifies correctly — covers providers that raise raw
        # ValueError without a dedicated class.
        result = classify_call_result(
            ValueError("Your prompt is too long for this model")
        )
        assert result.outcome is LLMCallOutcome.CONTEXT_OVERFLOW

    def test_text_marker_on_400_overrides_bad_request(self) -> None:
        # A real 400 whose message names overflow should classify as
        # CONTEXT_OVERFLOW, not BAD_REQUEST.
        result = classify_call_result(
            openai.BadRequestError(
                "This model's maximum context length is 8192 tokens",
                response=_resp(400),
                body=None,
            )
        )
        assert result.outcome is LLMCallOutcome.CONTEXT_OVERFLOW


# ---------------------------------------------------------------------------
# Output truncated / content filter
# ---------------------------------------------------------------------------


class TestOutputAndContentSignals:
    def test_openai_refusal_error(self) -> None:
        result = classify_call_result(OpenAIRefusalError("refused"))
        assert result.outcome is LLMCallOutcome.CONTENT_FILTERED

    def test_openai_moderation_error(self) -> None:
        # The real constructor takes keyword args we can't easily fake;
        # subclass to bypass.
        class _FakeModeration(OpenAIModerationError):
            def __init__(self) -> None:
                pass

        result = classify_call_result(_FakeModeration())
        assert result.outcome is LLMCallOutcome.CONTENT_FILTERED

    def test_length_finish_reason(self) -> None:
        # LengthFinishReasonError needs a ChatCompletion to construct; subclass.
        from openai import LengthFinishReasonError

        class _FakeLength(LengthFinishReasonError):
            def __init__(self) -> None:
                pass

        result = classify_call_result(_FakeLength())
        assert result.outcome is LLMCallOutcome.OUTPUT_TRUNCATED

    def test_content_filter_finish_reason(self) -> None:
        from openai import ContentFilterFinishReasonError

        result = classify_call_result(ContentFilterFinishReasonError())
        assert result.outcome is LLMCallOutcome.CONTENT_FILTERED


# ---------------------------------------------------------------------------
# Built-in / asyncio fallbacks
# ---------------------------------------------------------------------------


class TestBuiltinFallbacks:
    def test_asyncio_timeout_error_is_builtin_timeout(self) -> None:
        # Since Python 3.11, asyncio.TimeoutError is an alias for TimeoutError;
        # the classifier handles either route via the builtin hierarchy.
        import asyncio

        assert asyncio.TimeoutError is TimeoutError
        result = classify_call_result(TimeoutError())
        assert result.outcome is LLMCallOutcome.TIMEOUT

    def test_stream_chunk_timeout_via_builtin_hierarchy(self) -> None:
        # StreamChunkTimeoutError extends builtin TimeoutError.
        result = classify_call_result(StreamChunkTimeoutError(30.0))
        assert result.outcome is LLMCallOutcome.TIMEOUT

    def test_builtin_connection_refused(self) -> None:
        result = classify_call_result(ConnectionRefusedError("refused"))
        assert result.outcome is LLMCallOutcome.NETWORK_ERROR

    def test_builtin_connection_reset(self) -> None:
        result = classify_call_result(ConnectionResetError("reset"))
        assert result.outcome is LLMCallOutcome.NETWORK_ERROR


# ---------------------------------------------------------------------------
# Parser failures
# ---------------------------------------------------------------------------


class TestParseFailures:
    def test_output_parser_exception_alone(self) -> None:
        result = classify_call_result(OutputParserException("bad parse"))
        assert result.outcome is LLMCallOutcome.PARSE_FAILED

    def test_pydantic_validation_error(self) -> None:
        from pydantic import BaseModel

        class _M(BaseModel):
            x: int

        with pytest.raises(ValidationError) as exc_info:
            _M(x="not an int")  # type: ignore[arg-type]
        result = classify_call_result(exc_info.value)
        assert result.outcome is LLMCallOutcome.PARSE_FAILED

    def test_json_decode_error(self) -> None:
        import json

        with pytest.raises(json.JSONDecodeError) as exc_info:
            json.loads("not json")
        result = classify_call_result(exc_info.value)
        assert result.outcome is LLMCallOutcome.PARSE_FAILED


# ---------------------------------------------------------------------------
# Cause-chain priority
# ---------------------------------------------------------------------------


class TestCauseChainPriority:
    def test_parser_wrapping_rate_limit_classifies_as_rate_limited(self) -> None:
        # langchain wraps API errors in OutputParserException; bandit cares
        # about the root cause, not the wrapper.
        inner = openai.RateLimitError("rate", response=_resp(429), body=None)
        outer = OutputParserException("parser saw bad output")
        outer.__cause__ = inner
        result = classify_call_result(outer)
        assert result.outcome is LLMCallOutcome.RATE_LIMITED
        assert "OutputParserException" in result.cause_chain
        assert "RateLimitError" in result.cause_chain

    def test_chain_uses_context_when_cause_missing(self) -> None:
        # Implicit context (during-handling-of) is followed when __cause__ is None.
        inner = openai.AuthenticationError("no key", response=_resp(401), body=None)
        outer = RuntimeError("wrap")
        outer.__context__ = inner
        result = classify_call_result(outer)
        assert result.outcome is LLMCallOutcome.AUTH_FAILED

    def test_cause_priority_picks_more_actionable_outcome(self) -> None:
        # If chain has both TIMEOUT and RATE_LIMITED, RATE_LIMITED wins
        # because back-off is more actionable than retry.
        rl = openai.RateLimitError("rate", response=_resp(429), body=None)
        to = openai.APITimeoutError(request=_REQ)
        to.__cause__ = rl
        result = classify_call_result(to)
        assert result.outcome is LLMCallOutcome.RATE_LIMITED

    def test_self_referential_cycle_terminates(self) -> None:
        e1 = ValueError("a")
        e2 = ValueError("b")
        e1.__cause__ = e2
        e2.__cause__ = e1
        result = classify_call_result(e1)
        # No crash, chain length bounded.
        assert len(result.cause_chain) <= 8
        assert result.outcome is LLMCallOutcome.OTHER_EXCEPTION


# ---------------------------------------------------------------------------
# Retry-After extraction
# ---------------------------------------------------------------------------


class TestRetryAfter:
    def test_retry_after_integer_seconds(self) -> None:
        result = classify_call_result(
            openai.RateLimitError(
                "rate", response=_resp(429, {"retry-after": "17"}), body=None
            )
        )
        assert result.retry_after_seconds == 17.0

    def test_retry_after_float_seconds(self) -> None:
        result = classify_call_result(
            openai.RateLimitError(
                "rate", response=_resp(429, {"retry-after": "1.5"}), body=None
            )
        )
        assert result.retry_after_seconds == 1.5

    def test_retry_after_missing(self) -> None:
        result = classify_call_result(
            openai.RateLimitError("rate", response=_resp(429), body=None)
        )
        assert result.retry_after_seconds is None

    def test_retry_after_http_date_not_parsed(self) -> None:
        # HTTP-date form is not supported — caller treats None as "no hint".
        result = classify_call_result(
            openai.RateLimitError(
                "rate",
                response=_resp(429, {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}),
                body=None,
            )
        )
        assert result.retry_after_seconds is None

    def test_retry_after_negative_rejected(self) -> None:
        result = classify_call_result(
            openai.RateLimitError(
                "rate", response=_resp(429, {"retry-after": "-5"}), body=None
            )
        )
        assert result.retry_after_seconds is None


# ---------------------------------------------------------------------------
# Schema invariants
# ---------------------------------------------------------------------------


class TestSchemaInvariants:
    def test_result_is_frozen(self) -> None:
        result = classify_call_result(None)
        with pytest.raises(ValidationError):
            result.outcome = LLMCallOutcome.TIMEOUT  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            LLMCallResult(outcome=LLMCallOutcome.SUCCESS, bogus="x")  # type: ignore[call-arg]

    def test_http_status_range_enforced(self) -> None:
        with pytest.raises(ValidationError):
            LLMCallResult(outcome=LLMCallOutcome.SERVER_5XX, http_status=42)

    def test_action_property_returns_mapping_value(self) -> None:
        assert classify_call_result(None).action is BanditAction.DEFER_TO_OUTCOME
        assert (
            classify_call_result(RuntimeError("x")).action
            is BanditAction.INJECT_ZERO_REWARD
        )

    def test_is_failure_property(self) -> None:
        assert classify_call_result(None).is_failure is False
        assert classify_call_result(RuntimeError("x")).is_failure is True


# ---------------------------------------------------------------------------
# Defensive / hostile inputs
# ---------------------------------------------------------------------------


class TestHostileInputs:
    def test_bool_masquerading_as_status_code_rejected(self) -> None:
        # ``True`` is an int in Python; the extractor must refuse it.
        class _SneakyBool(Exception):
            status_code = True  # type: ignore[assignment]

        result = classify_call_result(_SneakyBool("x"))
        assert result.http_status is None

    def test_negative_status_code_rejected(self) -> None:
        class _Negative(Exception):
            status_code = -1

        result = classify_call_result(_Negative("x"))
        assert result.http_status is None

    def test_str_exception_that_raises(self) -> None:
        class _BadStr(Exception):
            def __str__(self) -> str:
                raise RuntimeError("__str__ blew up")

        # Must not crash; message ends up None.
        result = classify_call_result(_BadStr())
        assert result.message is None
        assert result.outcome is LLMCallOutcome.OTHER_EXCEPTION

    def test_message_pass_through_full_length(self) -> None:
        # Provider-side response is already bounded by max_tokens; we never
        # second-guess it. Whatever ``str(exc)`` returns lands verbatim.
        long = "y" * 5000
        result = classify_call_result(RuntimeError(long))
        assert result.message == long

    def test_unknown_runtime_error_is_other(self) -> None:
        assert (
            classify_call_result(RuntimeError("mystery")).outcome
            is LLMCallOutcome.OTHER_EXCEPTION
        )


# ---------------------------------------------------------------------------
# Audit-discovered defects: regression guards
# ---------------------------------------------------------------------------


class TestSurrogateInMessage:
    """The classifier returns ``LLMCallResult.message`` as ``str(exc)``. A
    provider exception text containing a UTF-16 surrogate code point would
    otherwise survive into the result and crash any downstream UTF-8
    encoder or JSON serializer (``orjson``, ``model_dump_json``, asyncpg
    TEXT write). ``_safe_message`` replaces surrogates with U+FFFD."""

    def test_lone_high_surrogate_replaced(self) -> None:
        result = classify_call_result(RuntimeError("a\ud83dz"))
        assert result.message is not None
        assert "\ud83d" not in result.message
        result.message.encode("utf-8")  # must not raise

    def test_lone_low_surrogate_replaced(self) -> None:
        result = classify_call_result(RuntimeError("a\udc00z"))
        assert result.message is not None
        result.message.encode("utf-8")

    def test_adjacent_surrogate_codepoints_both_replaced(self) -> None:
        # Two distinct codepoints, not a valid astral pair in Python str.
        result = classify_call_result(RuntimeError(chr(0xD800) + chr(0xDC00)))
        assert result.message is not None
        for ch in result.message:
            cp = ord(ch)
            assert not (0xD800 <= cp <= 0xDFFF), f"surrogate U+{cp:04X} survived"
        result.message.encode("utf-8")

    def test_real_astral_emoji_preserved(self) -> None:
        result = classify_call_result(RuntimeError("a😀z"))
        assert result.message == "a😀z"

    def test_model_dump_json_succeeds_after_classify(self) -> None:
        result = classify_call_result(RuntimeError("a\ud83dz"))
        # Must not raise PydanticSerializationError / UnicodeEncodeError.
        result.model_dump_json()


class TestRetryAfterNonFiniteRejected:
    """``Retry-After: inf`` or any non-finite parse poisons the bandit's
    sleep budget and diverges from telemetry (JSON emits ``null`` for
    non-finite floats, in-memory readers see ``math.inf``).
    ``_extract_retry_after`` must return ``None``."""

    @pytest.mark.parametrize(
        "header_value",
        ["inf", "Infinity", "-inf", "nan", "NaN", "1" + "0" * 400],
    )
    def test_non_finite_header_rejected(self, header_value: str) -> None:
        result = classify_call_result(
            openai.RateLimitError(
                "rate", response=_resp(429, {"retry-after": header_value}), body=None
            )
        )
        assert result.retry_after_seconds is None
        assert result.outcome is LLMCallOutcome.RATE_LIMITED  # outcome unaffected

    def test_finite_header_still_accepted(self) -> None:
        result = classify_call_result(
            openai.RateLimitError(
                "rate", response=_resp(429, {"retry-after": "30"}), body=None
            )
        )
        assert result.retry_after_seconds == 30.0

    def test_excessive_finite_header_rejected_at_schema(self) -> None:
        # 24 hours is the schema cap. Past that, reject so a misbehaving
        # upstream cannot park the bandit on a multi-year sleep.
        result = classify_call_result(
            openai.RateLimitError(
                "rate",
                response=_resp(429, {"retry-after": "100000"}),
                body=None,
            )
        )
        assert result.retry_after_seconds is None


class TestHostileClassMetadata:
    """``type(exc).__module__`` and ``__name__`` can be set to non-str
    values via metaclass shenanigans (legal in Python). The classifier
    must coerce defensively and never raise ``ValidationError`` from
    inside its own body — that would violate the documented totality
    contract."""

    def test_int_module_does_not_break_totality(self) -> None:
        class _Weird(Exception):
            pass

        _Weird.__module__ = 42  # type: ignore[assignment]
        result = classify_call_result(_Weird("x"))
        # Coerced via str() — exception_module is "42" (string), not int.
        assert result.exception_module == "42"
        assert result.exception_class == "_Weird"
        assert result.outcome is LLMCallOutcome.OTHER_EXCEPTION

    def test_none_module_handled(self) -> None:
        class _Weird(Exception):
            pass

        _Weird.__module__ = None  # type: ignore[assignment]
        result = classify_call_result(_Weird("x"))
        assert result.exception_module is None
        assert result.exception_class == "_Weird"

    # The metaclass-raises-on-__module__ scenario is covered by
    # TestTotalityWithHostileExceptionPlumbing.test_hostile_mro_via_metaclass_does_not_propagate.


class TestTotalityWithHostileExceptionPlumbing:
    """``classify_call_result`` is documented total: every constructible
    ``BaseException`` returns an ``LLMCallResult`` without raising. Python
    permits class authors to override ``__cause__`` / ``__context__`` /
    ``__mro__`` with descriptors that raise non-AttributeError. The
    classifier touches these attributes on whatever object the caller
    hands us, so it must defend against the hostile path."""

    def test_hostile_cause_property_does_not_propagate(self) -> None:
        class _HostileCause(Exception):
            @property
            def __cause__(self):  # type: ignore[override]
                raise RuntimeError("evil cause descriptor")

        # Must not raise — totality contract.
        result = classify_call_result(_HostileCause("x"))
        assert result.outcome is LLMCallOutcome.OTHER_EXCEPTION
        assert result.exception_class == "_HostileCause"

    def test_hostile_context_property_does_not_propagate(self) -> None:
        class _HostileContext(Exception):
            @property
            def __context__(self):  # type: ignore[override]
                raise RuntimeError("evil context descriptor")

        result = classify_call_result(_HostileContext("x"))
        assert result.outcome is LLMCallOutcome.OTHER_EXCEPTION
        assert result.exception_class == "_HostileContext"

    def test_hostile_mro_via_metaclass_does_not_propagate(self) -> None:
        class _BadMeta(type):
            @property
            def __mro__(cls):  # type: ignore[override]
                raise RuntimeError("evil mro descriptor")

        class _E(Exception, metaclass=_BadMeta):
            pass

        result = classify_call_result(_E("x"))
        # No MRO available -> all class-fingerprint branches skipped; falls
        # through to OTHER_EXCEPTION (or a built-in isinstance match).
        assert result.outcome is LLMCallOutcome.OTHER_EXCEPTION

    def test_base_exception_subclasses_total(self) -> None:
        # The signature is ``BaseException | None``; KeyboardInterrupt /
        # SystemExit / GeneratorExit are constructible BaseExceptions and
        # must not crash the classifier.
        for cls in (KeyboardInterrupt, SystemExit, GeneratorExit):
            result = classify_call_result(cls("x"))
            assert result.outcome is LLMCallOutcome.OTHER_EXCEPTION
            assert result.exception_class == cls.__name__


# ---------------------------------------------------------------------------
# litellm fingerprint coverage
# ---------------------------------------------------------------------------
# Rather than depend on the heavy litellm import in the test process, we
# synthesize exception classes whose name + MRO shape mirror the real
# litellm exceptions verified at audit time:
#   litellm.ContextWindowExceededError  → BadRequestError (status_code=400)
#   litellm.ContentPolicyViolationError → BadRequestError (status_code=400)
# Without the rule-table additions in this commit, both classify as
# BAD_REQUEST because the 400 status wins.


class _LiteLLMContextWindowExceededError(Exception):
    """Shape-equivalent stand-in for litellm.ContextWindowExceededError.

    The classifier matches on the leaf class name via the MRO walk, so the
    real class name on the synthesized type is what counts for the test.
    Status code 400 is set on the instance to mirror the litellm class-level
    default."""


_LiteLLMContextWindowExceededError.__name__ = "ContextWindowExceededError"


class _LiteLLMContentPolicyViolationError(Exception):
    pass


_LiteLLMContentPolicyViolationError.__name__ = "ContentPolicyViolationError"


class TestLiteLLMRuleCoverage:
    def test_context_window_exceeded_classifies_as_context_overflow(self) -> None:
        exc = _LiteLLMContextWindowExceededError("ctx window exceeded")
        exc.status_code = 400  # type: ignore[attr-defined]
        result = classify_call_result(exc)
        # The MRO-based override beats the 400 status fallback.
        assert result.outcome is LLMCallOutcome.CONTEXT_OVERFLOW
        assert result.http_status == 400

    def test_content_policy_violation_classifies_as_content_filtered(self) -> None:
        exc = _LiteLLMContentPolicyViolationError("policy violation")
        exc.status_code = 400  # type: ignore[attr-defined]
        result = classify_call_result(exc)
        assert result.outcome is LLMCallOutcome.CONTENT_FILTERED
        assert result.http_status == 400


# ---------------------------------------------------------------------------
# PR-B documented invariants
# ---------------------------------------------------------------------------
# These pin the contract PR-B will rely on (cause_chain ordering,
# retry_after_seconds tri-state, action lookup for every outcome).


class TestPRBContract:
    def test_cause_chain_starts_with_surface_exception(self) -> None:
        try:
            try:
                raise ValueError("inner")
            except ValueError as inner:
                raise RuntimeError("outer") from inner
        except RuntimeError as surface:
            result = classify_call_result(surface)
        # Index 0 must be the SURFACE exception's class name; the cause
        # walk extends outward toward the root cause.
        assert result.cause_chain[0] == "RuntimeError"
        assert result.cause_chain[-1] == "ValueError"

    def test_retry_after_zero_is_distinct_from_none(self) -> None:
        zero = openai.RateLimitError(
            "x", response=_resp(429, {"retry-after": "0"}), body=None
        )
        absent = openai.RateLimitError("x", response=_resp(429), body=None)
        zero_result = classify_call_result(zero)
        absent_result = classify_call_result(absent)
        # ``0.0`` means "retry immediately"; ``None`` means "no hint".
        # PR-B must distinguish these two.
        assert zero_result.retry_after_seconds == 0.0
        assert absent_result.retry_after_seconds is None

    def test_action_lookup_for_every_outcome(self) -> None:
        # PR-B will call ``result.action`` after ``classify_call_result``;
        # the property must resolve for every enum member.
        for outcome in LLMCallOutcome:
            stub = LLMCallResult(outcome=outcome)
            assert stub.action in {
                BanditAction.DEFER_TO_OUTCOME,
                BanditAction.INJECT_ZERO_REWARD,
            }

    def test_json_roundtrip_preserves_pr_b_relevant_fields(self) -> None:
        exc = openai.RateLimitError(
            "rate limit hit",
            response=_resp(429, {"retry-after": "42"}),
            body=None,
        )
        result = classify_call_result(exc, model_name="gpt-4o-2024-08-06")
        # Round-trip via the wire format PR-B telemetry will use.
        encoded = result.model_dump_json()
        decoded = LLMCallResult.model_validate_json(encoded)
        assert decoded == result
        assert decoded.outcome is LLMCallOutcome.RATE_LIMITED
        assert decoded.http_status == 429
        assert decoded.retry_after_seconds == 42.0
        assert decoded.model_name == "gpt-4o-2024-08-06"
        assert decoded.action is BanditAction.INJECT_ZERO_REWARD
