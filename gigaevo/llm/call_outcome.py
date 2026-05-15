"""LLM call outcome classification — pure, exception-driven, exhaustive.

================================================================================
CONTEXT
================================================================================

Every LLM call dispatched through the router stack either returns a usable
response or raises. Callers (``BanditModelRouter``, retry shims, telemetry)
need to react to the *kind* of failure, not the specific exception class:
a 429 rate-limit deserves a zero reward but no breaker trip; a 401 auth
error is operator-side and should not be confused with arm failure; a
length-limit truncation is the arm's fault; a langchain
``OutputParserException`` wrapping an ``openai.RateLimitError`` should be
classified by the *root* cause, not by the wrapper.

This module is the one classifier. Every other piece of router/bandit
plumbing consumes ``LLMCallResult`` rather than rolling its own try/except
ladder.

================================================================================
STATE MODEL — 12 outcomes, exhaustive
================================================================================

    SUCCESS              call returned a usable response
    RATE_LIMITED         HTTP 429 / provider rate-limit signal
    SERVER_5XX           HTTP 5xx that is not a more specific signal
    TIMEOUT              asyncio / httpx / openai timeout (HTTP 408/504 too)
    NETWORK_ERROR        connection refused, DNS, TLS, broken pipe, protocol
    AUTH_FAILED          HTTP 401 / 403 / invalid API key
    BAD_REQUEST          HTTP 4xx not classified as a more specific outcome
    CONTEXT_OVERFLOW     prompt exceeded model context window (HTTP 413, or
                         explicit ``ContextOverflowError`` family)
    OUTPUT_TRUNCATED     successful call but the response was cut off at the
                         token budget (``LengthFinishReasonError`` family)
    CONTENT_FILTERED     model refused / moderation blocked the response
                         (``OpenAIRefusalError`` / ``OpenAIModerationError`` /
                         ``ContentFilterFinishReasonError`` family)
    PARSE_FAILED         response received but parsing raised
                         (pydantic ``ValidationError``, ``OutputParserException``,
                         ``json.JSONDecodeError``, ``APIResponseValidationError``,
                         ``httpx.DecodingError``)
    OTHER_EXCEPTION      anything ``classify_call_result`` cannot name

The set is EXHAUSTIVE: ``classify_call_result`` is a total function from
``BaseException | None`` to ``LLMCallResult``. New providers add MRO names
to the existing rule tables — they do not add outcome classes unless the
bandit contract changes.

================================================================================
CLASSIFICATION (total function)
================================================================================

The classifier walks the ``__cause__`` / ``__context__`` chain (terminated
by cycle protection via an ``id()`` set, since Python chains are finite in
practice) and classifies each frame. Outcomes from the chain are then
collapsed via priority ranking (see ``_OUTCOME_PRIORITY``) so a wrapper
exception cannot mask a more informative root cause — a langchain
``OutputParserException`` whose ``__cause__`` is an ``openai.RateLimitError``
lands on ``RATE_LIMITED``, not ``PARSE_FAILED``.

Per-frame classification (``_classify_one``), in order:

    1. MRO-based specific overrides (context overflow, content filter,
       output-truncated) — these outrank HTTP status because the surface
       exception is more specific than the status code (e.g. a 400 that
       is really ``OpenAIContextOverflowError``).
    2. HTTP status lookup (``_STATUS_TO_OUTCOME``) — authoritative for
       common codes (401, 403, 408, 413, 429, 504); range fallthroughs
       cover the rest of 4xx (→ BAD_REQUEST) and 5xx (→ SERVER_5XX).
    3. MRO-based fingerprint match for TIMEOUT / NETWORK_ERROR /
       PARSE_FAILED.
    4. ``isinstance`` against built-ins (``TimeoutError``,
       ``ConnectionError``) — catches ``asyncio.TimeoutError`` (same
       class as ``TimeoutError`` since 3.11) and langchain-openai's
       ``StreamChunkTimeoutError`` (extends builtin ``TimeoutError``).
    5. Context-overflow text marker as a last-resort signal.
    6. Otherwise ``OTHER_EXCEPTION``.

The classifier never imports a provider SDK (openai, httpx,
langchain-openai, langchain-core, pydantic). It inspects ``type(exc).__mro__``
class names, exception attributes (``status_code``, ``response.status_code``,
``response.headers``) and ``str(exc)`` only.

================================================================================
BANDIT ACTION TABLE
================================================================================

    SUCCESS               → DEFER_TO_OUTCOME (reward arrives via on_mutation_outcome)
    every other outcome   → INJECT_ZERO_REWARD

The fine-grained distinction between "arm's fault" (PARSE_FAILED,
CONTENT_FILTERED, OUTPUT_TRUNCATED) and "operator's / infra's fault"
(AUTH_FAILED, RATE_LIMITED, SERVER_5XX, NETWORK_ERROR, TIMEOUT,
CONTEXT_OVERFLOW, BAD_REQUEST) is preserved for telemetry and future
circuit-breakers. The bandit ledger currently treats them the same
because mean-reward signal is self-correcting: an AUTH_FAILED storm
penalizes every arm equally, so relative ranking is preserved.

================================================================================
INVARIANTS (enforced at module load + tested in tests/llm/test_call_outcome.py)
================================================================================

    I1. classify_call_result(None).outcome is SUCCESS; no other input
        produces SUCCESS.
    I2. ``OUTCOME_ACTION`` is closed under ``LLMCallOutcome`` — every
        enum member is a key (asserted at module load).
    I3. SUCCESS is the only outcome with action DEFER_TO_OUTCOME; every
        other outcome has action INJECT_ZERO_REWARD (asserted at module
        load).
    I4. ``_OUTCOME_PRIORITY`` is a permutation of ``LLMCallOutcome``
        members (asserted at module load).
    I5. HTTP status precedence: a 4xx/5xx with class-name fingerprint
        for CONTEXT_OVERFLOW / CONTENT_FILTERED / OUTPUT_TRUNCATED wins
        over the bare status classification; otherwise status wins over
        generic class-name fingerprints.
    I6. Cause-chain priority: a wrapper exception with a more
        informative cause classifies by the cause, not the wrapper.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from enum import StrEnum
import math
import re
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

# ============================================================================
# Outcome enums
# ============================================================================


class LLMCallOutcome(StrEnum):
    """Terminal outcome of one LLM call attempt. Exhaustive over the failure
    modes the router stack currently observes against openai 2.x,
    httpx 0.28.x, langchain-openai 1.2.x, langchain-core 1.4.x, and
    litellm 1.x (rule tables include the litellm class names that do not
    inherit a known base — see ``_CONTEXT_OVERFLOW_MRO_NAMES`` and
    ``_CONTENT_FILTER_MRO_NAMES`` comments)."""

    SUCCESS = "success"
    RATE_LIMITED = "rate_limited"
    SERVER_5XX = "server_5xx"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    AUTH_FAILED = "auth_failed"
    BAD_REQUEST = "bad_request"
    CONTEXT_OVERFLOW = "context_overflow"
    OUTPUT_TRUNCATED = "output_truncated"
    CONTENT_FILTERED = "content_filtered"
    PARSE_FAILED = "parse_failed"
    OTHER_EXCEPTION = "other_exception"


class BanditAction(StrEnum):
    """Per-outcome bandit ledger reaction."""

    DEFER_TO_OUTCOME = "defer_to_outcome"
    INJECT_ZERO_REWARD = "inject_zero_reward"


# ============================================================================
# Outcome → action mapping (closed under LLMCallOutcome)
# ============================================================================


_OUTCOME_ACTION_DRAFT: dict[LLMCallOutcome, BanditAction] = {
    LLMCallOutcome.SUCCESS: BanditAction.DEFER_TO_OUTCOME,
    LLMCallOutcome.RATE_LIMITED: BanditAction.INJECT_ZERO_REWARD,
    LLMCallOutcome.SERVER_5XX: BanditAction.INJECT_ZERO_REWARD,
    LLMCallOutcome.TIMEOUT: BanditAction.INJECT_ZERO_REWARD,
    LLMCallOutcome.NETWORK_ERROR: BanditAction.INJECT_ZERO_REWARD,
    LLMCallOutcome.AUTH_FAILED: BanditAction.INJECT_ZERO_REWARD,
    LLMCallOutcome.BAD_REQUEST: BanditAction.INJECT_ZERO_REWARD,
    LLMCallOutcome.CONTEXT_OVERFLOW: BanditAction.INJECT_ZERO_REWARD,
    LLMCallOutcome.OUTPUT_TRUNCATED: BanditAction.INJECT_ZERO_REWARD,
    LLMCallOutcome.CONTENT_FILTERED: BanditAction.INJECT_ZERO_REWARD,
    LLMCallOutcome.PARSE_FAILED: BanditAction.INJECT_ZERO_REWARD,
    LLMCallOutcome.OTHER_EXCEPTION: BanditAction.INJECT_ZERO_REWARD,
}

OUTCOME_ACTION: Final[Mapping[LLMCallOutcome, BanditAction]] = MappingProxyType(
    _OUTCOME_ACTION_DRAFT
)

ZERO_REWARD_OUTCOMES: Final[frozenset[LLMCallOutcome]] = frozenset(
    outcome
    for outcome, action in OUTCOME_ACTION.items()
    if action is BanditAction.INJECT_ZERO_REWARD
)


# ============================================================================
# Cause-chain priority order — lower index wins
# ============================================================================
# Rationale per row:
#   RATE_LIMITED       — most actionable infra signal (back off)
#   AUTH_FAILED        — kill-switch: every arm will fail until the key is fixed
#   CONTEXT_OVERFLOW   — prompt-level, immediate; do not retry blindly
#   TIMEOUT            — infra-level transient
#   SERVER_5XX         — infra-level, possibly transient
#   NETWORK_ERROR      — infra-level, possibly transient
#   BAD_REQUEST        — request-level, retries unlikely to help
#   OUTPUT_TRUNCATED   — arm-level, prompt or budget tweak needed
#   CONTENT_FILTERED   — arm-level
#   PARSE_FAILED       — arm-level (wrapper of other failures often hits here)
#   OTHER_EXCEPTION    — unclassified
#   SUCCESS            — never appears in a cause walk; placed last for closure


_OUTCOME_PRIORITY: Final[tuple[LLMCallOutcome, ...]] = (
    LLMCallOutcome.RATE_LIMITED,
    LLMCallOutcome.AUTH_FAILED,
    LLMCallOutcome.CONTEXT_OVERFLOW,
    LLMCallOutcome.TIMEOUT,
    LLMCallOutcome.SERVER_5XX,
    LLMCallOutcome.NETWORK_ERROR,
    LLMCallOutcome.BAD_REQUEST,
    LLMCallOutcome.OUTPUT_TRUNCATED,
    LLMCallOutcome.CONTENT_FILTERED,
    LLMCallOutcome.PARSE_FAILED,
    LLMCallOutcome.OTHER_EXCEPTION,
    LLMCallOutcome.SUCCESS,
)

_OUTCOME_PRIORITY_INDEX: Final[Mapping[LLMCallOutcome, int]] = MappingProxyType(
    {outcome: i for i, outcome in enumerate(_OUTCOME_PRIORITY)}
)


# ============================================================================
# Module-load invariants
# ============================================================================
# These trip at import time if the tables fall out of sync with the enum —
# a regression that would otherwise leak into telemetry silently.


assert set(OUTCOME_ACTION) == set(LLMCallOutcome), (
    "OUTCOME_ACTION must cover every LLMCallOutcome member exactly once"
)
assert OUTCOME_ACTION[LLMCallOutcome.SUCCESS] is BanditAction.DEFER_TO_OUTCOME, (
    "SUCCESS must map to DEFER_TO_OUTCOME"
)
assert all(
    OUTCOME_ACTION[o] is BanditAction.INJECT_ZERO_REWARD
    for o in LLMCallOutcome
    if o is not LLMCallOutcome.SUCCESS
), "Every non-SUCCESS outcome must map to INJECT_ZERO_REWARD"
assert set(_OUTCOME_PRIORITY) == set(LLMCallOutcome), (
    "_OUTCOME_PRIORITY must be a permutation of LLMCallOutcome members"
)
assert len(_OUTCOME_PRIORITY) == len(LLMCallOutcome), (
    "_OUTCOME_PRIORITY must contain each outcome exactly once"
)


def _assert_disjoint(
    *named_sets: tuple[str, frozenset[str]],
) -> None:
    """Module-load guard: every class-name set must be pairwise disjoint so a
    single class never classifies as two outcomes."""
    seen: dict[str, str] = {}
    for set_name, names in named_sets:
        for name in names:
            if name in seen:
                raise AssertionError(
                    f"Class name {name!r} appears in both {seen[name]!r} and "
                    f"{set_name!r} — rule tables must be disjoint"
                )
            seen[name] = set_name


# ============================================================================
# Classification rule tables (private)
# ============================================================================
# Class-name sets are matched against the FULL MRO chain of the exception
# (``type(exc).__mro__``), so a subclass of a known base is still recognised
# without naming the leaf. All class names below have been verified against
# the installed package sources for openai 2.36.0, httpx 0.28.1,
# langchain-openai 1.2.1, langchain-core 1.4.0.


# HTTP status → outcome (explicit codes only; range fallthroughs handled in code)
_STATUS_TO_OUTCOME: Final[Mapping[int, LLMCallOutcome]] = MappingProxyType(
    {
        401: LLMCallOutcome.AUTH_FAILED,
        403: LLMCallOutcome.AUTH_FAILED,
        408: LLMCallOutcome.TIMEOUT,
        413: LLMCallOutcome.CONTEXT_OVERFLOW,
        429: LLMCallOutcome.RATE_LIMITED,
        504: LLMCallOutcome.TIMEOUT,
    }
)

# Context-overflow-specific exception class names (openai + langchain-core + litellm).
# Verified MROs:
#   OpenAIContextOverflowError      → BadRequestError → APIStatusError → ContextOverflowError
#   OpenAIAPIContextOverflowError   → APIError → ContextOverflowError
#   ContextOverflowError            → LangChainException
#   litellm.ContextWindowExceededError → BadRequestError → APIStatusError (status_code=400)
#     — litellm does NOT inherit from langchain's ContextOverflowError, so the
#     class name must appear here explicitly; otherwise the 400 status would
#     misclassify as BAD_REQUEST.
_CONTEXT_OVERFLOW_MRO_NAMES: Final[frozenset[str]] = frozenset(
    {
        "ContextOverflowError",  # langchain_core base
        "OpenAIContextOverflowError",  # langchain-openai
        "OpenAIAPIContextOverflowError",  # langchain-openai
        "ContextWindowExceededError",  # litellm
    }
)

# Successful call but output cut off at the token budget.
#   LengthFinishReasonError → OpenAIError (openai 2.x)
_OUTPUT_TRUNCATED_MRO_NAMES: Final[frozenset[str]] = frozenset(
    {
        "LengthFinishReasonError",
    }
)

# Refusal / content filter / moderation.
#   OpenAIRefusalError                 → Exception
#   OpenAIModerationError              → RuntimeError
#   ContentFilterFinishReasonError     → OpenAIError
#   litellm.ContentPolicyViolationError → BadRequestError (status_code=400)
#     — provider-side refusal surfaced as a 400; without this entry it would
#     fall through to BAD_REQUEST and lose the "arm-level refusal" signal.
_CONTENT_FILTER_MRO_NAMES: Final[frozenset[str]] = frozenset(
    {
        "OpenAIRefusalError",
        "OpenAIModerationError",
        "ContentFilterFinishReasonError",
        "ContentPolicyViolationError",  # litellm
    }
)

# Timeout exception class names. ``TimeoutError`` (builtin / 3.11+ alias for
# ``asyncio.TimeoutError``) is also covered via isinstance below.
#   httpx.TimeoutException, ConnectTimeout, ReadTimeout, WriteTimeout, PoolTimeout
#   openai.APITimeoutError
#   langchain_openai.StreamChunkTimeoutError (extends builtin TimeoutError)
_TIMEOUT_MRO_NAMES: Final[frozenset[str]] = frozenset(
    {
        "TimeoutError",
        "TimeoutException",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "APITimeoutError",
        "StreamChunkTimeoutError",
    }
)

# Network exception class names. ``ConnectionError`` builtin caught via isinstance.
#   httpx.NetworkError, ConnectError, ReadError, WriteError, CloseError,
#   ProtocolError, LocalProtocolError, RemoteProtocolError, ProxyError,
#   UnsupportedProtocol
#   openai.APIConnectionError
_NETWORK_MRO_NAMES: Final[frozenset[str]] = frozenset(
    {
        "ConnectionError",
        "NetworkError",
        "ConnectError",
        "ReadError",
        "WriteError",
        "CloseError",
        "ProtocolError",
        "LocalProtocolError",
        "RemoteProtocolError",
        "ProxyError",
        "UnsupportedProtocol",
        "APIConnectionError",
        "ConnectionRefusedError",
        "ConnectionResetError",
    }
)

# Parser / decoder failures.
#   pydantic.ValidationError                          → ValueError
#   json.JSONDecodeError                              → ValueError
#   langchain_core.exceptions.OutputParserException   → ValueError, LangChainException
#   openai.APIResponseValidationError                 → APIError (carries status_code)
#   httpx.DecodingError                               → RequestError
_PARSE_MRO_NAMES: Final[frozenset[str]] = frozenset(
    {
        "ValidationError",
        "JSONDecodeError",
        "OutputParserException",
        "APIResponseValidationError",
        "DecodingError",
    }
)

# Free-text markers for context overflow when no explicit class fired. These
# are common error-message fragments emitted by providers for input that
# exceeds the model context window.
_CONTEXT_OVERFLOW_MARKERS: Final[tuple[str, ...]] = (
    "context length",
    "context window",
    "context_length_exceeded",
    "maximum context length",
    "prompt is too long",
    "string too long",
    "input is too long",
    "reduce the length",
)

_assert_disjoint(
    ("_CONTEXT_OVERFLOW_MRO_NAMES", _CONTEXT_OVERFLOW_MRO_NAMES),
    ("_OUTPUT_TRUNCATED_MRO_NAMES", _OUTPUT_TRUNCATED_MRO_NAMES),
    ("_CONTENT_FILTER_MRO_NAMES", _CONTENT_FILTER_MRO_NAMES),
    ("_TIMEOUT_MRO_NAMES", _TIMEOUT_MRO_NAMES),
    ("_NETWORK_MRO_NAMES", _NETWORK_MRO_NAMES),
    ("_PARSE_MRO_NAMES", _PARSE_MRO_NAMES),
)


# ============================================================================
# Result schema
# ============================================================================


class LLMCallResult(BaseModel):
    """Pydantic record of a single LLM call result. Pure data — no behavior.

    Frozen + extra-forbidden + strict so the schema is a closed contract
    that telemetry and the bandit can rely on without defensive copying.

    Field semantics:
        ``outcome``              classified outcome (exhaustive over
                                 ``LLMCallOutcome``)
        ``exception_class``      ``type(exc).__name__`` of the surface
                                 exception. ``None`` only when ``outcome
                                 is SUCCESS``.
        ``exception_module``     full module path of the surface exception.
                                 ``None`` only when ``outcome is SUCCESS`` or
                                 ``__module__`` could not be coerced to ``str``.
        ``http_status``          status code recovered from the exception, if
                                 any. ``None`` means "no status was present on
                                 the exception" — NOT "status was zero".
        ``retry_after_seconds``  Retry-After header value when the response
                                 carries one and the parse succeeds. Three
                                 distinct values:
                                   * ``None`` — header absent / malformed /
                                     non-finite / above the 24-hour cap; PR-B
                                     callers should treat this as "no hint,
                                     use default backoff".
                                   * ``0.0`` — header present and equal to
                                     zero; provider explicitly said "retry
                                     immediately". PR-B callers MUST NOT
                                     conflate this with ``None``.
                                   * positive float — recommended sleep in
                                     seconds, bounded by ``[0.0, 86400.0]``.
        ``message``              ``str(exc)`` (or ``None`` when ``__str__``
                                 raises or returns empty). UTF-16 surrogates
                                 are scrubbed to ``U+FFFD`` so the field is
                                 always safe for ``model_dump_json`` and
                                 downstream JSON consumers (langfuse, tracker
                                 writes).
        ``cause_chain``          Tuple of class names walked through
                                 ``__cause__`` / ``__context__`` with cycle
                                 protection. Index ``0`` is the SURFACE
                                 exception's class name; index ``1``+ are
                                 (cause OR context, cause preferred) frames
                                 walked from the surface outward. The tuple
                                 is non-empty for every failure outcome and
                                 empty only for ``SUCCESS``.
        ``model_name``           Bandit-arm name attached by the caller for
                                 telemetry. Never inspected by the classifier
                                 — PR-B passes the routed model identifier
                                 here so the result can be correlated with
                                 the arm that produced it.

    PR-B integration contract:
        Callers wrap an LLM call in ``try``/``except BaseException``, pass
        the caught exception (or ``None`` on success) to
        ``classify_call_result``, then branch on ``result.action``:
        ``DEFER_TO_OUTCOME`` (success — reward arrives via
        ``on_mutation_outcome``) versus ``INJECT_ZERO_REWARD`` (any
        failure). For retry / backoff decisions, callers branch on
        ``result.outcome`` (rate-limited vs auth-failed vs parse-failed
        all differ) and use ``result.retry_after_seconds`` when set.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    outcome: LLMCallOutcome
    exception_class: str | None = None
    exception_module: str | None = None
    http_status: int | None = Field(default=None, ge=100, le=599)
    # 24-hour upper bound: matches no real provider policy but stops a
    # misbehaving upstream from parking the bandit on a multi-year sleep.
    retry_after_seconds: float | None = Field(default=None, ge=0.0, le=86400.0)
    message: str | None = None
    cause_chain: tuple[str, ...] = ()
    model_name: str | None = None

    @property
    def action(self) -> BanditAction:
        """Bandit ledger reaction for this outcome — pure lookup."""
        return OUTCOME_ACTION[self.outcome]

    @property
    def is_failure(self) -> bool:
        return self.outcome is not LLMCallOutcome.SUCCESS


# ============================================================================
# Public API
# ============================================================================


def classify_call_result(
    exc: BaseException | None,
    *,
    model_name: str | None = None,
) -> LLMCallResult:
    """Total classifier from ``(exc, model_name)`` to ``LLMCallResult``.

    ``exc is None`` ⇔ outcome is ``SUCCESS``. For any non-None exception
    the result's outcome is one of the eleven failure variants.

    Args:
        exc: the exception raised by the LLM call, or ``None`` for success.
        model_name: optional bandit-arm name attached to the result for
            telemetry — never inspected by the classifier itself.

    Returns:
        Frozen ``LLMCallResult`` whose ``outcome`` is exhaustive over
        ``LLMCallOutcome``.
    """
    if exc is None:
        return LLMCallResult(outcome=LLMCallOutcome.SUCCESS, model_name=model_name)

    outcome = _walk_and_classify(exc)
    exc_name, exc_module = _safe_class_metadata(exc)
    return LLMCallResult(
        outcome=outcome,
        exception_class=exc_name,
        exception_module=exc_module,
        http_status=_extract_status(exc),
        retry_after_seconds=_extract_retry_after(exc),
        message=_safe_message(exc),
        cause_chain=_collect_cause_chain(exc),
        model_name=model_name,
    )


# ============================================================================
# Internals
# ============================================================================


def _iter_cause_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield ``exc`` and walk ``__cause__`` / ``__context__`` with cycle
    protection. ``__cause__`` wins over ``__context__`` (explicit
    ``raise X from Y`` is more informative than the implicit context).

    ``__cause__`` / ``__context__`` are normally C-level slots on
    ``BaseException``, but a subclass may override them with a descriptor
    that raises (legal Python). Access is funnelled through
    ``_safe_getattr`` so a hostile property cannot break classifier
    totality. A non-BaseException value (reachable via property overrides
    that return arbitrary objects) terminates the walk rather than
    poisoning the cycle-detection set."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cause = _safe_getattr(cur, "__cause__")
        nxt = cause if cause is not None else _safe_getattr(cur, "__context__")
        cur = nxt if isinstance(nxt, BaseException) else None


def _walk_and_classify(exc: BaseException) -> LLMCallOutcome:
    """Classify each exception in the cause chain and return the
    highest-priority outcome — ensures a wrapper exception cannot mask
    a more informative root cause."""
    outcomes = [_classify_one(frame) for frame in _iter_cause_chain(exc)]
    return min(outcomes, key=_OUTCOME_PRIORITY_INDEX.__getitem__)


def _classify_one(exc: BaseException) -> LLMCallOutcome:
    """Classify a single exception in isolation. Returns OTHER_EXCEPTION
    when no rule matches."""
    mro_names = _mro_class_names(exc)
    message = _safe_message(exc) or ""

    # 1. MRO-based specific overrides (more informative than HTTP status).
    if mro_names & _CONTEXT_OVERFLOW_MRO_NAMES:
        return LLMCallOutcome.CONTEXT_OVERFLOW
    if mro_names & _CONTENT_FILTER_MRO_NAMES:
        return LLMCallOutcome.CONTENT_FILTERED
    if mro_names & _OUTPUT_TRUNCATED_MRO_NAMES:
        return LLMCallOutcome.OUTPUT_TRUNCATED

    # 2. HTTP status (authoritative for well-defined codes).
    status = _extract_status(exc)
    if status is not None:
        explicit = _STATUS_TO_OUTCOME.get(status)
        if explicit is not None:
            return explicit
        if 400 <= status < 500:
            # Catch-all for 400 / 404 / 409 / 422 / etc. that did not get
            # a more specific override above. Check the overflow text marker
            # before defaulting to BAD_REQUEST — some providers return 400
            # with no dedicated exception class for context overflow.
            if _matches_overflow_marker(message):
                return LLMCallOutcome.CONTEXT_OVERFLOW
            return LLMCallOutcome.BAD_REQUEST
        if 500 <= status < 600:
            return LLMCallOutcome.SERVER_5XX
        if 100 <= status < 400:
            # Defensive: exception that surfaced despite a success-ish status
            # almost always comes from a downstream parser stage.
            return LLMCallOutcome.PARSE_FAILED

    # 3. MRO-based class fingerprints (subclass-aware via full MRO walk).
    if mro_names & _TIMEOUT_MRO_NAMES:
        return LLMCallOutcome.TIMEOUT
    if mro_names & _NETWORK_MRO_NAMES:
        return LLMCallOutcome.NETWORK_ERROR
    if mro_names & _PARSE_MRO_NAMES:
        return LLMCallOutcome.PARSE_FAILED

    # 4. Built-in hierarchy fallback. ``TimeoutError`` aliases
    # ``asyncio.TimeoutError`` since 3.11 and is also the base for
    # ``StreamChunkTimeoutError``. ``ConnectionError`` covers
    # ``ConnectionRefusedError``, ``ConnectionResetError``,
    # ``ConnectionAbortedError``.
    if isinstance(exc, TimeoutError):
        return LLMCallOutcome.TIMEOUT
    if isinstance(exc, ConnectionError):
        return LLMCallOutcome.NETWORK_ERROR

    # 5. Free-text context-overflow marker (last resort).
    if _matches_overflow_marker(message):
        return LLMCallOutcome.CONTEXT_OVERFLOW

    return LLMCallOutcome.OTHER_EXCEPTION


def _mro_class_names(exc: BaseException) -> frozenset[str]:
    """Names of every class in the exception's MRO, frozen for set ops.

    A metaclass may override ``__mro__`` with a property that raises
    (legal Python). The classifier's totality contract forbids letting
    that propagate, so the access is wrapped and a hostile MRO collapses
    to an empty fingerprint set — every MRO-based branch in
    ``_classify_one`` then naturally skips, and the exception falls
    through to the status-code / isinstance / OTHER paths."""
    mro: object = _safe_getattr(type(exc), "__mro__")
    if mro is None:
        return frozenset()
    try:
        return frozenset(cls.__name__ for cls in mro)  # type: ignore[union-attr]
    except Exception:
        return frozenset()


def _safe_str_attr(obj: object, attr: str) -> str | None:
    """Return ``getattr(obj, attr)`` coerced to ``str`` with paranoid
    fallback. A metaclass that sets ``__module__`` to an int (legal in
    Python) would otherwise inject a non-str into the result schema and
    raise ``ValidationError`` from inside the classifier, violating the
    documented totality contract."""
    raw = _safe_getattr(obj, attr)
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    try:
        return str(raw)
    except Exception:
        return None


def _safe_class_metadata(exc: BaseException) -> tuple[str | None, str | None]:
    """Return ``(name, module)`` for ``type(exc)`` with defensive coercion."""
    cls = type(exc)
    return _safe_str_attr(cls, "__name__"), _safe_str_attr(cls, "__module__")


def _safe_getattr(obj: object, name: str) -> object | None:
    """``getattr(obj, name, None)`` that also suppresses non-AttributeError
    exceptions raised by hostile property/descriptor implementations.

    The stdlib ``getattr(..., default)`` only catches AttributeError; if a
    property's getter raises (e.g.) ``RuntimeError``, the call propagates.
    Provider SDK exception classes are well-behaved, but we touch attributes
    on whatever object the caller hands us, so the classifier must not
    crash on a misbehaving third-party descriptor."""
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


def _extract_status(exc: BaseException) -> int | None:
    """Recover an HTTP status code from common SDK attributes.

    ``type(x) is int`` rather than ``isinstance(x, int)`` so a ``bool``
    masquerading as int (Python quirk) is rejected. Restricts to the
    100-599 HTTP range so a stray ``status_code = -1`` sentinel is also
    rejected."""
    candidate = _safe_getattr(exc, "status_code")
    if type(candidate) is int and 100 <= candidate <= 599:
        return candidate
    response = _safe_getattr(exc, "response")
    if response is not None:
        rs = _safe_getattr(response, "status_code")
        if type(rs) is int and 100 <= rs <= 599:
            return rs
    return None


def _extract_retry_after(exc: BaseException) -> float | None:
    """Extract a Retry-After header value (seconds) from the exception's
    response, if present. Returns ``None`` when the header is missing,
    malformed, or the response object lacks a ``headers`` attribute.

    Only handles the integer-seconds form. HTTP-date form is not
    recognised — callers should treat ``None`` as "no hint available"."""
    response = _safe_getattr(exc, "response")
    if response is None:
        return None
    headers = _safe_getattr(response, "headers")
    if headers is None:
        return None
    get = _safe_getattr(headers, "get")
    if not callable(get):
        return None
    # httpx Headers is case-insensitive; a plain dict isn't, so try both.
    try:
        raw: object = get("retry-after")
        if raw is None:
            raw = get("Retry-After")
    except Exception:
        return None
    if raw is None or isinstance(raw, bool):
        return None
    if type(raw) is int or type(raw) is float:
        value = float(raw)
    elif isinstance(raw, str):
        try:
            value = float(raw)
        except ValueError:
            return None
    else:
        return None
    # Reject non-finite values: ``"inf"`` / ``"Infinity"`` / ``"nan"`` /
    # numbers that overflow ``float`` would otherwise propagate as
    # ``math.inf`` or ``math.nan`` into the bandit's sleep budget and
    # diverge from telemetry (JSON serializers emit ``null`` for
    # non-finite floats, so the in-memory value and the wire value
    # would disagree silently). The 24-hour cap matches the schema's
    # ``le=86400`` and is enforced here so that values past the cap
    # become ``None`` rather than crashing classifier totality at the
    # ``LLMCallResult`` validation step.
    if not math.isfinite(value) or value < 0.0 or value > _MAX_RETRY_AFTER_SECONDS:
        return None
    return value


_MAX_RETRY_AFTER_SECONDS: Final[float] = 86400.0


def _safe_message(exc: BaseException) -> str | None:
    """``str(exc)`` with paranoid handling of pathological ``__str__``.

    UTF-16 surrogate code points (``U+D800``-``U+DFFF``) are replaced with
    ``U+FFFD``. Python ``str`` permits surrogates but UTF-8 encoders and
    JSON serializers refuse them; without this scrub the classifier could
    return a result whose ``message`` field crashes ``model_dump_json``
    downstream. Provider response messages occasionally include broken
    surrogates from misbehaving tokenizers or sliced multibyte echoes.
    """
    try:
        text = str(exc)
    except Exception:
        return None
    if not text:
        return None
    return _strip_surrogates(text)


def _strip_surrogates(text: str) -> str:
    """Replace every UTF-16 surrogate code point with ``U+FFFD``.

    Replaces unconditionally rather than discriminating "lone" from "valid
    pair": Python ``str`` is sequence-of-code-points, not UTF-16, so a
    high+low pair is two independent code points that UTF-8 still rejects,
    not one astral character. Real astral characters live in a single code
    point above ``U+FFFF`` and never appear as a surrogate pair in Python
    ``str``."""
    return _SURROGATE_RE.sub("�", text)


_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _matches_overflow_marker(message: str) -> bool:
    """``True`` when *message* contains a context-overflow marker substring."""
    if not message:
        return False
    lowered = message.lower()
    return any(marker in lowered for marker in _CONTEXT_OVERFLOW_MARKERS)


def _collect_cause_chain(exc: BaseException) -> tuple[str, ...]:
    """Class names of every exception in the cause chain."""
    return tuple(type(frame).__name__ for frame in _iter_cause_chain(exc))


# ============================================================================
# Public invariant documentation (programmatic access for tests)
# ============================================================================


INVARIANTS: Final[tuple[str, ...]] = (
    "classify_call_result(None).outcome is SUCCESS; no other input produces SUCCESS.",
    "OUTCOME_ACTION is closed under LLMCallOutcome — every member is a key.",
    "SUCCESS is the only outcome with action DEFER_TO_OUTCOME; every other "
    "outcome has action INJECT_ZERO_REWARD.",
    "_OUTCOME_PRIORITY is a permutation of LLMCallOutcome members.",
    "HTTP status precedence: a 4xx/5xx with class-name fingerprint for "
    "CONTEXT_OVERFLOW / CONTENT_FILTERED / OUTPUT_TRUNCATED wins over the "
    "bare status classification; otherwise status wins over generic "
    "class-name fingerprints.",
    "Cause-chain priority: a wrapper exception with a more informative "
    "cause classifies by the cause, not the wrapper.",
    "The classifier never imports a provider SDK (openai, httpx, langchain-*, "
    "pydantic). It inspects MRO class names and exception attributes only.",
    "LLMCallResult is frozen, extra-forbidden, and strict — the result is a "
    "closed contract.",
)


__all__ = (
    "BanditAction",
    "INVARIANTS",
    "LLMCallOutcome",
    "LLMCallResult",
    "OUTCOME_ACTION",
    "ZERO_REWARD_OUTCOMES",
    "classify_call_result",
)
