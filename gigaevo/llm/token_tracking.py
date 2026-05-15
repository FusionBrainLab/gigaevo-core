import threading
from typing import Annotated, Any

from loguru import logger
from pydantic import BaseModel, Field, SkipValidation

from gigaevo.utils.trackers.base import LogWriter


def _coerce_int(value: Any) -> int:
    """Coerce a token-count value to ``int``, defaulting to ``0``.

    Hostile providers occasionally return ``None``, strings, floats with
    non-integer fractions, or arbitrary objects in their ``usage`` payload.
    Token counts are summed downstream into a Pydantic ``int`` field
    (``TokenUsage.cumulative``), so an uncoerced non-int would raise on the
    first ``cum.context += usage.context`` add. Coerce defensively and
    swallow conversion errors as ``0`` — token telemetry is observability-
    only and must never crash the LLM call site.
    """
    if isinstance(value, bool):
        # ``bool`` is a subclass of ``int`` — accept silently as 0/1 for
        # the rare provider that returns a flag instead of a count.
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        try:
            return int(value)
        except (ValueError, OverflowError):
            return 0
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


class TokenUsage(BaseModel):
    """Token counts for a single LLM call."""

    context: int = 0
    generated: int = 0
    reasoning: int = 0  # Reasoning tokens (subset of generated, for thinking models)
    total: int = 0

    @classmethod
    def from_response(cls, response: Any) -> "TokenUsage | None":
        """Extract token usage from LLM response metadata.

        Defensive against hostile / malformed payloads: a provider that
        returns a string (or any non-dict) in ``completion_tokens_details``,
        ``token_usage``, or ``usage`` must not propagate an ``AttributeError``
        into the call site. The bandit's ``_safe_track`` already swallows
        such failures, but direct callers (``MultiModelRouter`` in
        ``gigaevo/llm/models.py``) have no second-level guard, so the
        hardening lives here.
        """
        metadata = getattr(response, "response_metadata", None)
        if not metadata or not isinstance(metadata, dict):
            return None

        usage = metadata.get("token_usage") or metadata.get("usage")
        if not usage or not isinstance(usage, dict):
            return None

        # Extract reasoning tokens - try multiple possible field names/structures.
        # Each branch tolerates non-dict / non-int values from hostile providers.
        reasoning = 0
        # OpenAI o1/o3 style: completion_tokens_details.reasoning_tokens
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict):
            reasoning = _coerce_int(details.get("reasoning_tokens"))
        # Direct field (some providers)
        if not reasoning:
            reasoning = _coerce_int(usage.get("reasoning_tokens"))
        # Qwen/thinking models might use different names
        if not reasoning:
            reasoning = _coerce_int(usage.get("thinking_tokens"))

        return cls(
            context=_coerce_int(usage.get("prompt_tokens")),
            generated=_coerce_int(usage.get("completion_tokens")),
            reasoning=reasoning,
            total=_coerce_int(usage.get("total_tokens")),
        )


class TokenTracker(BaseModel):
    """Tracks per-call and cumulative token usage per model. Thread-safe."""

    model_config = {"arbitrary_types_allowed": True}

    name: str = "default"
    writer: LogWriter | None = None
    cumulative: dict[str, TokenUsage] = Field(default_factory=dict)
    lock: Annotated[threading.Lock, SkipValidation] = Field(
        default_factory=threading.Lock, exclude=True
    )

    def track(self, response: Any, model_name: str) -> None:
        """Track token usage from LLM response. Thread-safe."""
        if self.writer is None:
            return

        usage = TokenUsage.from_response(response)
        if usage is None:
            logger.debug(
                "[TokenTracker:{}] No token usage for {}", self.name, model_name
            )
            return

        with self.lock:
            if model_name not in self.cumulative:
                self.cumulative[model_name] = TokenUsage()
            cum = self.cumulative[model_name]
            cum.context += usage.context
            cum.generated += usage.generated
            cum.reasoning += usage.reasoning
            cum.total += usage.total

            self._write_metrics(model_name, usage, cum)

    def _write_metrics(
        self, model_name: str, usage: TokenUsage, cumulative: TokenUsage
    ) -> None:
        """Write per-call and cumulative metrics."""
        path = [self.name, model_name.replace("/", "_").replace(":", "_")]

        if self.writer is None:
            return
        self.writer.scalar("context_tokens", float(usage.context), path=path)
        self.writer.scalar("generated_tokens", float(usage.generated), path=path)
        self.writer.scalar("reasoning_tokens", float(usage.reasoning), path=path)
        self.writer.scalar("total_tokens", float(usage.total), path=path)

        self.writer.scalar(
            "cumulative_context_tokens", float(cumulative.context), path=path
        )
        self.writer.scalar(
            "cumulative_generated_tokens", float(cumulative.generated), path=path
        )
        self.writer.scalar(
            "cumulative_reasoning_tokens", float(cumulative.reasoning), path=path
        )
        self.writer.scalar(
            "cumulative_total_tokens", float(cumulative.total), path=path
        )

        logger.debug(
            "[TokenTracker:{}] {}: {} ctx + {} gen ({} reasoning) = {} (cumulative: {})",
            self.name,
            model_name,
            usage.context,
            usage.generated,
            usage.reasoning,
            usage.total,
            cumulative.total,
        )
