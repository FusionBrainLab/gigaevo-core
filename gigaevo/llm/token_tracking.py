import threading
from typing import Annotated, Any

from loguru import logger
from pydantic import BaseModel, Field, SkipValidation

from gigaevo.utils.trackers.base import LogWriter


class TokenUsage(BaseModel):
    """Token counts for a single LLM call."""

    context: int = 0
    generated: int = 0
    reasoning: int = 0  # Reasoning tokens (subset of generated, for thinking models)
    total: int = 0

    @classmethod
    def from_response(cls, response: Any) -> "TokenUsage | None":
        """Extract token usage from LLM response metadata."""
        if not hasattr(response, "response_metadata") or not response.response_metadata:
            return None

        usage = response.response_metadata.get(
            "token_usage"
        ) or response.response_metadata.get("usage")
        if not usage:
            return None

        # Extract reasoning tokens - try multiple possible field names/structures
        reasoning = 0
        # OpenAI o1/o3 style: completion_tokens_details.reasoning_tokens
        if details := usage.get("completion_tokens_details"):
            reasoning = details.get("reasoning_tokens", 0) or 0
        # Direct field (some providers)
        if not reasoning:
            reasoning = usage.get("reasoning_tokens", 0) or 0
        # Qwen/thinking models might use different names
        if not reasoning:
            reasoning = usage.get("thinking_tokens", 0) or 0

        return cls(
            context=usage.get("prompt_tokens", 0),
            generated=usage.get("completion_tokens", 0),
            reasoning=reasoning,
            total=usage.get("total_tokens", 0),
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
