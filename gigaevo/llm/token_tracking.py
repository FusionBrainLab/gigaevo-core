from collections.abc import Iterator
import contextlib
from contextvars import ContextVar
import threading
from typing import Annotated, Any

from loguru import logger
from pydantic import BaseModel, Field, SkipValidation

from gigaevo.utils.trackers.base import LogWriter

# ContextVar set by LangGraphAgent.acall_llm around `await llm.ainvoke(...)`
# so the shared MultiModelRouter can attribute tokens back to the calling
# stage (InsightsAgent / LineageAgent / MutationAgent / ...). Lives here
# (not in models.py) because models.py imports from this module — putting
# it in models.py would create a cycle.
_current_llm_stage_var: ContextVar[str | None] = ContextVar(
    "current_llm_stage", default=None
)

# Sentinel used when track() is called outside any agent context (e.g. a
# direct router invocation in a script). Tokens still book — they just land
# under this bucket so a per-stage TB plot has a visible "where did these
# come from?" series instead of silently dropping the call.
_UNATTRIBUTED_STAGE = "unattributed"


def get_current_llm_stage() -> str | None:
    """Return the LLM stage name set by the surrounding ``llm_stage_context``."""
    return _current_llm_stage_var.get()


@contextlib.contextmanager
def llm_stage_context(stage: str) -> Iterator[None]:
    """Set the current LLM stage for token-attribution within the block.

    The shared MultiModelRouter is the same object across InsightsAgent,
    LineageAgent, and MutationAgent, so per-stage attribution must happen
    via runtime context — not router-time naming. Each agent wraps its
    ``await llm.ainvoke(...)`` with this manager and the resulting
    ``TokenTracker.track`` call reads the var to bucket tokens by stage.
    """
    token = _current_llm_stage_var.set(stage)
    try:
        yield
    finally:
        _current_llm_stage_var.reset(token)


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
    """Tracks per-call and cumulative token usage per model. Thread-safe.

    Two cumulative buckets:

    - ``cumulative`` keyed by ``model_name`` — the legacy aggregate view that
      downstream readers (live profiler, retros) already depend on.
    - ``cumulative_by_stage`` keyed by ``(stage, model_name)`` — the new
      per-stage breakdown driven by ``llm_stage_context()``. The stage is
      read from ``get_current_llm_stage()`` at ``track()`` time; if no
      context is active, tokens land under ``_UNATTRIBUTED_STAGE`` so
      they're never silently lost.
    """

    model_config = {"arbitrary_types_allowed": True}

    name: str = "default"
    writer: LogWriter | None = None
    cumulative: dict[str, TokenUsage] = Field(default_factory=dict)
    cumulative_by_stage: dict[tuple[str, str], TokenUsage] = Field(default_factory=dict)
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

        stage = get_current_llm_stage() or _UNATTRIBUTED_STAGE

        with self.lock:
            if model_name not in self.cumulative:
                self.cumulative[model_name] = TokenUsage()
            cum = self.cumulative[model_name]
            cum.context += usage.context
            cum.generated += usage.generated
            cum.reasoning += usage.reasoning
            cum.total += usage.total

            stage_key = (stage, model_name)
            if stage_key not in self.cumulative_by_stage:
                self.cumulative_by_stage[stage_key] = TokenUsage()
            stage_cum = self.cumulative_by_stage[stage_key]
            stage_cum.context += usage.context
            stage_cum.generated += usage.generated
            stage_cum.reasoning += usage.reasoning
            stage_cum.total += usage.total

            self._write_metrics(model_name, usage, cum)
            self._write_stage_metrics(stage, model_name, usage, stage_cum)

    def _safe_model(self, model_name: str) -> str:
        return model_name.replace("/", "_").replace(":", "_")

    def _write_metrics(
        self, model_name: str, usage: TokenUsage, cumulative: TokenUsage
    ) -> None:
        """Write per-call and cumulative metrics."""
        path = [self.name, self._safe_model(model_name)]

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

    def _write_stage_metrics(
        self,
        stage: str,
        model_name: str,
        usage: TokenUsage,
        stage_cumulative: TokenUsage,
    ) -> None:
        """Write per-stage scalars at ``[name, stage, safe_model]`` so each
        agent gets its own TensorBoard panel for cost/usage.
        """
        if self.writer is None:
            return
        path = [self.name, stage, self._safe_model(model_name)]

        self.writer.scalar("context_tokens", float(usage.context), path=path)
        self.writer.scalar("generated_tokens", float(usage.generated), path=path)
        self.writer.scalar("reasoning_tokens", float(usage.reasoning), path=path)
        self.writer.scalar("total_tokens", float(usage.total), path=path)

        self.writer.scalar(
            "cumulative_context_tokens",
            float(stage_cumulative.context),
            path=path,
        )
        self.writer.scalar(
            "cumulative_generated_tokens",
            float(stage_cumulative.generated),
            path=path,
        )
        self.writer.scalar(
            "cumulative_reasoning_tokens",
            float(stage_cumulative.reasoning),
            path=path,
        )
        self.writer.scalar(
            "cumulative_total_tokens",
            float(stage_cumulative.total),
            path=path,
        )
