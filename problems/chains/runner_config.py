"""Chain runner configuration: feedback modes and execution modes.

Usage
-----
Chain execution behaviour is controlled by a ``RunnerConfig`` instance passed
to ``run_chain_on_dataset_stepwise``.  In production the config is read from
the ``GIGAEVO_CHAIN_RUNNER_CONFIG`` environment variable (JSON) so that it can
be set via the Hydra launcher without modifying validate.py.

Feedback modes
--------------
none     — no retry logic; single-pass execution (default).
simple   — rule-based retry: if an LLM step output matches one of the
           ``bad_patterns``, retry up to ``max_retries`` times with an
           appended feedback note.
dataset  — ground-truth retry: after the final step, samples whose output
           does not contain the expected answer (``sample[answer_key]``) get
           one additional LLM call that shows the expected format as a hint.
metrics  — quality-gate retry: a configurable ``metric_fn`` scores each LLM
           step output; outputs that score below ``threshold`` are retried.

Execution modes
---------------
fast        — standard single LLM call per step (default).
self_critic — generate → self-evaluate → regenerate loop: after each LLM
              step a second call evaluates the output (APPROVE / REJECT);
              rejected outputs are retried up to ``max_revisions`` times.

Setting the config at launch
-----------------------------
Hydra config group ``chains/runner`` (``config/chains/runner/``) holds named
presets.  The validate.py subprocess reads the env var::

    export GIGAEVO_CHAIN_RUNNER_CONFIG='{"feedback_mode":"simple","execution_mode":"fast"}'

Or with Hydra overrides::

    python run.py ... chains/runner=simple

The ``config/chains/runner/*.yaml`` files set ``GIGAEVO_CHAIN_RUNNER_CONFIG``
via the launcher's ``hydra.job.env_set`` mechanism.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
import json
import logging
import os
from typing import ClassVar

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FeedbackMode(StrEnum):
    NONE = "none"
    SIMPLE = "simple"
    DATASET = "dataset"
    METRICS = "metrics"


class StepExecutionMode(StrEnum):
    FAST = "fast"
    SELF_CRITIC = "self_critic"


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass
class SimpleRetryConfig:
    """Config for FeedbackMode.SIMPLE (pattern-based retry)."""

    bad_patterns: list[str] = field(
        default_factory=lambda: [
            "i don't know",
            "i cannot",
            "insufficient information",
            "not enough information",
            "unable to determine",
            "no information",
        ]
    )
    max_retries: int = 2
    feedback_message: str = (
        "Your previous response was not satisfactory. "
        "Provide a more specific and well-reasoned answer based on the available information."
    )
    case_sensitive: bool = False


@dataclass
class DatasetFeedbackConfig:
    """Config for FeedbackMode.DATASET (ground-truth reflection)."""

    answer_key: str = "answer"
    max_retries: int = 1
    feedback_template: str = (
        "Your previous answer did not match the expected format or content. "
        "Expected answer type: {expected}. "
        "Your answer: {actual}. "
        "Reconsider your reasoning and provide a more accurate answer."
    )


@dataclass
class MetricFeedbackConfig:
    """Config for FeedbackMode.METRICS (quality-gate retry).

    ``metric_fn`` is a callable ``(outputs: list[str]) -> list[float]`` that
    scores a batch of LLM step outputs.  If ``None``, a default length-based
    heuristic is used (outputs shorter than ``min_output_length`` tokens score 0).
    """

    threshold: float = 0.4
    max_retries: int = 2
    feedback_message: str = (
        "Your previous answer was too brief or lacked sufficient detail. "
        "Provide a more complete and specific response."
    )
    metric_fn: Callable[[list[str]], list[float]] | None = field(
        default=None, repr=False
    )
    min_output_length: int = 20  # fallback heuristic: word count


@dataclass
class SelfCriticConfig:
    """Config for StepExecutionMode.SELF_CRITIC."""

    max_revisions: int = 2
    evaluator_prompt_template: str = (
        "Evaluate the following reasoning step output.\n\n"
        "Step objective: {aim}\n"
        "Output to evaluate:\n{output}\n\n"
        "Does this output fully address the objective with specific, "
        "grounded reasoning and concrete details?\n"
        "Respond with exactly one of:\n"
        "  APPROVE — if the output is satisfactory\n"
        "  REJECT:<brief_reason> — if it is not (one sentence after the colon)"
    )
    disapprove_feedback_template: str = (
        "Your previous answer was rejected: {reason}. "
        "Please revise your response to address this critique."
    )


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class RunnerConfig:
    """Full chain runner configuration (feedback mode + execution mode)."""

    feedback_mode: FeedbackMode = FeedbackMode.NONE
    execution_mode: StepExecutionMode = StepExecutionMode.FAST

    simple_retry: SimpleRetryConfig = field(default_factory=SimpleRetryConfig)
    dataset_feedback: DatasetFeedbackConfig = field(
        default_factory=DatasetFeedbackConfig
    )
    metric_feedback: MetricFeedbackConfig = field(default_factory=MetricFeedbackConfig)
    self_critic: SelfCriticConfig = field(default_factory=SelfCriticConfig)

    _TOP_LEVEL_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "feedback_mode",
            "execution_mode",
            "simple_retry",
            "dataset_feedback",
            "metric_feedback",
            "self_critic",
        }
    )
    _SUB_KEYS: ClassVar[dict[str, frozenset[str]]] = {
        "simple_retry": frozenset(
            {"bad_patterns", "max_retries", "feedback_message", "case_sensitive"}
        ),
        "dataset_feedback": frozenset(
            {"answer_key", "max_retries", "feedback_template"}
        ),
        "metric_feedback": frozenset(
            {"threshold", "max_retries", "feedback_message", "min_output_length"}
        ),
        "self_critic": frozenset(
            {
                "max_revisions",
                "evaluator_prompt_template",
                "disapprove_feedback_template",
            }
        ),
    }

    @classmethod
    def from_dict(cls, data: dict, *, strict: bool = False) -> RunnerConfig:
        """Build from a plain dict (e.g. parsed from JSON env var).

        Unknown keys cause a warning by default (so misspelled YAML does not
        silently fall back to defaults — the exact bug class treatment-verifier
        was added to catch).  Set ``strict=True`` to raise ``ValueError``
        instead.
        """
        unknown_top = set(data) - cls._TOP_LEVEL_KEYS
        if unknown_top:
            msg = f"Unknown RunnerConfig keys: {sorted(unknown_top)}"
            if strict:
                raise ValueError(msg)
            _LOG.warning(msg)

        for sub_key, allowed in cls._SUB_KEYS.items():
            if sub_key in data and isinstance(data[sub_key], dict):
                unknown_sub = set(data[sub_key]) - allowed
                if unknown_sub:
                    msg = f"Unknown RunnerConfig[{sub_key}] keys: {sorted(unknown_sub)}"
                    if strict:
                        raise ValueError(msg)
                    _LOG.warning(msg)

        cfg = cls()
        if "feedback_mode" in data:
            cfg.feedback_mode = FeedbackMode(data["feedback_mode"])
        if "execution_mode" in data:
            cfg.execution_mode = StepExecutionMode(data["execution_mode"])

        if "simple_retry" in data:
            sub = data["simple_retry"]
            cfg.simple_retry = SimpleRetryConfig(
                bad_patterns=sub.get("bad_patterns", cfg.simple_retry.bad_patterns),
                max_retries=sub.get("max_retries", cfg.simple_retry.max_retries),
                feedback_message=sub.get(
                    "feedback_message", cfg.simple_retry.feedback_message
                ),
                case_sensitive=sub.get(
                    "case_sensitive", cfg.simple_retry.case_sensitive
                ),
            )
        if "dataset_feedback" in data:
            sub = data["dataset_feedback"]
            cfg.dataset_feedback = DatasetFeedbackConfig(
                answer_key=sub.get("answer_key", cfg.dataset_feedback.answer_key),
                max_retries=sub.get("max_retries", cfg.dataset_feedback.max_retries),
                feedback_template=sub.get(
                    "feedback_template", cfg.dataset_feedback.feedback_template
                ),
            )
        if "metric_feedback" in data:
            sub = data["metric_feedback"]
            cfg.metric_feedback = MetricFeedbackConfig(
                threshold=sub.get("threshold", cfg.metric_feedback.threshold),
                max_retries=sub.get("max_retries", cfg.metric_feedback.max_retries),
                feedback_message=sub.get(
                    "feedback_message", cfg.metric_feedback.feedback_message
                ),
                min_output_length=sub.get(
                    "min_output_length", cfg.metric_feedback.min_output_length
                ),
            )
        if "self_critic" in data:
            sub = data["self_critic"]
            cfg.self_critic = SelfCriticConfig(
                max_revisions=sub.get("max_revisions", cfg.self_critic.max_revisions),
                evaluator_prompt_template=sub.get(
                    "evaluator_prompt_template",
                    cfg.self_critic.evaluator_prompt_template,
                ),
                disapprove_feedback_template=sub.get(
                    "disapprove_feedback_template",
                    cfg.self_critic.disapprove_feedback_template,
                ),
            )
        return cfg

    @classmethod
    def from_env(cls, *, strict: bool = False) -> RunnerConfig:
        """Read config from ``GIGAEVO_CHAIN_RUNNER_CONFIG`` env var (JSON).

        Logs a warning and falls back to the default config (NONE / FAST) when
        the env var contains invalid JSON.  Pass ``strict=True`` to raise
        instead — recommended for experimental runs where a silent fallback
        would mask a treatment config bug.
        """
        raw = os.environ.get("GIGAEVO_CHAIN_RUNNER_CONFIG", "")
        if not raw:
            return cls()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            msg = (
                "GIGAEVO_CHAIN_RUNNER_CONFIG is not valid JSON "
                f"({exc.msg}) — falling back to defaults"
            )
            if strict:
                raise ValueError(msg) from exc
            _LOG.warning(msg)
            return cls()
        return cls.from_dict(data, strict=strict)

    def to_json(self) -> str:
        """Serialise to JSON string for setting the env var."""
        d: dict = {
            "feedback_mode": self.feedback_mode.value,
            "execution_mode": self.execution_mode.value,
        }
        if self.feedback_mode is FeedbackMode.SIMPLE:
            d["simple_retry"] = {
                "bad_patterns": self.simple_retry.bad_patterns,
                "max_retries": self.simple_retry.max_retries,
                "feedback_message": self.simple_retry.feedback_message,
                "case_sensitive": self.simple_retry.case_sensitive,
            }
        elif self.feedback_mode is FeedbackMode.DATASET:
            d["dataset_feedback"] = {
                "answer_key": self.dataset_feedback.answer_key,
                "max_retries": self.dataset_feedback.max_retries,
                "feedback_template": self.dataset_feedback.feedback_template,
            }
        elif self.feedback_mode is FeedbackMode.METRICS:
            d["metric_feedback"] = {
                "threshold": self.metric_feedback.threshold,
                "max_retries": self.metric_feedback.max_retries,
                "feedback_message": self.metric_feedback.feedback_message,
                "min_output_length": self.metric_feedback.min_output_length,
            }
            # metric_fn is a runtime callable and cannot be round-tripped
            # through JSON/env vars. Warn loudly so callers do not assume their
            # custom metric survives ``to_json() -> from_env()``.
            if self.metric_feedback.metric_fn is not None:
                _LOG.warning(
                    "RunnerConfig.to_json() is dropping metric_feedback.metric_fn "
                    "(callables cannot be serialised). The default length "
                    "heuristic will be used after round-trip."
                )
        if self.execution_mode is StepExecutionMode.SELF_CRITIC:
            d["self_critic"] = {
                "max_revisions": self.self_critic.max_revisions,
                "evaluator_prompt_template": self.self_critic.evaluator_prompt_template,
                "disapprove_feedback_template": self.self_critic.disapprove_feedback_template,
            }
        return json.dumps(d)
