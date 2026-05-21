"""Tests for RunnerConfig serialization/deserialization and chain_runner
feedback/self-critic paths.

Coverage
--------
- ``RunnerConfig.from_dict`` / ``from_env`` / ``to_json`` round-trips.
- Unknown-key warnings (soft) and strict-mode failures.
- Invalid JSON handling in ``from_env``.
- ``_apply_simple_retry`` — pattern matching, retry bounds, no-op cases.
- ``_apply_metric_feedback`` — default heuristic, custom metric, threshold.
- ``_apply_self_critic`` — APPROVE short-circuit, REJECT → regenerate.
- DATASET correction path via ``_run_chain_on_dataset_stepwise``.
- ``step_max_tokens`` is forwarded into the DATASET correction call.
"""

from __future__ import annotations

import json
import logging

import pytest

pytest.importorskip("mmar_carl")

from mmar_carl import LLMStepDescription  # noqa: E402
from mmar_carl.models.config import (  # noqa: E402
    ToolStepConfig,  # noqa: F401  (parity with sibling test)
)

from problems.chains.chain_runner import (  # noqa: E402
    _apply_metric_feedback,
    _apply_self_critic,
    _apply_simple_retry,
    _run_chain_on_dataset_stepwise,
)
from problems.chains.runner_config import (  # noqa: E402
    DatasetFeedbackConfig,
    FeedbackMode,
    MetricFeedbackConfig,
    RunnerConfig,
    SelfCriticConfig,
    SimpleRetryConfig,
    StepExecutionMode,
)
from problems.chains.types import ChainSpec  # noqa: E402

# ===========================================================================
# Helpers
# ===========================================================================


def _single_llm_chain() -> ChainSpec:
    step = LLMStepDescription(
        number=1,
        title="Answer",
        dependencies=[],
        aim="Give an answer",
        stage_action="Answer the question",
        reasoning_questions="",
        example_reasoning="",
    )
    return ChainSpec(system_prompt="sys", steps=[step])


class _ScriptedClient:
    """LLM client that returns scripted responses in call order.

    ``copy()`` returns ``self`` so all callers share the same response queue
    and call log — sufficient for deterministic tests.
    """

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    def copy(self) -> _ScriptedClient:
        return self

    async def __call__(self, prompt: str, **kwargs) -> str:
        self.calls.append((prompt, kwargs))
        if not self._responses:
            return ""
        return self._responses.pop(0)


# ===========================================================================
# RunnerConfig — (de)serialization
# ===========================================================================


class TestRunnerConfigSerialization:
    def test_defaults(self):
        cfg = RunnerConfig()
        assert cfg.feedback_mode is FeedbackMode.NONE
        assert cfg.execution_mode is StepExecutionMode.FAST

    def test_from_dict_basic_modes(self):
        cfg = RunnerConfig.from_dict(
            {"feedback_mode": "simple", "execution_mode": "self_critic"}
        )
        assert cfg.feedback_mode is FeedbackMode.SIMPLE
        assert cfg.execution_mode is StepExecutionMode.SELF_CRITIC

    def test_from_dict_nested_sub_config(self):
        cfg = RunnerConfig.from_dict(
            {
                "feedback_mode": "simple",
                "simple_retry": {
                    "bad_patterns": ["nope"],
                    "max_retries": 7,
                    "case_sensitive": True,
                },
            }
        )
        assert cfg.simple_retry.bad_patterns == ["nope"]
        assert cfg.simple_retry.max_retries == 7
        assert cfg.simple_retry.case_sensitive is True
        # Untouched fields keep defaults
        assert "not satisfactory" in cfg.simple_retry.feedback_message

    def test_to_json_roundtrip_none(self):
        cfg = RunnerConfig()
        data = json.loads(cfg.to_json())
        back = RunnerConfig.from_dict(data)
        assert back.feedback_mode is FeedbackMode.NONE
        assert back.execution_mode is StepExecutionMode.FAST

    def test_to_json_roundtrip_self_critic_dataset(self):
        cfg = RunnerConfig(
            feedback_mode=FeedbackMode.DATASET,
            execution_mode=StepExecutionMode.SELF_CRITIC,
            dataset_feedback=DatasetFeedbackConfig(answer_key="gt", max_retries=3),
            self_critic=SelfCriticConfig(max_revisions=5),
        )
        data = json.loads(cfg.to_json())
        back = RunnerConfig.from_dict(data)
        assert back.feedback_mode is FeedbackMode.DATASET
        assert back.execution_mode is StepExecutionMode.SELF_CRITIC
        assert back.dataset_feedback.answer_key == "gt"
        assert back.dataset_feedback.max_retries == 3
        assert back.self_critic.max_revisions == 5

    def test_to_json_roundtrip_metrics(self):
        cfg = RunnerConfig(
            feedback_mode=FeedbackMode.METRICS,
            metric_feedback=MetricFeedbackConfig(threshold=0.75, max_retries=4),
        )
        back = RunnerConfig.from_dict(json.loads(cfg.to_json()))
        assert back.feedback_mode is FeedbackMode.METRICS
        assert back.metric_feedback.threshold == 0.75
        assert back.metric_feedback.max_retries == 4

    def test_to_json_warns_when_metric_fn_set(self, caplog):
        cfg = RunnerConfig(
            feedback_mode=FeedbackMode.METRICS,
            metric_feedback=MetricFeedbackConfig(metric_fn=lambda xs: [1.0] * len(xs)),
        )
        with caplog.at_level(logging.WARNING, logger="problems.chains.runner_config"):
            cfg.to_json()
        assert any("metric_fn" in r.message for r in caplog.records)


class TestRunnerConfigUnknownKeys:
    def test_unknown_top_level_key_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="problems.chains.runner_config"):
            RunnerConfig.from_dict({"feedback_mode": "none", "typoed_key": 42})
        assert any("typoed_key" in r.message for r in caplog.records)

    def test_unknown_sub_key_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="problems.chains.runner_config"):
            RunnerConfig.from_dict(
                {
                    "feedback_mode": "simple",
                    "simple_retry": {"bad_patters": ["oops"]},  # typo
                }
            )
        assert any("bad_patters" in r.message for r in caplog.records)

    def test_strict_mode_rejects_unknown_top_key(self):
        with pytest.raises(ValueError, match="Unknown RunnerConfig keys"):
            RunnerConfig.from_dict({"feedbak_mode": "simple"}, strict=True)

    def test_strict_mode_rejects_unknown_sub_key(self):
        with pytest.raises(ValueError, match=r"RunnerConfig\[simple_retry\]"):
            RunnerConfig.from_dict(
                {"feedback_mode": "simple", "simple_retry": {"nope": 1}},
                strict=True,
            )


class TestRunnerConfigFromEnv:
    def test_empty_env_returns_defaults(self, monkeypatch):
        monkeypatch.delenv("GIGAEVO_CHAIN_RUNNER_CONFIG", raising=False)
        cfg = RunnerConfig.from_env()
        assert cfg.feedback_mode is FeedbackMode.NONE

    def test_valid_env_round_trips(self, monkeypatch):
        payload = RunnerConfig(feedback_mode=FeedbackMode.SIMPLE).to_json()
        monkeypatch.setenv("GIGAEVO_CHAIN_RUNNER_CONFIG", payload)
        cfg = RunnerConfig.from_env()
        assert cfg.feedback_mode is FeedbackMode.SIMPLE

    def test_invalid_json_logs_warning_and_returns_defaults(self, monkeypatch, caplog):
        monkeypatch.setenv("GIGAEVO_CHAIN_RUNNER_CONFIG", "{not-valid-json")
        with caplog.at_level(logging.WARNING, logger="problems.chains.runner_config"):
            cfg = RunnerConfig.from_env()
        assert cfg.feedback_mode is FeedbackMode.NONE
        assert any("not valid JSON" in r.message for r in caplog.records)

    def test_invalid_json_strict_raises(self, monkeypatch):
        monkeypatch.setenv("GIGAEVO_CHAIN_RUNNER_CONFIG", "{not-valid-json")
        with pytest.raises(ValueError, match="not valid JSON"):
            RunnerConfig.from_env(strict=True)


# ===========================================================================
# Feedback helpers — _apply_simple_retry / _apply_metric_feedback
# ===========================================================================


@pytest.mark.asyncio
class TestSimpleRetry:
    async def test_no_bad_pattern_skips_retry(self):
        client = _ScriptedClient([])  # would explode if invoked
        cfg = RunnerConfig(simple_retry=SimpleRetryConfig(max_retries=3))
        import asyncio as _aio

        out = await _apply_simple_retry(
            ["a clean answer"], ["prompt"], client, _aio.Semaphore(1), cfg, {}
        )
        assert out == ["a clean answer"]
        assert client.calls == []

    async def test_retries_and_replaces_bad_output(self):
        # First output trips pattern; scripted retry returns clean text.
        client = _ScriptedClient(["now a proper answer"])
        cfg = RunnerConfig(
            simple_retry=SimpleRetryConfig(bad_patterns=["i don't know"], max_retries=2)
        )
        import asyncio as _aio

        out = await _apply_simple_retry(
            ["I don't know really"],
            ["original prompt"],
            client,
            _aio.Semaphore(1),
            cfg,
            {"max_tokens": 321},
        )
        assert out == ["now a proper answer"]
        # Overrides were forwarded
        assert client.calls and client.calls[0][1].get("max_tokens") == 321
        # Feedback message was appended to the retry prompt
        assert "not satisfactory" in client.calls[0][0]

    async def test_retry_bound_respected(self):
        client = _ScriptedClient(["still idk", "still idk", "still idk"])
        cfg = RunnerConfig(
            simple_retry=SimpleRetryConfig(bad_patterns=["idk"], max_retries=2)
        )
        import asyncio as _aio

        out = await _apply_simple_retry(
            ["idk"], ["p"], client, _aio.Semaphore(1), cfg, {}
        )
        # Bad text persists, but no more than max_retries calls were made.
        assert len(client.calls) == 2
        assert "idk" in out[0].lower()

    async def test_case_insensitive_matching(self):
        client = _ScriptedClient(["good"])
        cfg = RunnerConfig(
            simple_retry=SimpleRetryConfig(
                bad_patterns=["i don't know"], max_retries=1, case_sensitive=False
            )
        )
        import asyncio as _aio

        out = await _apply_simple_retry(
            ["I DON'T KNOW"], ["p"], client, _aio.Semaphore(1), cfg, {}
        )
        assert out == ["good"]


@pytest.mark.asyncio
class TestMetricFeedback:
    async def test_default_heuristic_retries_short_outputs(self):
        # min_output_length=5 words: "hi" scores 0, retry returns long text.
        client = _ScriptedClient(["this is a much longer response than before"])
        cfg = RunnerConfig(
            metric_feedback=MetricFeedbackConfig(
                threshold=0.5, max_retries=1, min_output_length=5
            )
        )
        import asyncio as _aio

        out = await _apply_metric_feedback(
            ["hi"], ["p"], client, _aio.Semaphore(1), cfg, {}
        )
        assert out == ["this is a much longer response than before"]

    async def test_custom_metric_fn_picks_which_to_retry(self):
        # Custom metric: odd indices score below threshold.
        retries = ["RETRIED_1"]
        client = _ScriptedClient(retries)
        cfg = RunnerConfig(
            metric_feedback=MetricFeedbackConfig(
                threshold=0.5,
                max_retries=1,
                metric_fn=lambda xs: [0.0 if i % 2 else 1.0 for i, _ in enumerate(xs)],
            )
        )
        import asyncio as _aio

        out = await _apply_metric_feedback(
            ["keep_0", "kill_1"], ["p0", "p1"], client, _aio.Semaphore(1), cfg, {}
        )
        assert out[0] == "keep_0"
        assert out[1] == "RETRIED_1"
        # Only one retry call (for index 1).
        assert len(client.calls) == 1


# ===========================================================================
# Self-critic
# ===========================================================================


@pytest.mark.asyncio
class TestSelfCritic:
    async def test_approve_short_circuits(self):
        # Evaluator returns APPROVE — no regeneration.
        client = _ScriptedClient(["APPROVE"])
        cfg = RunnerConfig(self_critic=SelfCriticConfig(max_revisions=3))
        step = LLMStepDescription(
            number=1,
            title="T",
            dependencies=[],
            aim="aim",
            stage_action="s",
            reasoning_questions="",
            example_reasoning="",
        )
        import asyncio as _aio

        out = await _apply_self_critic(
            ["original"], ["p"], step, client, _aio.Semaphore(1), cfg, {}
        )
        assert out == ["original"]
        # Only the evaluator call; no regeneration.
        assert len(client.calls) == 1

    async def test_reject_triggers_regeneration(self):
        # Round 1: evaluator rejects, regeneration returns "fixed"
        # Round 2: evaluator approves.
        client = _ScriptedClient(["REJECT: too short", "fixed answer", "APPROVE"])
        cfg = RunnerConfig(self_critic=SelfCriticConfig(max_revisions=3))
        step = LLMStepDescription(
            number=1,
            title="T",
            dependencies=[],
            aim="aim",
            stage_action="s",
            reasoning_questions="",
            example_reasoning="",
        )
        import asyncio as _aio

        out = await _apply_self_critic(
            ["bad"], ["orig"], step, client, _aio.Semaphore(1), cfg, {}
        )
        assert out == ["fixed answer"]


# ===========================================================================
# DATASET post-loop correction (end-to-end through _run_chain_on_dataset_stepwise)
# ===========================================================================


@pytest.mark.asyncio
class TestDatasetCorrection:
    async def test_correction_fires_when_answer_missing(self):
        chain = _single_llm_chain()
        # 1st: wrong answer; 2nd: correction containing expected answer.
        client = _ScriptedClient(["something else", "final answer: 42"])
        cfg = RunnerConfig(
            feedback_mode=FeedbackMode.DATASET,
            dataset_feedback=DatasetFeedbackConfig(answer_key="answer", max_retries=1),
        )

        results = await _run_chain_on_dataset_stepwise(
            chain=chain,
            client=client,
            dataset=[{"answer": "42"}],
            outer_context_builder=lambda s: "ctx",
            runner_config=cfg,
        )
        assert "42" in results[0].final_output
        # One initial + one correction call.
        assert len(client.calls) == 2

    async def test_no_correction_when_answer_present(self):
        chain = _single_llm_chain()
        client = _ScriptedClient(["The answer is 42."])
        cfg = RunnerConfig(
            feedback_mode=FeedbackMode.DATASET,
            dataset_feedback=DatasetFeedbackConfig(answer_key="answer", max_retries=3),
        )

        results = await _run_chain_on_dataset_stepwise(
            chain=chain,
            client=client,
            dataset=[{"answer": "42"}],
            outer_context_builder=lambda s: "ctx",
            runner_config=cfg,
        )
        assert "42" in results[0].final_output
        assert len(client.calls) == 1

    async def test_step_max_tokens_forwarded_to_correction(self):
        chain = _single_llm_chain()
        client = _ScriptedClient(["wrong", "right 7"])
        cfg = RunnerConfig(
            feedback_mode=FeedbackMode.DATASET,
            dataset_feedback=DatasetFeedbackConfig(answer_key="answer", max_retries=1),
        )

        await _run_chain_on_dataset_stepwise(
            chain=chain,
            client=client,
            dataset=[{"answer": "7"}],
            outer_context_builder=lambda s: "ctx",
            step_max_tokens={1: 555},
            runner_config=cfg,
        )
        # Both the initial step call AND the correction call get max_tokens.
        assert len(client.calls) == 2
        assert client.calls[0][1].get("max_tokens") == 555
        assert client.calls[1][1].get("max_tokens") == 555
