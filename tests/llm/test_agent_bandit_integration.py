"""Regression-lock integration tests: agent layer + ``BanditModelRouter``.

The bandit-classifier wiring is exercised by unit tests in
``tests/evolution/test_bandit.py`` against the router in isolation. These
tests verify the *agent-to-bandit* contract end to end: when an agent
constructs a structured-output chain on top of a real
``BanditModelRouter`` and the underlying call raises, the bandit's
failure hook must still fire — i.e. no intermediate ``Runnable`` in the
agent layer swallows the exception before the bandit can record a
zero-reward injection.

A regression in any agent that introduces an internal retry/fallback
wrapper, or that catches the LLM exception before it reaches the bandit
hook, would surface here.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import HumanMessage
import pytest

from gigaevo.llm.agents.insights import InsightsAgent
from gigaevo.llm.agents.lineage import LineageAgent
from gigaevo.llm.agents.mutation import MutationAgent, MutationStructuredOutput
from gigaevo.llm.agents.scoring import ScoringAgent
from gigaevo.llm.bandit import BanditModelRouter
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.metrics.formatter import MetricsFormatter
from gigaevo.programs.program import Program

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _ctx() -> MetricsContext:
    return MetricsContext(
        specs={
            "score": MetricSpec(
                description="primary",
                is_primary=True,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
                sentinel_value=-1.0,
            )
        }
    )


def _bandit(
    *,
    ainvoke_side_effect: BaseException | None = None,
    invoke_side_effect: BaseException | None = None,
    ainvoke_return: Any = None,
) -> tuple[BanditModelRouter, MagicMock]:
    """Build a one-arm ``BanditModelRouter`` whose underlying model can be
    flaky.

    Returns the router and the underlying mock so tests can override
    behaviour per call.  ``with_structured_output`` returns the *same*
    mock so the inner ``model.ainvoke``/``invoke`` is exercised by the
    structured-output dispatch path.
    """
    model = MagicMock()
    model.model_name = "flaky"
    if ainvoke_side_effect is not None:
        model.ainvoke = AsyncMock(side_effect=ainvoke_side_effect)
    else:
        model.ainvoke = AsyncMock(return_value=ainvoke_return)
    if invoke_side_effect is not None:
        model.invoke = MagicMock(side_effect=invoke_side_effect)
    else:
        model.invoke = MagicMock(return_value=ainvoke_return)
    model.with_structured_output = MagicMock(return_value=model)

    router = BanditModelRouter(
        [model], [1.0], fitness_key="score", higher_is_better=True
    )
    router._langfuse = None
    router._tracker = MagicMock()
    return router, model


def _program(metrics: dict | None = None, code: str = "def f(): return 0") -> Program:
    p = Program(code=code)
    if metrics:
        p.add_metrics(metrics)
    return p


# ---------------------------------------------------------------------------
# MutationAgent <-> BanditModelRouter
# ---------------------------------------------------------------------------


class TestMutationAgentBanditWiring:
    """``MutationAgent.acall_llm`` catches LLM exceptions and turns them
    into ``state["error"]`` so the LangGraph chain returns an empty-code
    parsed_output rather than aborting the DAG.  The bandit's failure
    hook must still fire *before* that catch, so the ledger stays in
    step (``total_pulls == window_size``) even though the agent layer
    swallows the exception for downstream sanity."""

    @pytest.mark.asyncio
    async def test_transport_failure_fires_bandit_hook_through_agent(self) -> None:
        router, _model = _bandit(ainvoke_side_effect=RuntimeError("rate limit"))
        agent = MutationAgent(
            llm=router,
            system_prompt="sys",
            user_prompt_template="Mutate {count}:\n{parent_blocks}",
            mutation_mode="rewrite",
        )

        state = {
            "input": [_program()],
            "mutation_mode": "rewrite",
            "messages": [HumanMessage(content="hi")],
            "llm_response": None,
            "final_code": "",
            "mutation_label": "",
        }
        result = await agent.acall_llm(state)  # type: ignore[arg-type]

        # Agent-side: exception swallowed into state["error"].
        assert result["llm_response"] is None
        assert "rate limit" in result["error"]

        # Bandit-side: failure hook fired exactly once → ledger in step.
        stats = router.get_bandit_stats()
        assert stats["flaky"]["total_pulls"] == 1
        assert stats["flaky"]["window_size"] == 1

    @pytest.mark.asyncio
    async def test_repeated_failures_keep_ledger_in_step_through_agent(self) -> None:
        router, _model = _bandit(ainvoke_side_effect=RuntimeError("boom"))
        agent = MutationAgent(
            llm=router,
            system_prompt="sys",
            user_prompt_template="Mutate {count}:\n{parent_blocks}",
            mutation_mode="rewrite",
        )

        for _ in range(5):
            state = {
                "input": [_program()],
                "mutation_mode": "rewrite",
                "messages": [HumanMessage(content="hi")],
                "llm_response": None,
                "final_code": "",
                "mutation_label": "",
            }
            await agent.acall_llm(state)  # type: ignore[arg-type]

        stats = router.get_bandit_stats()
        assert stats["flaky"]["total_pulls"] == 5
        assert stats["flaky"]["window_size"] == 5

    @pytest.mark.asyncio
    async def test_success_path_defers_reward_through_agent(self) -> None:
        # On the success path the bandit defers the reward to
        # on_mutation_outcome, so total_pulls advances but window_size
        # stays at 0.  Confirms no over-injection from the agent layer.
        # The bandit forces ``include_raw=True`` on the underlying
        # ``with_structured_output`` call, so the mock must return the
        # langchain dict shape (raw, parsed, parsing_error) — not the
        # bare pydantic object.
        parsed = MutationStructuredOutput(
            archetype="x",
            justification="y",
            insights_used=[],
            code="def f(): return 1",
        )
        success_response = {
            "raw": MagicMock(),
            "parsed": parsed,
            "parsing_error": None,
        }
        router, _model = _bandit(ainvoke_return=success_response)
        agent = MutationAgent(
            llm=router,
            system_prompt="sys",
            user_prompt_template="Mutate {count}:\n{parent_blocks}",
            mutation_mode="rewrite",
        )

        state = {
            "input": [_program()],
            "mutation_mode": "rewrite",
            "messages": [HumanMessage(content="hi")],
            "llm_response": None,
            "final_code": "",
            "mutation_label": "",
        }
        await agent.acall_llm(state)  # type: ignore[arg-type]

        stats = router.get_bandit_stats()
        assert stats["flaky"]["total_pulls"] == 1
        # Reward deferred to on_mutation_outcome → no entry in window yet.
        assert stats["flaky"]["window_size"] == 0


# ---------------------------------------------------------------------------
# InsightsAgent / LineageAgent / ScoringAgent <-> BanditModelRouter
# ---------------------------------------------------------------------------


class TestStructuredAgentsBanditWiring:
    """Non-mutation agents inherit ``base.acall_llm`` which does *not*
    swallow exceptions.  The bandit's failure hook must fire and the
    exception must propagate out of ``agent.arun`` so the DAG runner can
    discard the program.  This locks in that no agent has accidentally
    added a retry/fallback wrapper that would swallow the failure
    before the hook fires."""

    def _insights_agent(self, router: BanditModelRouter) -> InsightsAgent:
        return InsightsAgent(
            llm=router,
            system_prompt_template="sys",
            user_prompt_template="code={code} metrics={metrics} errors={error_section} max={max_insights}",
            max_insights=3,
            metrics_formatter=MetricsFormatter(_ctx()),
        )

    def _lineage_agent(self, router: BanditModelRouter) -> LineageAgent:
        return LineageAgent(
            llm=router,
            system_prompt="sys",
            user_prompt_template=(
                "task={task_description} m={metric_name} d={metric_description} "
                "delta={delta} h={higher_is_better_text} interp={delta_interpretation} "
                "pe={parent_errors} ce={child_errors} am={additional_metrics} "
                "db={diff_blocks} pc={parent_code}"
            ),
            task_description="t",
            metrics_formatter=MetricsFormatter(_ctx()),
        )

    def _scoring_agent(self, router: BanditModelRouter) -> ScoringAgent:
        return ScoringAgent(
            llm=router,
            system_prompt="sys",
            user_prompt_template="code={code} trait={trait_description} max={max_score}",
            trait_description="novelty",
            max_score=1.0,
        )

    @pytest.mark.asyncio
    async def test_insights_agent_failure_fires_bandit_hook(self) -> None:
        router, _model = _bandit(ainvoke_side_effect=RuntimeError("boom"))
        agent = self._insights_agent(router)

        program = _program(metrics={"score": 0.5})

        # Failure propagates out of agent.arun (base.acall_llm does not
        # swallow, parse_response is never reached).
        with pytest.raises(RuntimeError, match="boom"):
            await agent.arun(program)

        stats = router.get_bandit_stats()
        assert stats["flaky"]["total_pulls"] == 1
        assert stats["flaky"]["window_size"] == 1

    @pytest.mark.asyncio
    async def test_scoring_agent_failure_fires_bandit_hook(self) -> None:
        router, _model = _bandit(ainvoke_side_effect=RuntimeError("scoring boom"))
        agent = self._scoring_agent(router)

        program = _program()

        with pytest.raises(RuntimeError, match="scoring boom"):
            await agent.arun(program)

        stats = router.get_bandit_stats()
        assert stats["flaky"]["total_pulls"] == 1
        assert stats["flaky"]["window_size"] == 1

    @pytest.mark.asyncio
    async def test_lineage_agent_failure_fires_bandit_hook(self) -> None:
        router, _model = _bandit(ainvoke_side_effect=RuntimeError("lineage boom"))
        agent = self._lineage_agent(router)

        parent = _program(metrics={"score": 0.4}, code="def f(): return 0")
        child = _program(metrics={"score": 0.6}, code="def f(): return 1")

        with pytest.raises(RuntimeError, match="lineage boom"):
            await agent.arun(parents=[parent], program=child)

        stats = router.get_bandit_stats()
        assert stats["flaky"]["total_pulls"] == 1
        assert stats["flaky"]["window_size"] == 1


# ---------------------------------------------------------------------------
# Documented gap: silent ``parsed=None`` from ``include_raw=True`` chain
# ---------------------------------------------------------------------------


class TestSilentNoneParseGapIsDocumented:
    """``BanditModelRouter.with_structured_output`` passes
    ``include_raw=True`` to the underlying langchain wrapper, which means
    a schema-parse failure surfaces as ``response['parsing_error']``
    instead of raising.  ``_StructuredOutputRouter._process`` returns
    ``response.get('parsed')`` (i.e. ``None``) without firing the
    failure hook, so the pull is recorded but no reward injection lands
    in the window — ledger asymmetry.

    The fix belongs in ``gigaevo/llm/models.py``
    (``_StructuredOutputRouter._process``) and is queued as **FU12**.
    This regression-lock test documents the current (broken) behaviour
    so the fix can be detected: when FU12 lands, ``window_size`` will
    equal ``total_pulls`` and this test will need its assertion
    flipped.
    """

    @pytest.mark.asyncio
    async def test_silent_none_parse_currently_leaves_window_empty(self) -> None:
        # Simulate the langchain include_raw=True response shape on parse
        # failure: parsed=None, parsing_error=<exc>.  The _process path
        # returns None silently — exactly the path FU12 must fix.
        silent_none = {
            "raw": MagicMock(),
            "parsed": None,
            "parsing_error": ValueError("schema validation failed"),
        }
        router, _model = _bandit(ainvoke_return=silent_none)

        agent = MutationAgent(
            llm=router,
            system_prompt="sys",
            user_prompt_template="Mutate {count}:\n{parent_blocks}",
            mutation_mode="rewrite",
        )

        state = {
            "input": [_program()],
            "mutation_mode": "rewrite",
            "messages": [HumanMessage(content="hi")],
            "llm_response": None,
            "final_code": "",
            "mutation_label": "",
        }
        result = await agent.acall_llm(state)  # type: ignore[arg-type]

        # Current behaviour (the gap): MutationAgent.acall_llm crashes on
        # ``None.archetype`` access, catches it, sets state["error"], but
        # the bandit hook was *never* invoked from the structured-output
        # dispatch because _StructuredOutputRouter._process returned
        # None without raising.
        assert result["llm_response"] is None
        assert "NoneType" in result["error"]

        stats = router.get_bandit_stats()
        # Pull recorded, but no reward injection — FU12 must close this.
        assert stats["flaky"]["total_pulls"] == 1
        assert stats["flaky"]["window_size"] == 0
