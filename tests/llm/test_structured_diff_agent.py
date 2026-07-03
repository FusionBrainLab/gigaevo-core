"""Tests for DiffMutationAgent with a fake structured-output router."""

from __future__ import annotations

import json

import pytest

from gigaevo.chains.dag_changes import AllowedDagChanges
from gigaevo.exceptions import MutationError
from gigaevo.llm.agents.structured_diff import DiffMutationAgent
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Program
from tests.chains.test_dag_changes import make_genome


class FakeStructuredRouter:
    def __init__(self, payload):
        self.payload = payload
        self.schema_seen = None
        self.kwargs_seen = None
        self.messages_seen = None

    def with_structured_output(self, schema, **kwargs):
        self.schema_seen = schema
        self.kwargs_seen = kwargs
        return self

    async def ainvoke(self, messages):
        self.messages_seen = messages
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def _metrics_context() -> MetricsContext:
    return MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="Mean ROUGE-L F1",
                higher_is_better=True,
                is_primary=True,
                lower_bound=0.0,
                upper_bound=1.0,
                decimals=3,
            )
        }
    )


DIFF_PAYLOAD = {
    "reasoning": "add a verification step",
    "base_parent": "A",
    "slot_1": {"kind": "keep", "id": "a1"},
    "slot_2": {
        "kind": "new",
        "title": "Verify facts",
        "aim": "Cross-check the draft",
        "dependencies": ["slot_1"],
    },
    "slot_3": {"kind": "keep", "id": "a2", "dependencies": ["slot_2"]},
}


def _make_agent(payload):
    changes = AllowedDagChanges()
    router = FakeStructuredRouter(payload)
    agent = DiffMutationAgent(
        llm=router,
        allowed_changes=changes,
        task_description="Summarize Russian news in one sentence.",
        metrics_context=_metrics_context(),
    )
    return agent, router, changes


async def test_arun_returns_valid_child_code_and_diff():
    agent, router, changes = _make_agent(DIFF_PAYLOAD)
    parents_map = {"A": make_genome(2), "B": make_genome(3)}
    parents = [
        Program(code=parents_map["A"], iteration=0),
        Program(code=parents_map["B"], iteration=0),
    ]
    schema = changes.build_schema(parents_map)
    result = await agent.arun(
        parents=parents, parents_map=parents_map, diff_schema=schema
    )
    child = json.loads(result["code"])
    assert len(child["steps"]) == 3
    assert result["diff"] == DIFF_PAYLOAD
    assert router.kwargs_seen == {"method": "json_schema"}
    assert router.schema_seen["name"] == "chain_dag_diff"
    assert router.schema_seen["schema"]["title"] == "chain_dag_diff"


async def test_prompt_contains_parents_and_diff_language():
    agent, router, changes = _make_agent(DIFF_PAYLOAD)
    parents_map = {"A": make_genome(2), "B": make_genome(3)}
    parents = [Program(code=c, iteration=0) for c in parents_map.values()]
    schema = changes.build_schema(parents_map)
    await agent.arun(parents=parents, parents_map=parents_map, diff_schema=schema)
    system, user = router.messages_seen
    assert "POSITIONAL-SLOT CHAIN DIFF" in system.content
    assert "Summarize Russian news" in system.content
    assert "a1" in user.content and "b3" in user.content


async def test_router_parse_failure_raises():
    agent, _, changes = _make_agent(ValueError("Structured output parse failed"))
    parents_map = {"A": make_genome(2)}
    parents = [Program(code=parents_map["A"], iteration=0)]
    schema = changes.build_schema(parents_map)
    with pytest.raises(ValueError, match="parse failed"):
        await agent.arun(parents=parents, parents_map=parents_map, diff_schema=schema)


async def test_schema_invalid_payload_raises_diff_schema_error():
    agent, _, changes = _make_agent({"reasoning": "x", "base_parent": "A"})
    parents_map = {"A": make_genome(2)}
    parents = [Program(code=parents_map["A"], iteration=0)]
    schema = changes.build_schema(parents_map)
    with pytest.raises(MutationError, match="diff_schema_error"):
        await agent.arun(parents=parents, parents_map=parents_map, diff_schema=schema)
