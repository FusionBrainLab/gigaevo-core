"""Tests for DiffMutationAgent with a fake structured-output router."""

from __future__ import annotations

import json

import pytest

from gigaevo.chains.dag_changes import AllowedDagChanges
from gigaevo.evolution.mutation.allowed_changes import AllowedChanges, DiffSchema
from gigaevo.evolution.mutation.constants import MUTATION_CONTEXT_METADATA_KEY
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
    "archetype": "Guided Innovation",
    "justification": "insight 1 flags a terse final step; a verify pass re-grounds it",
    "insights_used": ["insight: structure — final step"],
    "insight_ids_used": [{"parent": "A", "insight": 1}],
    "card_ids_used": [],
    "changes": [
        {
            "description": "Added a verify step after the draft",
            "explanation": "the drafter drops source entities; a cross-check restores them",
        }
    ],
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
    assert router.kwargs_seen == {}
    assert router.schema_seen["name"] == "chain_dag_diff"
    assert router.schema_seen["schema"]["title"] == "chain_dag_diff"


async def test_structured_diff_uses_router_configured_transport():
    agent, router, changes = _make_agent(DIFF_PAYLOAD)
    parents_map = {"A": make_genome(2)}
    parents = [Program(code=parents_map["A"], iteration=0)]

    await agent.arun(
        parents=parents,
        parents_map=parents_map,
        diff_schema=changes.build_schema(parents_map),
    )

    assert router.kwargs_seen == {}


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
    # missing the now-required archetype/justification evidence fields
    agent, _, changes = _make_agent(
        {"base_parent": "A", "slot_1": {"kind": "keep", "id": "a1"}}
    )
    parents_map = {"A": make_genome(2)}
    parents = [Program(code=parents_map["A"], iteration=0)]
    schema = changes.build_schema(parents_map)
    with pytest.raises(MutationError, match="diff_schema_error"):
        await agent.arun(parents=parents, parents_map=parents_map, diff_schema=schema)


async def test_parent_evaluation_context_included_when_present():
    agent, router, changes = _make_agent(DIFF_PAYLOAD)
    parents_map = {"A": make_genome(2), "B": make_genome(3)}
    parents = [
        Program(
            code=parents_map["A"],
            iteration=0,
            metadata={MUTATION_CONTEXT_METADATA_KEY: "ctx A"},
        ),
        Program(code=parents_map["B"], iteration=0),
    ]
    schema = changes.build_schema(parents_map)
    await agent.arun(parents=parents, parents_map=parents_map, diff_schema=schema)
    _, user = router.messages_seen
    assert "=== Parent A evaluation context ===" in user.content
    assert "ctx A" in user.content
    assert "=== Parent B evaluation context ===" not in user.content


async def test_citation_integrity_grounds_letter_parent_insight():
    agent, _, changes = _make_agent(DIFF_PAYLOAD)
    parents_map = {"A": make_genome(2), "B": make_genome(3)}
    # DIFF_PAYLOAD cites {"parent": "A", "insight": 1}; render insight 1 into A's
    # evaluation-context block using the marker InsightsMutationContext.format emits
    parents = [
        Program(
            code=parents_map["A"],
            iteration=0,
            metadata={
                MUTATION_CONTEXT_METADATA_KEY: (
                    "## Program Insights\n1. **[structure]** final step is terse"
                )
            },
        ),
        Program(code=parents_map["B"], iteration=0),
    ]
    schema = changes.build_schema(parents_map)
    result = await agent.arun(
        parents=parents, parents_map=parents_map, diff_schema=schema
    )
    integrity = result["metadata"]["citation_integrity"]
    assert integrity["cited"] == 1
    assert integrity["grounded"] == 1


class NeutralChanges(AllowedChanges):
    """Genome-agnostic contract probe: no chain vocabulary anywhere."""

    def __init__(self, payload_transform=None):
        self._transform = payload_transform or (lambda p: p)

    def build_schema(self, parents):
        return DiffSchema(
            json_schema={"title": "neutral_diff", "type": "object"},
            validate=self._transform,
        )

    def render_parents(self, parents):
        return " / ".join(parents.values())

    def apply(self, diff, parents):
        return json.dumps({"child": True})

    def describe(self):
        return "NEUTRAL DIFF LANGUAGE"


def _make_neutral_agent(payload, payload_transform=None):
    changes = NeutralChanges(payload_transform)
    router = FakeStructuredRouter(payload)
    agent = DiffMutationAgent(
        llm=router,
        allowed_changes=changes,
        task_description="Optimize any genome family.",
        metrics_context=_metrics_context(),
    )
    return agent, router, changes


async def test_agent_is_genome_agnostic_over_custom_changes():
    payload = {"move": "swap"}
    agent, router, changes = _make_neutral_agent(payload)
    parents_map = {"A": "genome-alpha", "B": "genome-beta"}
    parents = [Program(code=c, iteration=0) for c in parents_map.values()]
    result = await agent.arun(
        parents=parents,
        parents_map=parents_map,
        diff_schema=changes.build_schema(parents_map),
    )
    assert json.loads(result["code"]) == {"child": True}
    assert result["diff"] == payload
    assert router.schema_seen["name"] == "neutral_diff"
    system, user = router.messages_seen
    assert "NEUTRAL DIFF LANGUAGE" in system.content
    assert "genome-alpha / genome-beta" in user.content


async def test_non_dict_payload_is_wrapped_for_storage():
    agent, _, changes = _make_neutral_agent(["x"], payload_transform=tuple)
    parents_map = {"A": "genome-alpha"}
    parents = [Program(code=parents_map["A"], iteration=0)]
    result = await agent.arun(
        parents=parents,
        parents_map=parents_map,
        diff_schema=changes.build_schema(parents_map),
    )
    assert result["diff"] == {"payload": ["x"]}
