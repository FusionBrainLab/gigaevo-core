"""Tests for StructuredDiffMutationOperator with a fake structured-output router."""

from __future__ import annotations

import json

import pytest

from gigaevo.chains.dag_changes import AllowedDagChanges
from gigaevo.evolution.mutation.base import MutationSpec
from gigaevo.evolution.mutation.structured_diff import StructuredDiffMutationOperator
from gigaevo.exceptions import MutationError
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Program
from tests.chains.test_dag_changes import make_genome
from tests.llm.test_structured_diff_agent import DIFF_PAYLOAD, FakeStructuredRouter


class _Ctx:
    task_description = "Summarize Russian news in one sentence."
    metrics_context = MetricsContext(
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


def _operator(payload):
    return StructuredDiffMutationOperator(
        llm_wrapper=FakeStructuredRouter(payload),
        allowed_changes=AllowedDagChanges(),
        problem_context=_Ctx(),
    )


async def test_mutate_single_returns_spec_with_diff_metadata():
    operator = _operator(DIFF_PAYLOAD)
    parents = [
        Program(code=make_genome(2), iteration=0),
        Program(code=make_genome(3), iteration=0),
    ]
    spec = await operator.mutate_single(parents)
    assert spec is not None
    assert spec.name == "structured_diff"
    assert spec.parents == parents
    assert len(json.loads(spec.code)["steps"]) == 3
    assert spec.metadata[MutationSpec.META_OUTPUT] == DIFF_PAYLOAD
    # evidence-field parity: archetype flows to MutationSpec, citation integrity stamped
    assert spec.mutation_archetype == "Guided Innovation"
    assert set(spec.metadata["citation_integrity"]) == {
        "cited",
        "grounded",
        "cards_cited",
        "cards_grounded",
    }


async def test_no_parents_returns_none():
    operator = _operator(DIFF_PAYLOAD)
    assert await operator.mutate_single([]) is None


async def test_schema_invalid_payload_surfaces_mutation_error():
    operator = _operator({"base_parent": "A", "slot_1": {"kind": "keep", "id": "a1"}})
    parents = [Program(code=make_genome(2), iteration=0)]
    with pytest.raises(MutationError, match="diff_schema_error"):
        await operator.mutate_single(parents)


async def test_router_failure_wrapped_as_llm_call_error():
    operator = _operator(ValueError("Structured output parse failed: boom"))
    parents = [Program(code=make_genome(2), iteration=0)]
    with pytest.raises(MutationError, match="llm_call_error"):
        await operator.mutate_single(parents)
