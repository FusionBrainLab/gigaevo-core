from __future__ import annotations

import json
from pathlib import Path

from gigaevo.evolution.mutation.base import MutationSpec
from gigaevo.evolution.mutation.structured_diff import StructuredDiffMutationOperator
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Program
from problems.dag_tab.allowed_changes import AllowedDagTabChanges
from problems.dag_tab.graph import FeatureGraph


ROOT = Path(__file__).parents[2]
PARENT = (ROOT / "problems/dag_tab/initial_programs/baseline.json").read_text()
PAYLOAD = {
    "archetype": "Guided Innovation",
    "justification": "Add a stable population-per-occupancy feature.",
    "insights_used": [],
    "insight_ids_used": [],
    "card_ids_used": [],
    "changes": [
        {
            "description": "Add population interaction",
            "explanation": "The raw population and occupancy columns encode density.",
        }
    ],
    "base_parent": "A",
    "slot_1": {
        "kind": "new",
        "id": "population_per_occupancy",
        "input_cols": ["x4", "x5"],
        "output_cols": ["fe_population_per_occupancy"],
        "code": "df['fe_population_per_occupancy'] = df['x4'] / (df['x5'].abs() + 1e-6)\nreturn df",
        "rationale": "Estimate block-group density per average occupancy.",
        "is_output": True,
    },
    "slot_2": None,
}


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
        return self.payload


class Context:
    task_description = "Evolve a tabular feature graph."
    metrics_context = MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="Cross-validation score",
                higher_is_better=True,
                is_primary=True,
                lower_bound=-1.0,
                upper_bound=1.0,
                decimals=4,
            )
        }
    )


async def test_generic_operator_evolves_feature_graph_json():
    router = FakeStructuredRouter(PAYLOAD)
    operator = StructuredDiffMutationOperator(
        llm_wrapper=router,
        allowed_changes=AllowedDagTabChanges(max_nodes=2),
        problem_context=Context(),
    )
    parent = Program(code=PARENT, iteration=0)

    spec = await operator.mutate_single([parent])

    assert spec is not None
    child = FeatureGraph.from_json(spec.code)
    assert child.nodes[0].id == "population_per_occupancy"
    assert spec.name == "structured_diff"
    assert spec.parents == [parent]
    assert spec.metadata[MutationSpec.META_OUTPUT] == PAYLOAD
    assert spec.mutation_archetype == "Guided Innovation"
    assert router.kwargs_seen == {"method": "json_schema"}
    assert router.schema_seen["name"] == "dag_tab_feature_graph_diff"
    assert "POSITIONAL-SLOT TABULAR FEATURE-GRAPH DIFF" in (
        router.messages_seen[0].content
    )
    assert "a1" in router.messages_seen[1].content
    assert json.loads(spec.code)["dataset"] == "california"
