"""NoiseAwareMemoryGuidedPipelineBuilder wiring.

The noise pipeline swaps the ``CallValidatorFunction`` node's stage class for
``ProgramMetadataValidatorStage`` — same node name, so the DAG must be
wire-identical to the parent builder (edges, exec deps, node set). The parent
must keep the stock stage class — control-arm parity.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.entrypoint.lineage_memory_pipeline import (
    MemoryGuidedMutationPipelineBuilder,
)
from gigaevo.entrypoint.noise_aware_pipeline import (
    NoiseAwareMemoryGuidedPipelineBuilder,
)
from gigaevo.llm.models import MultiModelRouter
from gigaevo.memory.provider import MemoryProvider
from gigaevo.problems.context import ProblemContext
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.stages.python_executors.execution import CallValidatorFunction
from gigaevo.programs.stages.validator_metadata import ProgramMetadataValidatorStage
from gigaevo.runner.dag_blueprint import DAGBlueprint

VALIDATOR_NODE = "CallValidatorFunction"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(problem_dir) -> EvolutionContext:
    (problem_dir / "validate.py").write_text(
        "def validate(payload):\n    return {'fitness': 0.0}\n"
    )
    metrics_ctx = MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="main metric",
                is_primary=True,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
        }
    )
    problem_ctx = MagicMock(spec=ProblemContext)
    problem_ctx.problem_dir = problem_dir
    problem_ctx.task_description = "Solve the task."
    problem_ctx.metrics_context = metrics_ctx
    problem_ctx.is_contextual = False
    return EvolutionContext(
        problem_ctx=problem_ctx,
        llm_wrapper=MagicMock(spec=MultiModelRouter),
        storage=MagicMock(spec=ProgramStorage),
        prompts_dir=None,
        memory_provider=MagicMock(spec=MemoryProvider),
    )


def _build(tmp_path, **kwargs) -> DAGBlueprint:
    return NoiseAwareMemoryGuidedPipelineBuilder(
        _make_ctx(tmp_path), **kwargs
    ).build_blueprint()


def _build_parent(tmp_path, **kwargs) -> DAGBlueprint:
    return MemoryGuidedMutationPipelineBuilder(
        _make_ctx(tmp_path), **kwargs
    ).build_blueprint()


def _edge_triples(bp: DAGBlueprint) -> set[tuple[str, str, str]]:
    return {
        (e.source_stage, e.destination_stage, e.input_name) for e in bp.data_flow_edges
    }


def _dep_triples(bp: DAGBlueprint) -> set[tuple[str, str, str]]:
    return {
        (stage, dep.stage_name, dep.condition)
        for stage, deps in (bp.exec_order_deps or {}).items()
        for dep in deps
    }


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


class TestNoiseAwareWiring:
    def test_validator_stage_swapped(self, tmp_path):
        bp = _build(tmp_path)
        stage = bp.nodes[VALIDATOR_NODE]()
        assert isinstance(stage, ProgramMetadataValidatorStage)

    def test_dag_is_wire_identical_to_parent(self, tmp_path):
        noise, parent = _build(tmp_path), _build_parent(tmp_path)
        assert set(noise.nodes) == set(parent.nodes)
        assert _edge_triples(noise) == _edge_triples(parent)
        assert _dep_triples(noise) == _dep_triples(parent)

    def test_wire_parity_holds_with_archive_gate(self, tmp_path):
        noise = _build(tmp_path, archive_gate_enabled=True)
        parent = _build_parent(tmp_path, archive_gate_enabled=True)
        assert "ArchivePotentialGateStage" in noise.nodes
        assert set(noise.nodes) == set(parent.nodes)
        assert _dep_triples(noise) == _dep_triples(parent)


class TestControlParity:
    def test_parent_builder_keeps_stock_validator_stage(self, tmp_path):
        bp = _build_parent(tmp_path)
        stage = bp.nodes[VALIDATOR_NODE]()
        assert isinstance(stage, CallValidatorFunction)
        assert not isinstance(stage, ProgramMetadataValidatorStage)

    def test_swapped_stage_constructed_like_parent_validator(self, tmp_path):
        # The feature hand-duplicates the parent's construction kwargs; if the
        # parent feature ever changes how it builds CallValidatorFunction, the
        # treatment arm's validator must not silently diverge from control.
        noise_stage = _build(tmp_path).nodes[VALIDATOR_NODE]()
        parent_stage = _build_parent(tmp_path).nodes[VALIDATOR_NODE]()
        for attr in (
            "function_name",
            "python_path",
            "max_output_size",
            "max_memory_mb",
            "timeout",
            "_validator_code",
        ):
            assert getattr(noise_stage, attr) == getattr(parent_stage, attr), attr
