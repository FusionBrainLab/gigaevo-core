"""Reverse repacking + memory-block placement, default-off behind builder flags."""

from __future__ import annotations

import inspect
from pathlib import Path

from omegaconf import OmegaConf

from gigaevo.entrypoint.lineage_memory_pipeline import (
    GuidedMutationPipelineBuilder,
    MemoryGuidedMutationPipelineBuilder,
)
from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_CANDIDATE_SLATE_METADATA_KEY,
    MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY,
    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
)
from gigaevo.memory.provider import MemoryProvider
from gigaevo.memory.read.auction import AuctionBid
from gigaevo.memory.read.reader import MemorySelection
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.common import StringContainer
from gigaevo.programs.stages.memory_context import MemoryContextStage
from gigaevo.programs.stages.mutation_context import MutationContextStage

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _metrics_context() -> MetricsContext:
    return MetricsContext(
        specs={
            "score": MetricSpec(
                description="main score",
                is_primary=True,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=100.0,
            ),
        }
    )


class FixedProvider(MemoryProvider):
    def __init__(self, selection: MemorySelection) -> None:
        self._selection = selection

    async def select_cards(
        self,
        program: Program,
        *,
        task_description: str,
        metrics_description: str,
        parent_context: str | None = None,
    ) -> MemorySelection:
        return self._selection


def _memory_stage(**overrides) -> MemoryContextStage:
    selection = MemorySelection(
        cards=("best card text", "worst card text"),
        card_ids=("best-id", "worst-id"),
    )
    params = {
        "memory_provider": FixedProvider(selection),
        "task_description": "task",
        "metrics_description": "metrics",
        "timeout": 5.0,
    }
    params.update(overrides)
    return MemoryContextStage(**params)


def _prog() -> Program:
    return Program(code="def solve(): return 0", state=ProgramState.RUNNING)


async def test_memory_context_default_keeps_selection_order():
    stage = _memory_stage()
    stage.attach_inputs({})
    out = await stage.compute(_prog())
    assert out.data == (
        "[card 1] id=best-id\nbest card text\n\n[card 2] id=worst-id\nworst card text"
    )


async def test_memory_context_reverse_repack_renders_worst_first():
    stage = _memory_stage(reverse_repack=True)
    stage.attach_inputs({})
    prog = _prog()
    out = await stage.compute(prog)
    assert out.data == (
        "[card 1] id=worst-id\nworst card text\n\n[card 2] id=best-id\nbest card text"
    )
    assert prog.metadata[MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY] == [
        "best-id",
        "worst-id",
    ]


async def test_memory_context_no_card_control_withholds_rendered_cards():
    bid = AuctionBid(
        card_id="best-id",
        posterior_a=10.0,
        posterior_b=1.0,
        theta=0.9,
        baseline_a=3.0,
        baseline_b=3.0,
        baseline_theta=0.5,
        selected=True,
    )
    selection = MemorySelection(
        cards=("best card text",),
        card_ids=("best-id",),
        slate=(bid,),
    )
    stage = _memory_stage(
        memory_provider=FixedProvider(selection),
        no_card_control_probability=1.0,
    )
    stage.attach_inputs({})
    prog = _prog()

    out = await stage.compute(prog)

    assert out.data == ""
    assert prog.metadata[MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY] == []
    assert prog.metadata[MUTATION_MEMORY_NO_CARD_CONTROL_METADATA_KEY] is True
    assert prog.metadata[MUTATION_MEMORY_CANDIDATE_SLATE_METADATA_KEY] == [
        bid.model_dump()
    ]


def _mutation_inputs() -> dict:
    return {
        "metrics": None,
        "insights": None,
        "lineage_ancestors": None,
        "lineage_descendants": None,
        "evolutionary_statistics": None,
        "formatted": StringContainer(data="FORMATTED TAIL"),
        "memory": StringContainer(data="MEMORY BLOCK"),
    }


async def test_mutation_context_default_places_memory_before_formatted():
    stage = MutationContextStage(metrics_context=_metrics_context(), timeout=5.0)
    stage.attach_inputs(_mutation_inputs())
    out = await stage.compute(_prog())
    assert out.data.index("MEMORY BLOCK") < out.data.index("FORMATTED TAIL")


async def test_mutation_context_memory_last_places_memory_at_end():
    stage = MutationContextStage(
        metrics_context=_metrics_context(), timeout=5.0, memory_last=True
    )
    stage.attach_inputs(_mutation_inputs())
    out = await stage.compute(_prog())
    assert out.data.index("FORMATTED TAIL") < out.data.index("MEMORY BLOCK")
    assert out.data.rstrip().endswith("MEMORY BLOCK")


def test_flags_default_off():
    assert (
        inspect.signature(MemoryContextStage.__init__)
        .parameters["reverse_repack"]
        .default
        is False
    )
    assert (
        inspect.signature(MutationContextStage.__init__)
        .parameters["memory_last"]
        .default
        is False
    )
    assert (
        inspect.signature(GuidedMutationPipelineBuilder.__init__)
        .parameters["memory_block_last"]
        .default
        is False
    )
    extra = inspect.signature(MemoryGuidedMutationPipelineBuilder.__init__).parameters
    assert extra["reverse_repack"].default is False
    assert extra["no_card_control_probability"].default == 0.0
    assert extra["memory_block_last"].default is False


def test_intra_extra_yaml_ships_flags_off():
    cfg = OmegaConf.load(_REPO_ROOT / "config" / "pipeline" / "memory_guided.yaml")
    assert cfg.pipeline_builder.reverse_repack is False
    assert cfg.pipeline_builder.no_card_control_probability == 0.10
    assert cfg.pipeline_builder.memory_block_last is False
