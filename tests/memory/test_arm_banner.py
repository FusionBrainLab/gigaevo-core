"""Tests for the one-line memory-arm banner logged at run startup.

The three paper arms differ only in which memory components Hydra wires:

* arm 1  — ``pipeline=standard`` (defaults): Null provider, Null tracker,
  no post_step_hook.
* arm 2′ — ``pipeline=intra_extra_memory ideas_tracker=default memory=none``:
  write side live, read path Null (write-cost-controlled baseline).
* arm 3  — ``+ memory=local``: full read/write loop.

The banner makes the resolved arm verifiable from the first log lines instead
of from ``.hydra/config.yaml`` archaeology after the run.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from loguru import logger
import pytest

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.entrypoint.lineage_memory_pipeline import (
    IntraExtraMemoryPipelineBuilder,
    IntraMemoryPipelineBuilder,
)
from gigaevo.evolution.engine.hooks import IncrementalPostRunHook, NullPostRunHook
from gigaevo.llm.models import MultiModelRouter
from gigaevo.memory.arm_banner import log_memory_arm_banner
from gigaevo.memory.core import MemorySelection
from gigaevo.memory.live_memory_hook import LiveMemoryRefreshHook
from gigaevo.memory.provider import MemoryProvider, NullMemoryProvider
from gigaevo.problems.context import ProblemContext
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.program import Program


class _StubTracker(IncrementalPostRunHook):
    async def on_run_complete(self, storage) -> None:  # type: ignore[no-untyped-def]
        pass

    async def run_increment(self, programs, *, posterior_programs=None) -> None:  # type: ignore[no-untyped-def]
        pass


class _CardProvider(MemoryProvider):
    async def select_cards(
        self,
        program: Program,
        *,
        task_description: str,
        metrics_description: str,
    ) -> MemorySelection:
        return MemorySelection(cards=[], card_ids=[])


def _make_ctx(provider: MemoryProvider | None = None) -> EvolutionContext:
    problem_ctx = MagicMock(spec=ProblemContext)
    problem_ctx.problem_dir = Path("/fake/problem")
    problem_ctx.task_description = "Solve the task."
    problem_ctx.metrics_context = MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="main metric",
                is_primary=True,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
            )
        }
    )
    problem_ctx.is_contextual = False
    kwargs = {} if provider is None else {"memory_provider": provider}
    return EvolutionContext(
        problem_ctx=problem_ctx,
        llm_wrapper=MagicMock(spec=MultiModelRouter),
        storage=MagicMock(spec=ProgramStorage),
        prompts_dir=None,
        **kwargs,
    )


@pytest.fixture
def captured() -> list[str]:
    messages: list[str] = []
    handle = logger.add(messages.append, level="INFO", format="{message}")
    yield messages
    logger.remove(handle)


def _banner_lines(messages: list[str]) -> list[str]:
    # The builders' arm-mismatch warnings share the [Memory][Arm] prefix;
    # only the banner itself carries the provider= field.
    return [m.strip() for m in messages if "[Memory][Arm] provider=" in m]


def test_standard_no_memory_arm(captured):
    log_memory_arm_banner(
        provider=NullMemoryProvider(),
        tracker=NullPostRunHook(),
        post_step_hook=None,
        pipeline_builder=IntraMemoryPipelineBuilder(_make_ctx()),
    )
    (line,) = _banner_lines(captured)
    assert "provider=NullMemoryProvider" in line
    assert "tracker=NullPostRunHook" in line
    assert "post_step_hook=None" in line
    assert "pipeline_builder=IntraMemoryPipelineBuilder" in line


def test_write_only_baseline_arm(captured):
    # arm 2′: tracker + live hook present, read path Null — the banner must
    # show that combination verbatim so a misconfigured arm 3 is caught at
    # startup, not after the run.
    tracker = _StubTracker()
    hook = LiveMemoryRefreshHook(
        tracker=tracker, storage=MagicMock(spec=ProgramStorage)
    )
    log_memory_arm_banner(
        provider=NullMemoryProvider(),
        tracker=tracker,
        post_step_hook=hook,
        pipeline_builder=IntraExtraMemoryPipelineBuilder(_make_ctx()),
    )
    (line,) = _banner_lines(captured)
    assert "provider=NullMemoryProvider" in line
    assert "tracker=_StubTracker" in line
    assert "post_step_hook=LiveMemoryRefreshHook" in line
    assert "pipeline_builder=IntraExtraMemoryPipelineBuilder" in line


def test_full_memory_arm(captured):
    tracker = _StubTracker()
    provider = _CardProvider()
    log_memory_arm_banner(
        provider=provider,
        tracker=tracker,
        post_step_hook=LiveMemoryRefreshHook(
            tracker=tracker, storage=MagicMock(spec=ProgramStorage)
        ),
        pipeline_builder=IntraExtraMemoryPipelineBuilder(_make_ctx(provider)),
    )
    (line,) = _banner_lines(captured)
    assert "provider=_CardProvider" in line
    assert "post_step_hook=LiveMemoryRefreshHook" in line


def test_missing_components_render_as_none(captured):
    log_memory_arm_banner(
        provider=None, tracker=None, post_step_hook=None, pipeline_builder=None
    )
    (line,) = _banner_lines(captured)
    assert line.count("None") == 4
