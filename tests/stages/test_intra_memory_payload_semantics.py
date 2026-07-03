"""Semantics of the intra-memory LLM payload and rendered card.

Child deltas must be ORIENTED (positive = child improved on parent regardless
of metric direction), crossover children must be attributed to their base
parent (donor-side diffs mostly reflect the OTHER parent's code), and the
rendered card must describe the recency window honestly instead of claiming
the full mutation count.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.evolution.mutation.constants import (
    MUTATION_MEMORY_BASE_ID_METADATA_KEY,
    MUTATION_OUTPUT_METADATA_KEY,
)
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.common import StringList
from gigaevo.programs.stages.lineage_memory import (
    INTRA_SYSTEM_PROMPT_TEMPLATE,
    IntraCardLLMOutput,
    IntraMemoryStage,
    IntraTriedStrategyLLM,
    _render_intra_card_text,
)

_PRIMARY = "fitness"


def _metrics_context(*, higher_is_better: bool = True) -> MetricsContext:
    return MetricsContext.from_descriptions(
        primary_key=_PRIMARY,
        primary_description="primary fitness",
        higher_is_better=higher_is_better,
    )


def _program(
    *,
    score: float | None = None,
    valid: bool = True,
    parents: list[str] | None = None,
    children: list[str] | None = None,
    base_parent_id: str | None = None,
    code: str = "# tagged",
) -> Program:
    prog = Program(code=code, state=ProgramState.RUNNING)
    if score is not None:
        prog.metrics = {_PRIMARY: score, "is_valid": 1.0 if valid else 0.0}
    if parents:
        prog.lineage.parents = list(parents)
    if children:
        prog.lineage.children = list(children)
    if base_parent_id is not None:
        prog.set_metadata(MUTATION_MEMORY_BASE_ID_METADATA_KEY, base_parent_id)
    return prog


def _mock_llm() -> MagicMock:
    structured = MagicMock()
    structured.ainvoke = AsyncMock(
        return_value=IntraCardLLMOutput(
            tried_strategies=[
                IntraTriedStrategyLLM(
                    label="x",
                    child_indices=[0],
                    representative_anchors=["alpha=1"],
                    mechanism_note="speeds inner loop",
                )
            ],
            summary="ok",
        )
    )
    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=structured)
    llm._structured = structured
    return llm


def _build_stage(
    *,
    storage: MagicMock,
    llm: MagicMock,
    higher_is_better: bool = True,
) -> IntraMemoryStage:
    return IntraMemoryStage(
        llm=llm,
        storage=storage,
        metrics_context=_metrics_context(higher_is_better=higher_is_better),
        max_children=8,
        timeout=30.0,
    )


async def _run_stage(
    parent: Program, child: Program, *, higher_is_better: bool = True
) -> tuple[MagicMock, str]:
    storage = MagicMock()
    storage.mget = AsyncMock(return_value=[child])
    storage.update = AsyncMock()
    llm = _mock_llm()
    stage = _build_stage(storage=storage, llm=llm, higher_is_better=higher_is_better)
    stage.attach_inputs({"children_ids": StringList(items=[child.id])})
    out = await stage.compute(parent)
    return llm, out.data


def _payload_children(llm: MagicMock) -> list[dict]:
    user_msg = llm._structured.ainvoke.await_args.args[0][1].content
    payload = json.loads(user_msg.split("Input JSON follows:\n\n", 1)[1])
    return payload["children"]


@pytest.mark.asyncio
async def test_child_delta_oriented_positive_when_minimize_child_improves() -> None:
    """Lower-is-better metric: child 0.3 vs parent 0.5 is an IMPROVEMENT and
    must reach the LLM as a positive delta, matching the ancestral trail's
    orientation convention."""
    child = _program(score=0.3)
    parent = _program(score=0.5, children=[child.id])

    llm, _ = await _run_stage(parent, child, higher_is_better=False)
    (entry,) = _payload_children(llm)
    assert entry["delta"] == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_child_delta_raw_when_maximize_child_improves() -> None:
    """Higher-is-better metric: orientation is the identity."""
    child = _program(score=0.15)
    parent = _program(score=0.1, children=[child.id])

    llm, _ = await _run_stage(parent, child, higher_is_better=True)
    (entry,) = _payload_children(llm)
    assert entry["delta"] == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_minimize_improvement_buckets_as_improving_in_card() -> None:
    """The rendered distribution must count a minimize-metric improvement as
    improving, not catastrophic (buckets consume the oriented delta)."""
    child = _program(score=0.3)
    parent = _program(score=0.5, children=[child.id])

    _, rendered = await _run_stage(parent, child, higher_is_better=False)
    assert "improving=1" in rendered
    assert "catastrophic=0" in rendered
    assert "verdict: improved" in rendered


@pytest.mark.asyncio
async def test_crossover_child_marked_base_for_base_parent() -> None:
    """Two-parent child whose recorded base is THIS parent → crossover_role='base'."""
    parent = _program(score=0.1)
    child = _program(
        score=0.15,
        parents=[parent.id, "donor-parent-id"],
        base_parent_id=parent.id,
    )
    parent.lineage.children = [child.id]

    llm, _ = await _run_stage(parent, child)
    (entry,) = _payload_children(llm)
    assert entry["crossover_role"] == "base"


@pytest.mark.asyncio
async def test_crossover_child_marked_donor_for_other_parent() -> None:
    """Two-parent child whose recorded base is the OTHER parent → 'donor':
    its diff against this parent mostly reflects the base parent's code, not
    a mutation move tried on this parent."""
    parent = _program(score=0.1)
    child = _program(
        score=0.15,
        parents=["base-parent-id", parent.id],
        base_parent_id="base-parent-id",
    )
    parent.lineage.children = [child.id]

    llm, _ = await _run_stage(parent, child)
    (entry,) = _payload_children(llm)
    assert entry["crossover_role"] == "donor"


@pytest.mark.asyncio
async def test_crossover_role_falls_back_to_first_parent_without_stamp() -> None:
    """Children persisted before the base-id stamp existed: treat parents[0]
    as the base (the engine's own default)."""
    parent = _program(score=0.1)
    child = _program(score=0.15, parents=["other-id", parent.id])
    parent.lineage.children = [child.id]

    llm, _ = await _run_stage(parent, child)
    (entry,) = _payload_children(llm)
    assert entry["crossover_role"] == "donor"


@pytest.mark.asyncio
async def test_minimize_regression_buckets_as_catastrophic() -> None:
    """Lower-is-better metric: child 0.7 vs parent 0.5 is a REGRESSION —
    negative oriented delta, catastrophic bucket."""
    child = _program(score=0.7)
    parent = _program(score=0.5, children=[child.id])

    llm, rendered = await _run_stage(parent, child, higher_is_better=False)
    (entry,) = _payload_children(llm)
    assert entry["delta"] == pytest.approx(-0.2)
    assert "catastrophic=1" in rendered
    assert "improving=0" in rendered


@pytest.mark.asyncio
async def test_crossover_role_falls_back_to_mutation_output_base_parent() -> None:
    """Children without the base-id stamp but with the mutator's structured
    output recover the base from its 1-based ``base_parent`` index (same
    resolution the ideas tracker uses)."""
    parent = _program(score=0.1)
    child = _program(score=0.15, parents=["other-id", parent.id])
    child.set_metadata(
        MUTATION_OUTPUT_METADATA_KEY,
        {"archetype": "explore", "justification": "swap base", "base_parent": 2},
    )
    parent.lineage.children = [child.id]

    llm, _ = await _run_stage(parent, child)
    (entry,) = _payload_children(llm)
    assert entry["crossover_role"] == "base"


@pytest.mark.asyncio
async def test_crossover_role_recovers_base_from_letter_base_parent() -> None:
    """Diff-operator children emit ``base_parent`` as a namespace LETTER ('B' =
    parent 2). The stamp-less fallback must resolve it via base_parent_index, not
    treat the letter as a non-int and mis-default to parents[0]."""
    parent = _program(score=0.1)
    child = _program(score=0.15, parents=["other-id", parent.id])
    child.set_metadata(
        MUTATION_OUTPUT_METADATA_KEY,
        {"archetype": "explore", "justification": "swap base", "base_parent": "B"},
    )
    parent.lineage.children = [child.id]

    llm, _ = await _run_stage(parent, child)
    (entry,) = _payload_children(llm)
    assert entry["crossover_role"] == "base"


@pytest.mark.asyncio
async def test_invalid_crossover_child_carries_role_and_full_code() -> None:
    parent = _program(score=0.1)
    child = _program(
        score=-1000.0,
        valid=False,
        parents=["base-id", parent.id],
        base_parent_id="base-id",
    )
    parent.lineage.children = [child.id]

    llm, _ = await _run_stage(parent, child)
    (entry,) = _payload_children(llm)
    assert entry["crossover_role"] == "donor"
    assert entry["change_form"] == "full_code"


@pytest.mark.asyncio
async def test_payload_reads_mutator_self_report_from_mutation_output() -> None:
    """archetype/justification come from the ``mutation_output`` metadata the
    operator actually stamps (a bare ``mutation`` key is never written)."""
    parent = _program(score=0.1)
    child = _program(score=0.15, parents=[parent.id])
    child.set_metadata(
        MUTATION_OUTPUT_METADATA_KEY,
        {"archetype": "explore", "justification": "tighten radius"},
    )
    parent.lineage.children = [child.id]

    llm, _ = await _run_stage(parent, child)
    (entry,) = _payload_children(llm)
    assert entry["mutation_archetype"] == "explore"
    assert entry["mutation_justification"] == "tighten radius"


@pytest.mark.asyncio
async def test_single_parent_child_carries_no_crossover_role() -> None:
    parent = _program(score=0.1)
    child = _program(score=0.15, parents=[parent.id])
    parent.lineage.children = [child.id]

    llm, _ = await _run_stage(parent, child)
    (entry,) = _payload_children(llm)
    assert "crossover_role" not in entry


def test_render_intra_card_describes_recency_window_not_total_mutations() -> None:
    """The card covers only the selector's recency window, so the header must
    not claim a total mutation count."""
    card = {
        "parent_id": "p12345678",
        "parent_fitness": 0.10,
        "n_attempts": 4,
        "delta_distribution": {
            "min": 0.01,
            "median": 0.03,
            "max": 0.05,
            "improving": 2,
            "neutral": 2,
            "catastrophic": 0,
            "n_failed": 0,
        },
        "tried_strategies": [],
        "summary": "steady tweaks",
    }
    rendered = _render_intra_card_text(card)
    assert "has been mutated" not in rendered
    assert "4 most recent" in rendered


def test_intra_prompt_claims_window_and_oriented_deltas() -> None:
    """The analyst prompt must not overclaim completeness and must describe
    the delta as oriented."""
    assert "EVERY child" not in INTRA_SYSTEM_PROMPT_TEMPLATE
    assert "most recent" in INTRA_SYSTEM_PROMPT_TEMPLATE
    assert "oriented" in INTRA_SYSTEM_PROMPT_TEMPLATE.lower()
    assert "crossover_role" in INTRA_SYSTEM_PROMPT_TEMPLATE
