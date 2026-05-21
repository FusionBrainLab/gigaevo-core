"""IntraMemoryStage (DAG-native): inputs come from upstream stages via
``attach_inputs`` and the framework's ``InputHashCache`` keys on those
inputs — so a second call with the same evaluated-children id list reuses
the cached card via the DAG runner (verified by hash stability), and a new
child id flips the hash.

The stage uses ``llm.with_structured_output(IntraCardStructuredOutput)``; tests
mock that method so the structured-LLM ``ainvoke`` returns a Pydantic instance.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.programs.core_types import ProgramStageResult, StageError
from gigaevo.programs.metrics.context import MetricsContext
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from gigaevo.programs.stages.common import StringContainer, StringList
from gigaevo.programs.stages.lineage_memory import (
    INTRA_MEMORY_CARD_METADATA_KEY,
    IntraCardStructuredOutput,
    IntraDeltaDistribution,
    IntraMemoryStage,
    IntraTriedStrategy,
    _render_intra_card_text,
)

_PRIMARY = "fitness"


def _metrics_context() -> MetricsContext:
    return MetricsContext.from_descriptions(
        primary_key=_PRIMARY,
        primary_description="primary fitness",
        higher_is_better=True,
    )


def _program(
    *,
    score: float | None = None,
    valid: bool = True,
    children: list[str] | None = None,
    code: str = "# tagged",
) -> Program:
    prog = Program(code=code, state=ProgramState.RUNNING)
    if score is not None:
        prog.metrics = {_PRIMARY: score, "is_valid": 1.0 if valid else 0.0}
    if children:
        prog.lineage.children = list(children)
    return prog


def _card_obj() -> IntraCardStructuredOutput:
    return IntraCardStructuredOutput(
        parent_id="p",
        parent_fitness=0.1,
        n_attempts=1,
        delta_distribution=IntraDeltaDistribution(
            min=0.05,
            median=0.05,
            max=0.05,
            improving=1,
            neutral=0,
            catastrophic=0,
        ),
        tried_strategies=[
            IntraTriedStrategy(
                label="x",
                n_attempts=1,
                mean_delta=0.05,
                verdict="improved",
                notes="",
            )
        ],
        untried_hints=["try y"],
        summary="ok",
    )


def _mock_llm() -> MagicMock:
    """Mock LLM whose ``with_structured_output`` returns a structured_llm whose
    ``ainvoke`` always yields the canned IntraCardStructuredOutput.
    """
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=_card_obj())
    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=structured)
    # Convenience accessor for tests to read invocation counts.
    llm._structured = structured
    return llm


def _build_stage(
    *,
    storage: MagicMock,
    llm: MagicMock,
    task_description: str = "",
) -> IntraMemoryStage:
    return IntraMemoryStage(
        llm=llm,
        storage=storage,
        metrics_context=_metrics_context(),
        max_children=8,
        task_description=task_description,
        timeout=30.0,
    )


@pytest.mark.asyncio
async def test_intra_memory_empty_children_short_circuits() -> None:
    """No children_ids → no LLM call, empty StringContainer."""
    parent = _program(score=0.1)
    storage = MagicMock()
    storage.mget = AsyncMock(return_value=[])
    storage.update = AsyncMock()

    llm = _mock_llm()
    stage = _build_stage(storage=storage, llm=llm)
    stage.attach_inputs({"children_ids": None})

    out = await stage.compute(parent)
    assert isinstance(out, StringContainer) and out.data == ""
    assert llm._structured.ainvoke.await_count == 0


@pytest.mark.asyncio
async def test_intra_memory_renders_card_with_children() -> None:
    """children_ids non-empty + parent has fitness → LLM called, card written."""
    child = _program(score=0.15)
    parent = _program(score=0.1, children=[child.id])

    storage = MagicMock()
    storage.mget = AsyncMock(return_value=[child])
    storage.update = AsyncMock()

    llm = _mock_llm()
    stage = _build_stage(storage=storage, llm=llm)
    stage.attach_inputs({"children_ids": StringList(items=[child.id])})

    out = await stage.compute(parent)
    assert isinstance(out, StringContainer)
    assert out.data
    assert llm._structured.ainvoke.await_count == 1
    # Card persisted on parent metadata for future mutator prompts.
    assert isinstance(parent.get_metadata(INTRA_MEMORY_CARD_METADATA_KEY), str)
    storage.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_intra_memory_cache_hash_stable_for_same_inputs() -> None:
    """Same children_ids list → identical inputs hash → framework cache HIT."""
    child = _program(score=0.15)
    storage = MagicMock()
    storage.mget = AsyncMock(return_value=[child])
    storage.update = AsyncMock()

    stage_a = _build_stage(storage=storage, llm=_mock_llm())
    stage_a.attach_inputs({"children_ids": StringList(items=[child.id])})
    hash_a = stage_a.compute_inputs_hash()

    stage_b = _build_stage(storage=storage, llm=_mock_llm())
    stage_b.attach_inputs({"children_ids": StringList(items=[child.id])})
    hash_b = stage_b.compute_inputs_hash()

    assert hash_a is not None
    assert hash_a == hash_b, (
        "Same children_ids + memory_cards must hash identically so the "
        "framework cache reuses the prior result"
    )


@pytest.mark.asyncio
async def test_intra_memory_cache_hash_flips_when_child_added() -> None:
    """Adding a new child id flips the inputs hash → cache MISS, re-run."""
    child1 = _program(score=0.15)
    child2 = _program(score=0.20)
    storage = MagicMock()
    storage.mget = AsyncMock(return_value=[])
    storage.update = AsyncMock()

    stage_before = _build_stage(storage=storage, llm=_mock_llm())
    stage_before.attach_inputs({"children_ids": StringList(items=[child1.id])})
    hash_before = stage_before.compute_inputs_hash()

    stage_after = _build_stage(storage=storage, llm=_mock_llm())
    stage_after.attach_inputs(
        {
            "children_ids": StringList(items=[child1.id, child2.id]),
        }
    )
    hash_after = stage_after.compute_inputs_hash()

    assert hash_before is not None and hash_after is not None
    assert hash_before != hash_after


@pytest.mark.asyncio
async def test_intra_memory_payload_includes_error_summary_for_failed_children() -> (
    None
):
    """Failed children must carry their formatted error text into the LLM payload."""
    failed_child = _program(score=0.0, valid=False)
    failed_child.stage_results["Mutate"] = ProgramStageResult.failure(
        error=StageError(
            type="IndexError",
            message="IndexError when k=4 reached",
            stage="Mutate",
        ),
    )
    parent = _program(score=0.1, children=[failed_child.id])

    storage = MagicMock()
    storage.mget = AsyncMock(return_value=[failed_child])
    storage.update = AsyncMock()

    llm = _mock_llm()
    stage = _build_stage(storage=storage, llm=llm)
    stage.attach_inputs(
        {
            "children_ids": StringList(items=[failed_child.id]),
        }
    )

    await stage.compute(parent)
    assert llm._structured.ainvoke.await_count == 1
    messages = llm._structured.ainvoke.await_args.args[0]
    user_msg_content = messages[1].content
    assert "error_summary" in user_msg_content
    assert "IndexError when k=4 reached" in user_msg_content


# ---------------------------------------------------------------------------
# Invalid-child sentinel filtering: delta_distribution / mean_delta must not
# blend the -1000 invalid-program sentinel into central tendency stats.
# Failures are tracked separately via `n_failed` counters.
# ---------------------------------------------------------------------------


def test_intra_delta_distribution_carries_n_failed_counter() -> None:
    """Schema exposes an `n_failed` field on the distribution (default 0)."""
    dist = IntraDeltaDistribution(
        min=0.05,
        median=0.05,
        max=0.05,
        improving=1,
        neutral=0,
        catastrophic=0,
    )
    assert dist.n_failed == 0

    dist_with_failures = IntraDeltaDistribution(
        min=0.05,
        median=0.05,
        max=0.05,
        improving=1,
        neutral=0,
        catastrophic=0,
        n_failed=3,
    )
    assert dist_with_failures.n_failed == 3


def test_intra_tried_strategy_accepts_null_mean_delta_and_n_failed() -> None:
    """All-invalid clusters must be expressible: mean_delta=None, verdict='failed'."""
    failed_only = IntraTriedStrategy(
        label="grid_naive",
        n_attempts=2,
        mean_delta=None,
        verdict="failed",
        n_failed=2,
        notes="IndexError: barycentric conversion out of bounds",
    )
    assert failed_only.mean_delta is None
    assert failed_only.n_failed == 2
    assert failed_only.verdict == "failed"


def test_render_intra_card_reports_n_failed_separately_from_delta_stats() -> None:
    """Renderer shows `n_failed=N` on the distribution line; cluster reports
    failure count alongside attempts; null mean_delta renders as 'n/a'."""
    card = {
        "parent_id": "p12345678",
        "parent_fitness": 0.10,
        "n_attempts": 4,
        "delta_distribution": {
            "min": 0.01,
            "median": 0.03,
            "max": 0.05,
            "improving": 2,
            "neutral": 0,
            "catastrophic": 0,
            "n_failed": 2,
        },
        "tried_strategies": [
            {
                "label": "valid_cluster",
                "n_attempts": 2,
                "mean_delta": 0.03,
                "verdict": "improved",
                "n_failed": 0,
                "notes": "",
            },
            {
                "label": "broken_cluster",
                "n_attempts": 2,
                "mean_delta": None,
                "verdict": "failed",
                "n_failed": 2,
                "notes": "boundary violation",
            },
        ],
        "untried_hints": ["try sobol"],
        "summary": "two valid wins; structured grid keeps failing",
    }
    rendered = _render_intra_card_text(card)
    # n_failed surfaces explicitly, not buried in min/median
    assert "n_failed=2" in rendered
    # Polluted -500/-1000 numbers must not leak in via a default placeholder
    assert "-1000" not in rendered
    assert "-500" not in rendered
    # Cluster with failures shows the failed count
    assert "2 failed" in rendered or "failed=2" in rendered
    # Cluster with no valid children must not show a numeric mean_delta
    broken_line = next(
        line for line in rendered.splitlines() if "broken_cluster" in line
    )
    assert "None" not in broken_line  # don't dump literal Python None
    assert "mean delta n/a" in broken_line or "verdict: failed" in broken_line


# ---------------------------------------------------------------------------
# Diff-vs-full-code payload format. The intra analyst's primary task is
# clustering children by what their code CHANGED relative to the parent.
# Shipping each child's full source repeats the parent's boilerplate N times
# and forces the LLM to mentally diff before clustering. Default to a unified
# diff; fall back to full code only when (a) the diff would be no smaller
# than the file or (b) the child is invalid (we need full-line context for
# error_summary references).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_intra_payload_emits_diff_for_small_mutation() -> None:
    """Small change → child entry carries `diff` field (unified format), not `code`.

    Real programs are hundreds of lines; the diff form wins by huge margins
    there. Below we simulate that with a 30-line program where only one
    constant changes — diff (≈10 lines of hunk + 5 lines context × 2) is
    far smaller than the full body.
    """
    # 30-line body where only one constant moves.
    boilerplate = "\n".join(
        [
            "import math",
            "",
            "def shared_helper_a(x):",
            "    return math.sqrt(x) + 1",
            "",
            "def shared_helper_b(x):",
            "    return math.log(x + 1) + 2",
            "",
            "def shared_helper_c(x):",
            "    return math.sin(x) * 3",
            "",
            "def shared_helper_d(x):",
            "    return math.cos(x) * 4",
            "",
            "def shared_helper_e(x):",
            "    return math.exp(-x) * 5",
            "",
            "def shared_helper_f(x):",
            "    return math.tanh(x) * 6",
            "",
            "",
        ]
    )
    parent_code = boilerplate + "def f(x):\n    return x + 1\n"
    child_code = boilerplate + "def f(x):\n    return x + 42\n"
    child = _program(score=0.15, code=child_code)
    parent = _program(score=0.1, children=[child.id], code=parent_code)

    storage = MagicMock()
    storage.mget = AsyncMock(return_value=[child])
    storage.update = AsyncMock()

    llm = _mock_llm()
    stage = _build_stage(storage=storage, llm=llm)
    stage.attach_inputs({"children_ids": StringList(items=[child.id])})

    await stage.compute(parent)
    user_msg = llm._structured.ainvoke.await_args.args[0][1].content
    assert '"change_form": "diff"' in user_msg
    assert '"diff":' in user_msg
    assert "@@" in user_msg, "unified-diff hunk marker missing from payload"
    # parent code is in the payload (parent.code); the child's full source is NOT.
    # Only one `"code":` occurrence in the JSON (the parent's).
    assert user_msg.count('"code":') == 1


@pytest.mark.asyncio
async def test_intra_payload_emits_full_code_for_structural_rewrite() -> None:
    """When the unified diff is no smaller than the child source, ship full code."""
    parent_code = "def f(x):\n    return x\n"
    # Completely different shape — diff = parent + child contents → larger than child alone.
    child_code = (
        "import math\n"
        "class Solver:\n"
        "    def compute(self, n):\n"
        "        return math.factorial(n)\n"
    )
    child = _program(score=0.15, code=child_code)
    parent = _program(score=0.1, children=[child.id], code=parent_code)

    storage = MagicMock()
    storage.mget = AsyncMock(return_value=[child])
    storage.update = AsyncMock()

    llm = _mock_llm()
    stage = _build_stage(storage=storage, llm=llm)
    stage.attach_inputs({"children_ids": StringList(items=[child.id])})

    await stage.compute(parent)
    user_msg = llm._structured.ainvoke.await_args.args[0][1].content
    assert '"change_form": "full_code"' in user_msg
    assert "math.factorial" in user_msg  # child source carried in full
    assert '"diff":' not in user_msg


@pytest.mark.asyncio
async def test_intra_payload_emits_full_code_for_invalid_child() -> None:
    """Invalid children always ship full code so error_summary line refs work."""
    parent_code = "def f(x):\n    return x + 1\n"
    child_code = "def f(x):\n    return x + 2  # boom\n"  # tiny diff
    child = _program(score=0.0, valid=False, code=child_code)
    parent = _program(score=0.1, children=[child.id], code=parent_code)

    storage = MagicMock()
    storage.mget = AsyncMock(return_value=[child])
    storage.update = AsyncMock()

    llm = _mock_llm()
    stage = _build_stage(storage=storage, llm=llm)
    stage.attach_inputs({"children_ids": StringList(items=[child.id])})

    await stage.compute(parent)
    user_msg = llm._structured.ainvoke.await_args.args[0][1].content
    # Invalid → full code, even though the diff would be tiny.
    assert '"change_form": "full_code"' in user_msg
    assert "boom" in user_msg
