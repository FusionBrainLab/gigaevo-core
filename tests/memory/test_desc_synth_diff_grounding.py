"""Fix #2: idea-card descriptions are grounded in the actual parent→child diff.

The clustering analyzer mints an idea-card whose text was previously a paraphrase
of the mutation's free-form self-report, never checked against the code that ran.
These tests pin the plumbing of the fix: the rep's diff reaches the synthesis
prompt, and the single-member branch (which previously did no LLM authoring) now
routes through the same diff-grounded call. Actual faithfulness is the LLM's job,
gated by a conservative prompt rule; here we verify the diff is delivered.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gigaevo.memory.ideas_tracker.analyzers import (
    ClusteringAnalyzer,
    IdeaCluster,
)
from gigaevo.memory.ideas_tracker.models import EmbeddedIdea, ProgramRecord
from gigaevo.memory.ideas_tracker.schemas import SynthesisedDescription
from tests.fakes.llm_router import FakeMemoryRouter


class _CapturingCalls:
    """Records (step, content) at the analyzer call surface, fixed synthesis back."""

    def __init__(self, response: str = "SYNTHESIZED") -> None:
        self.response = response
        self.captured: list[tuple[str, object]] = []

    def call_structured(self, step, schema, content=""):  # type: ignore[no-untyped-def]
        raise AssertionError("sync call_structured() invoked in async test path")

    async def call_structured_async(self, step, schema, content=""):  # type: ignore[no-untyped-def]
        self.captured.append((step, content))
        assert schema is SynthesisedDescription
        return SynthesisedDescription(description=self.response)


def _build_clustering_analyzer(cap: _CapturingCalls) -> ClusteringAnalyzer:
    with patch(
        "gigaevo.memory.ideas_tracker.analyzers.SentenceTransformer",
        return_value=MagicMock(),
    ):
        analyzer = ClusteringAnalyzer(llm=FakeMemoryRouter(allow_sync=False))
    analyzer.call_structured = cap.call_structured  # type: ignore[method-assign]
    analyzer.call_structured_async = cap.call_structured_async  # type: ignore[method-assign]
    return analyzer


_PARENT = "def solve(x):\n    return x + 1\n"
_CHILD = "def solve(x):\n    return x * 2\n"


@pytest.mark.asyncio
async def test_synthesise_description_passes_diff_slot() -> None:
    cap = _CapturingCalls()
    analyzer = _build_clustering_analyzer(cap)

    out = await analyzer._synthesise_description(
        "rep desc",
        [],
        [],
        child_code=_CHILD,
        parent_code=_PARENT,
    )

    assert out == "SYNTHESIZED"
    step, content = cap.captured[-1]
    assert step == "cluster_desc_synth"
    assert isinstance(content, dict)
    assert "<INSERT_DIFF>" in content
    diff = content["<INSERT_DIFF>"]
    assert "return x + 1" in diff
    assert "return x * 2" in diff


@pytest.mark.asyncio
async def test_single_member_cluster_is_diff_grounded() -> None:
    cap = _CapturingCalls()
    analyzer = _build_clustering_analyzer(cap)

    rep = EmbeddedIdea(
        description="raw drifted self-report", source_program_id="prog-1"
    )
    cluster = IdeaCluster("c1")
    cluster.add_member(rep)
    record = ProgramRecord(
        id="prog-1",
        fitness=0.5,
        generation=2,
        parents=["root"],
        code=_CHILD,
        parent_code=_PARENT,
        improvements=[{"description": "raw drifted self-report", "explanation": ""}],
    )

    idea = await analyzer._cluster_to_idea(cluster, {"prog-1": record})

    # Routed through synthesis instead of echoing the drifted self-report verbatim.
    assert idea.description == "SYNTHESIZED"
    assert any(step == "cluster_desc_synth" for step, _ in cap.captured)
    diff = cap.captured[-1][1]["<INSERT_DIFF>"]
    assert "return x + 1" in diff
    assert "return x * 2" in diff


@pytest.mark.asyncio
async def test_single_member_without_code_stays_verbatim() -> None:
    # No diff to ground on (rep record absent) -> no spurious LLM call, verbatim text.
    cap = _CapturingCalls()
    analyzer = _build_clustering_analyzer(cap)

    rep = EmbeddedIdea(description="verbatim text", source_program_id="missing")
    cluster = IdeaCluster("c1")
    cluster.add_member(rep)

    idea = await analyzer._cluster_to_idea(cluster, {})

    assert idea.description == "verbatim text"
    assert cap.captured == []
