"""Tests for the new INTRA STRATEGY SIGNAL banner produced by
``MutationSuggestionAgent._format_intra_signal_block``.

This replaces the prior regex-on-rendered-markdown
``_format_exhaustion_block`` (deleted). The new banner is:

  - Always emitted when a structured signal exists (informational, severity-tiered).
  - Computed upstream by ``IntraMemoryStage._derive_intra_signal``; the
    agent renders it without any pattern-matching on the rendered card.
  - Compact (<1kB even in the worst case) — the prior 3kB "HARD CONSTRAINT"
    lecture is gone; severity-graduated guidance lives in system.txt.
  - Task-agnostic: never names a specific testbed.
"""

from __future__ import annotations

from gigaevo.llm.agents.mutation_suggestions import MutationSuggestionAgent


def _signal(
    severity: str = "healthy",
    clusters: list[tuple[str, str, int]] | None = None,
    improving: int = 2,
    neutral: int = 0,
    catastrophic: int = 0,
    n_failed: int = 0,
) -> dict:
    clusters = clusters or []
    return {
        "severity": severity,
        "n_clusters": len(clusters),
        "n_negative": sum(1 for _, v, _ in clusters if v in {"regressed", "failed"}),
        "clusters": [
            {"label": label, "verdict": verdict, "n_attempts": n}
            for label, verdict, n in clusters
        ],
        "delta_dist": {
            "min": -0.001,
            "median": 0.0,
            "max": 0.002,
            "improving": improving,
            "neutral": neutral,
            "catastrophic": catastrophic,
            "n_failed": n_failed,
        },
    }


def test_none_signal_returns_empty():
    assert MutationSuggestionAgent._format_intra_signal_block(None) == ""


def test_missing_signal_returns_empty():
    assert MutationSuggestionAgent._format_intra_signal_block({}) == ""


def test_healthy_signal_renders_informational():
    """Per Opus reviewer: surface ALWAYS so the LLM can do weighting itself."""
    sig = _signal(
        severity="healthy",
        clusters=[("variant_a", "improved", 2), ("variant_b", "improved", 1)],
        improving=3,
    )
    block = MutationSuggestionAgent._format_intra_signal_block(sig)
    assert block != ""
    assert "INTRA STRATEGY SIGNAL" in block
    assert "healthy" in block.lower()


def test_negative_signal_has_severity_badge():
    sig = _signal(
        severity="negative",
        clusters=[("a", "regressed", 2), ("b", "improved", 1)],
        improving=1,
        catastrophic=1,
    )
    block = MutationSuggestionAgent._format_intra_signal_block(sig)
    assert "negative" in block.lower()
    # Full cluster breakdown (per Opus reviewer)
    assert "a" in block
    assert "b" in block


def test_exhausted_signal_has_severity_badge():
    sig = _signal(
        severity="exhausted",
        clusters=[("greedy", "regressed", 2), ("lookahead", "failed", 2)],
        improving=0,
        n_failed=2,
    )
    block = MutationSuggestionAgent._format_intra_signal_block(sig)
    assert "exhausted" in block.lower()


def test_block_shows_all_clusters_not_just_negatives():
    """Per Opus reviewer: render full positive/neutral/negative breakdown
    so a signal that fires on a single regressed cluster amid four
    improving ones is not training the LLM to discount it."""
    sig = _signal(
        severity="negative",
        clusters=[
            ("Mech_neg", "regressed", 1),
            ("Mech_pos_1", "improved", 2),
            ("Mech_pos_2", "improved", 1),
            ("Mech_neutral", "neutral", 1),
        ],
        improving=3,
        neutral=1,
        catastrophic=1,
    )
    block = MutationSuggestionAgent._format_intra_signal_block(sig)
    for label in ("Mech_neg", "Mech_pos_1", "Mech_pos_2", "Mech_neutral"):
        assert label in block


def test_block_shows_delta_distribution():
    sig = _signal(
        severity="negative",
        clusters=[("a", "regressed", 1)],
        improving=1,
        neutral=2,
        catastrophic=3,
        n_failed=4,
    )
    block = MutationSuggestionAgent._format_intra_signal_block(sig)
    assert "improving" in block.lower()
    # All four numbers should appear
    for n in (1, 2, 3, 4):
        assert str(n) in block


def test_block_shows_negative_count_ratio():
    """Severity line should help the LLM see the proportion at a glance."""
    sig = _signal(
        severity="exhausted",
        clusters=[("a", "regressed", 1), ("b", "failed", 1), ("c", "improved", 1)],
        improving=1,
        n_failed=1,
    )
    block = MutationSuggestionAgent._format_intra_signal_block(sig)
    # n_negative=2, n_clusters=3 → "2/3" or "2 of 3" should appear
    assert "2" in block and "3" in block


def test_block_is_compact_under_800_chars():
    """No 3kB lecture — the hard banner is gone (per all 3 reviewers)."""
    sig = _signal(
        severity="exhausted",
        clusters=[
            ("cluster_one", "regressed", 3),
            ("cluster_two", "regressed", 2),
            ("cluster_three", "failed", 2),
            ("cluster_four", "improved", 1),
            ("cluster_five", "neutral", 1),
        ],
        improving=1,
        neutral=1,
        catastrophic=2,
        n_failed=2,
    )
    block = MutationSuggestionAgent._format_intra_signal_block(sig)
    assert len(block) < 800, f"signal block too long ({len(block)} chars): {block!r}"


def test_block_is_task_agnostic():
    sig = _signal(
        severity="exhausted",
        clusters=[("Generic A", "regressed", 1), ("Generic B", "failed", 1)],
        improving=0,
        n_failed=1,
    )
    block = MutationSuggestionAgent._format_intra_signal_block(sig)
    for tok in (
        "heilbron",
        "triangle",
        "barycentric",
        "annealing",
        "hotpot",
        "hover",
        "ifbench",
    ):
        assert tok not in block.lower(), f"signal leaked '{tok}'"


def test_block_does_not_contain_hard_constraint_language():
    """The new design does NOT hard-override anything — that lives in
    system.txt as graded guidance."""
    sig = _signal(
        severity="exhausted",
        clusters=[("a", "regressed", 1), ("b", "regressed", 1)],
        improving=0,
    )
    block = MutationSuggestionAgent._format_intra_signal_block(sig)
    lowered = block.lower()
    for forbidden in ("hard constraint", "overrides", "must name", "rejected here"):
        assert forbidden not in lowered, (
            f"signal block leaked old hard-banner language: '{forbidden}'"
        )


def test_old_exhaustion_block_helper_is_removed():
    """The old regex-based helper must no longer exist on the agent."""
    assert not hasattr(MutationSuggestionAgent, "_format_exhaustion_block"), (
        "Old _format_exhaustion_block helper should be deleted; the trigger "
        "now lives in IntraMemoryStage._derive_intra_signal."
    )
    for name in (
        "_CLUSTER_BULLET_RE",
        "_IMPROVING_RE",
        "_CATASTROPHIC_RE",
        "_N_FAILED_RE",
    ):
        assert not hasattr(MutationSuggestionAgent, name), (
            f"Old regex constant {name} should be deleted with the helper."
        )
