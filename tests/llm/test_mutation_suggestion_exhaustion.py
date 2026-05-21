"""Tests for MutationSuggestionAgent._format_exhaustion_block."""

from __future__ import annotations

from gigaevo.llm.agents.mutation_suggestions import MutationSuggestionAgent


def _make_intra(
    clusters: list[tuple[str, str]], improving=2, catastrophic=0, n_failed=0
) -> str:
    """Build a synthetic intra card mirroring the real renderer."""
    bullets = "\n".join(
        f"- *{name}* — 3 attempt(s), mean delta +0.001, verdict: {verdict} — notes here"
        for name, verdict in clusters
    )
    return (
        "Parent `abc123` (fitness=0.001) has been mutated 5 time(s).\n"
        f"Delta distribution (valid children only): min=-0.001, median=+0.001, "
        f"max=+0.002; improving={improving}, neutral=0, catastrophic={catastrophic}; "
        f"n_failed={n_failed} (excluded)\n\n"
        f"**Already tried:**\n{bullets}\n\n"
        "_summary line_"
    )


def test_empty_intra_returns_empty():
    assert MutationSuggestionAgent._format_exhaustion_block("") == ""


def test_no_intra_section_returns_empty():
    assert MutationSuggestionAgent._format_exhaustion_block(None) == ""  # type: ignore[arg-type]


def test_single_improved_cluster_does_not_trigger():
    intra = _make_intra([("Adaptive step", "improved")], improving=2)
    assert MutationSuggestionAgent._format_exhaustion_block(intra) == ""


def test_single_regressed_cluster_does_not_trigger():
    intra = _make_intra([("Adaptive step", "regressed")], improving=0, catastrophic=1)
    # Only 1 negative cluster; cond_a fails. cond_b needs cat+n_failed>=2.
    assert MutationSuggestionAgent._format_exhaustion_block(intra) == ""


def test_two_regressed_clusters_triggers_cond_a():
    intra = _make_intra(
        [("Greedy init", "regressed"), ("Lookahead refine", "regressed")],
        improving=0,
    )
    banner = MutationSuggestionAgent._format_exhaustion_block(intra)
    assert "EXHAUSTION ALERT" in banner
    assert "Greedy init" in banner
    assert "Lookahead refine" in banner
    assert "2 distinct cluster(s)" in banner


def test_one_regressed_one_failed_triggers_cond_a():
    intra = _make_intra(
        [("Cluster A", "regressed"), ("Cluster B", "failed")],
        improving=0,
        n_failed=1,
    )
    banner = MutationSuggestionAgent._format_exhaustion_block(intra)
    assert banner != ""
    assert "Cluster A" in banner
    assert "Cluster B" in banner


def test_distribution_only_trigger_cond_b():
    # Only 1 cluster, but catastrophic + n_failed = 2, improving = 0 → cond_b
    intra = _make_intra(
        [("Sole cluster", "regressed")], improving=0, catastrophic=2, n_failed=0
    )
    banner = MutationSuggestionAgent._format_exhaustion_block(intra)
    assert banner != ""
    assert "catastrophic + n_failed" in banner


def test_distribution_cond_b_blocked_by_improving():
    # cat+n_failed >= 2 but improving > 0 → no cond_b trigger; only 1 negative cluster
    intra = _make_intra(
        [("Sole cluster", "regressed")], improving=1, catastrophic=2, n_failed=0
    )
    assert MutationSuggestionAgent._format_exhaustion_block(intra) == ""


def test_mixed_two_negative_one_improved_triggers():
    intra = _make_intra(
        [
            ("Mech1", "regressed"),
            ("Mech2", "failed"),
            ("Mech3", "improved"),
        ],
        improving=2,
        n_failed=1,
    )
    banner = MutationSuggestionAgent._format_exhaustion_block(intra)
    assert banner != ""
    # AVOID-LIST contains the negative ones, not the improved
    assert "Mech1" in banner
    assert "Mech2" in banner
    # Full tried context lists all three
    assert "Mech3" in banner


def test_no_heilbron_specific_tokens_in_banner():
    """Banner must be task-agnostic — never names a specific testbed."""
    intra = _make_intra(
        [("Generic A", "regressed"), ("Generic B", "regressed")], improving=0
    )
    banner = MutationSuggestionAgent._format_exhaustion_block(intra)
    for tok in (
        "heilbron",
        "triangle",
        "barycentric",
        "simulated annealing",
        "hotpot",
        "hover",
    ):
        assert tok not in banner.lower(), f"banner leaked '{tok}'"


def test_realistic_cycle4b_intra_triggers():
    """Real intra-card excerpt from cycle-4b parent ae4c95e2 (2 regressed clusters)."""
    intra = (
        "Parent `ae4c95e2` (fitness=0.0089) has been mutated 4 time(s).\n"
        "Delta distribution (valid children only): min=-0.005, median=-0.001, "
        "max=+0.001; improving=0, neutral=0, catastrophic=2; n_failed=2 (excluded from stats above)\n\n"
        "**Already tried:**\n"
        "- *Stress-driven sampling* — 2 attempt(s), mean delta -0.003, verdict: regressed — bad mechanism description\n"
        "- *Fibonacci spacing* — 2 attempt(s), mean delta -0.001, verdict: regressed — boundary issues\n\n"
        "_summary_"
    )
    banner = MutationSuggestionAgent._format_exhaustion_block(intra)
    assert "Stress-driven sampling" in banner
    assert "Fibonacci spacing" in banner
    assert banner.startswith("## EXHAUSTION ALERT")
