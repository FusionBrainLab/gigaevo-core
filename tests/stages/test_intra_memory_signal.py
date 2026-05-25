"""Tests for `_derive_intra_signal` — the upstream replacement for the
agent's regex-on-rendered-text exhaustion predicate.

The derive function reads the structured `IntraCardStructuredOutput` (as a
dict) and returns an `IntraMemorySignal` with three-tier severity:

  - healthy   = no clusters with verdict in {regressed, failed}
  - negative  = >=1 negative cluster but trigger conditions don't fire
  - exhausted = original cond_a (>=2 negative clusters) OR cond_b
                (improving == 0 AND catastrophic + n_failed >= 2)

Always populated when intra data exists; downstream prompt grades the
guidance by severity.
"""

from __future__ import annotations

from gigaevo.programs.stages.lineage_memory import (
    IntraMemorySignal,
    _derive_intra_signal,
)


def _card(
    clusters: list[tuple[str, str, int]] | None = None,
    improving: int = 2,
    neutral: int = 0,
    catastrophic: int = 0,
    n_failed: int = 0,
) -> dict:
    """Build a card dict matching IntraCardStructuredOutput.model_dump() shape."""
    return {
        "parent_id": "abc12345",
        "parent_fitness": 0.001,
        "n_attempts": 5,
        "delta_distribution": {
            "min": -0.001,
            "median": 0.0,
            "max": 0.002,
            "improving": improving,
            "neutral": neutral,
            "catastrophic": catastrophic,
            "n_failed": n_failed,
        },
        "tried_strategies": [
            {
                "label": label,
                "verdict": verdict,
                "n_attempts": n,
                "mean_delta": 0.0,
                "n_failed": 0,
                "notes": "",
            }
            for label, verdict, n in (clusters or [])
        ],
        "summary": "test summary",
    }


def test_returns_intra_memory_signal_instance():
    signal = _derive_intra_signal(_card([("a", "improved", 1)]))
    assert isinstance(signal, IntraMemorySignal)


def test_healthy_when_all_improved():
    signal = _derive_intra_signal(
        _card([("a", "improved", 2), ("b", "improved", 1)], improving=3)
    )
    assert signal.severity == "healthy"
    assert signal.n_negative == 0
    assert signal.n_clusters == 2


def test_healthy_when_no_clusters_at_all():
    """Degenerate but consistent: empty tried_strategies → healthy."""
    signal = _derive_intra_signal(_card([], improving=0))
    assert signal.severity == "healthy"
    assert signal.n_clusters == 0


def test_negative_when_single_regressed_cluster():
    signal = _derive_intra_signal(
        _card(
            [("greedy_init", "regressed", 2), ("lookahead", "improved", 1)],
            improving=1,
            catastrophic=1,
        )
    )
    assert signal.severity == "negative"
    assert signal.n_negative == 1


def test_exhausted_cond_a_two_regressed():
    signal = _derive_intra_signal(
        _card(
            [("greedy_init", "regressed", 2), ("lookahead", "regressed", 1)],
            improving=0,
            catastrophic=2,
        )
    )
    assert signal.severity == "exhausted"
    assert signal.n_negative == 2


def test_exhausted_cond_a_regressed_plus_failed():
    signal = _derive_intra_signal(
        _card(
            [("A", "regressed", 2), ("B", "failed", 2)],
            improving=0,
            n_failed=2,
        )
    )
    assert signal.severity == "exhausted"


def test_exhausted_cond_b_dist_only():
    """One cluster, but dist shows catastrophic+n_failed>=2 AND improving==0."""
    signal = _derive_intra_signal(
        _card(
            [("Sole cluster", "regressed", 3)],
            improving=0,
            catastrophic=2,
            n_failed=0,
        )
    )
    assert signal.severity == "exhausted"


def test_negative_when_cond_b_blocked_by_improving():
    """cat+n_failed>=2 but improving>0 → cond_b doesn't fire; only 1 neg cluster."""
    signal = _derive_intra_signal(
        _card(
            [("Sole cluster", "regressed", 3)],
            improving=1,
            catastrophic=2,
            n_failed=0,
        )
    )
    assert signal.severity == "negative"


def test_clusters_include_full_breakdown_not_just_negatives():
    """Per Opus reviewer: signal carries ALL clusters (positive/neutral/negative)
    so the LLM can do the weighting, not just the negative ones."""
    signal = _derive_intra_signal(
        _card(
            [
                ("A", "regressed", 1),
                ("B", "improved", 1),
                ("C", "neutral", 1),
                ("D", "failed", 1),
            ],
            improving=1,
            neutral=1,
            catastrophic=1,
            n_failed=1,
        )
    )
    labels = [c.label for c in signal.clusters]
    assert labels == ["A", "B", "C", "D"]
    assert signal.n_clusters == 4
    assert signal.n_negative == 2


def test_delta_distribution_fields_preserved():
    signal = _derive_intra_signal(
        _card(
            [("a", "improved", 1)],
            improving=3,
            neutral=2,
            catastrophic=1,
            n_failed=1,
        )
    )
    dd = signal.delta_dist
    assert dd["improving"] == 3
    assert dd["neutral"] == 2
    assert dd["catastrophic"] == 1
    assert dd["n_failed"] == 1


def test_missing_distribution_falls_back_to_zero_counts():
    """Defensive: a card with missing dist field should not crash."""
    card = _card([("a", "improved", 1)])
    card["delta_distribution"] = {}
    signal = _derive_intra_signal(card)
    assert signal.severity == "healthy"
    assert signal.delta_dist["improving"] == 0


def test_severity_literal_type():
    """severity must be one of the three documented literals."""
    for clusters, exp_severity in (
        ([("a", "improved", 1)], "healthy"),
        ([("a", "regressed", 1)], "negative"),
        ([("a", "regressed", 1), ("b", "failed", 1)], "exhausted"),
    ):
        signal = _derive_intra_signal(_card(clusters, improving=0, n_failed=1))
        assert signal.severity == exp_severity, (clusters, signal.severity)
