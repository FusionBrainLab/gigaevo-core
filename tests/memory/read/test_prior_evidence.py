"""Evicted-card evidence persistence for empirical-Bayes cold priors."""

from __future__ import annotations

import json


def test_jsonl_evicted_evidence_round_trip_last_wins_and_skips_malformed(
    make_card, make_event, tmp_path
):
    from gigaevo.memory.prior_evidence import JsonlEvictedEvidence

    missing = JsonlEvictedEvidence(tmp_path / "missing.jsonl")
    assert missing.cards() == ()

    path = tmp_path / "nested" / "prior_evidence.jsonl"
    evidence = JsonlEvictedEvidence(path)
    first = make_card(category="first", gain_events=(make_event(-1.0),))
    evidence.record(first)

    (round_tripped,) = evidence.cards()
    assert (
        round_tripped.id,
        round_tripped.kind,
        round_tripped.category,
        round_tripped.gain_events,
    ) == (first.id, first.kind, first.category, first.gain_events)
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row == {
        "schema_version": "prior_evidence.v1",
        "card": first.model_dump(mode="json"),
    }

    with path.open("a", encoding="utf-8") as f:
        f.write("{malformed\n")
    last = first.model_copy(
        update={"category": "last", "gain_events": (make_event(1.0),)}
    )
    evidence.record(last)

    assert evidence.cards() == (last,)
