"""Structural/integrity checks for the memory-card health monitor.

Covers the per-card adequacy snapshot and the cross-card integrity invariants
of the absorbed-id alias layer (an absorbed id must reference a *removed* card,
be claimed by exactly one survivor, and never name its own survivor).
"""

import json

from tools.memory_card_health import (
    HealthFlag,
    assess_card,
    assess_run,
    load_card_bank,
)


def _mem(card_id="mem-1", **over):
    card = {
        "id": card_id,
        "description": "Bin gain-to-bid ratio before clipping the upper tail.",
        "keywords": ["binning", "clip"],
        "programs": ["p1", "p2"],
        "gain_events": None,
        "absorbed_ids": [],
    }
    card.update(over)
    return card


def _bank(*cards):
    return {c["id"]: c for c in cards}


def test_assess_card_counts_attributes():
    h = assess_card("mem-1", _mem(gain_events=[{"gain": 0.1}], absorbed_ids=["mem-9"]))
    assert h.card_type == "mem"
    assert h.n_keywords == 2
    assert h.n_programs == 2
    assert h.n_gain_events == 1
    assert h.absorbed_ids == ("mem-9",)
    assert h.missing_description is False


def test_assess_card_flags_missing_description():
    assert assess_card("mem-1", _mem(description="")).missing_description is True
    assert assess_card("mem-1", _mem(description=None)).missing_description is True


def test_assess_card_handles_absent_optional_fields():
    h = assess_card("mem-1", {"id": "mem-1", "description": "x"})
    assert h.n_keywords == 0
    assert h.n_programs == 0
    assert h.n_gain_events == 0
    assert h.absorbed_ids == ()


def test_assess_run_rolls_up_counts():
    bank = _bank(
        _mem("mem-1", keywords=[], gain_events=[{"gain": 0.1}]),
        _mem("mem-2", absorbed_ids=["mem-3"]),
        {"id": "prog-1", "description": "a program card"},
    )
    r = assess_run("S1", bank)
    assert r.n_cards == 3
    assert r.n_mem == 2
    assert r.n_program == 1
    assert r.n_zero_keywords == 1
    assert r.n_with_gain_events == 1
    assert r.n_with_absorbed == 1


def test_assess_run_flags_absorbed_id_still_live():
    # An absorbed id that is ALSO a live card means the survivor and the folded
    # card both exist -> the fold would double-count that id's events.
    bank = _bank(_mem("mem-1", absorbed_ids=["mem-2"]), _mem("mem-2"))
    flags = assess_run("S1", bank).flags
    assert HealthFlag("absorbed_id_still_live", "mem-1", "mem-2") in flags


def test_assess_run_flags_cross_absorbed():
    # The same absorbed id claimed by two survivors -> its events fold onto both.
    bank = _bank(
        _mem("mem-1", absorbed_ids=["mem-9"]),
        _mem("mem-2", absorbed_ids=["mem-9"]),
    )
    flags = assess_run("S1", bank).flags
    kinds = {(f.kind, f.detail) for f in flags}
    assert ("cross_absorbed", "mem-9") in kinds


def test_assess_run_flags_self_absorbed():
    bank = _bank(_mem("mem-1", absorbed_ids=["mem-1"]))
    flags = assess_run("S1", bank).flags
    assert HealthFlag("self_absorbed", "mem-1", "mem-1") in flags


def test_assess_run_flags_duplicate_descriptions():
    bank = _bank(
        _mem("mem-1", description="identical text"),
        _mem("mem-2", description="identical text"),
    )
    flags = assess_run("S1", bank).flags
    assert any(f.kind == "duplicate_description" for f in flags)


def test_assess_run_clean_bank_has_no_flags():
    bank = _bank(_mem("mem-1"), _mem("mem-2", description="a different idea entirely"))
    assert assess_run("S1", bank).flags == ()


def test_load_card_bank_missing_file_is_empty(tmp_path):
    assert load_card_bank(tmp_path) == {}


def test_load_card_bank_reads_memory_cards(tmp_path):
    mem = tmp_path / "memory"
    mem.mkdir()
    payload = {"memory_cards": {"mem-1": _mem("mem-1")}}
    (mem / "api_index.json").write_text(json.dumps(payload))
    bank = load_card_bank(tmp_path)
    assert set(bank) == {"mem-1"}
    assert bank["mem-1"]["description"].startswith("Bin gain-to-bid")
