"""Replay equivalence on real archived idea banks (task #87 reproduction).

The admitter must reproduce the offline gate-replay study exactly on the
two durably archived banks (db11 house: 103 ideas -> sign-based 28;
db12 california: 85 -> 28). The third replay bank (db15 smoke) was lost
(its /tmp source was emptied; tracker checkpoints are file-based, not in Redis)
— its per-idea diagnostics survive only in _paper/findings/gate_replay_raw.json.

Skipped when the experiment archive is not mounted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gigaevo.memory.core.admitter import SignBasedAdmitter
from gigaevo.memory.core.idea_stats import IdeaStats, coerce_metric

ARCHIVE_ROOT = Path(
    "/mnt/virtual_ai0001071-04017_SR004-nfs1/CFS-SR008/workspace/mathemage/experiment_archive"
)

BANKS = {
    "db11_house": ARCHIVE_ROOT
    / "db11_tabular_house_newmem-baseline/tracker/banks.json",
    "db12_california": ARCHIVE_ROOT
    / "db12_tabular_california_newmem-baseline/tracker/banks.json",
}

EXPECTED = {
    "db11_house": {"ideas": 103, "sign_based": 28},
    "db12_california": {"ideas": 85, "sign_based": 28},
}

pytestmark = pytest.mark.skipif(
    not all(p.is_file() for p in BANKS.values()),
    reason="experiment archive with replay banks not available",
)


def load_bank(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, list) and data and isinstance(data[-1], dict):
        if "active_bank" in data[-1]:
            return data[-1]["active_bank"]
    return data


def stats_from_bank(ideas: list[dict]) -> list[IdeaStats]:
    rows = []
    for idea in ideas:
        stats = idea.get("evolution_statistics") or {}
        for quartile, block in stats.items():
            if not isinstance(block, dict):
                continue
            rec = {
                k: v
                for k, v in block.items()
                if v is not None and k in IdeaStats.model_fields
            }
            rec["idea_id"] = idea.get("id") or idea.get("idea_id")
            rec["quartile"] = quartile
            rec["description"] = idea.get("description", "")
            rows.append(IdeaStats.model_validate(rec))
    return rows


@pytest.fixture(scope="module", params=sorted(BANKS))
def bank_case(request):
    name = request.param
    stats = stats_from_bank(load_bank(BANKS[name]))
    return name, stats


def test_bank_has_expected_idea_count(bank_case):
    name, stats = bank_case
    assert len({s.idea_id for s in stats}) == EXPECTED[name]["ideas"]


def test_sign_based_admitter_matches_replay_counts(bank_case):
    name, stats = bank_case
    got = SignBasedAdmitter().select(stats)
    assert len(got) == EXPECTED[name]["sign_based"]
    assert len({s.idea_id for s in got}) == len(got)
    for s in got:
        median = coerce_metric(s.IntroGain_best_median)
        assert median is not None and median > 0
