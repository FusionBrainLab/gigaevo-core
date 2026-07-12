"""bank_cycle_weight: correctness across cached (tuple) and uncached banks."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from gigaevo.memory.cards import ContextualGain, DecisionContext
from gigaevo.memory.read.staleness import bank_cycle_weight


@pytest.fixture
def make_stamped_event():
    def _make(hours: float) -> ContextualGain:
        return ContextualGain(
            context=DecisionContext(
                parent_metrics={},
                timestamp=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=hours),
            ),
            gain=0.1,
        )

    return _make


def test_weight_counts_strictly_newer_bank_events(make_card, make_stamped_event):
    old = make_card(gain_events=(make_stamped_event(0),))
    newer = make_card(gain_events=(make_stamped_event(1), make_stamped_event(2)))
    bank = (old, newer)
    assert bank_cycle_weight(old, bank, 1.0) == pytest.approx(2.0 ** (-2 / 2))
    assert bank_cycle_weight(newer, bank, 1.0) == 1.0


def test_weight_stable_across_repeated_reads_of_same_bank(
    make_card, make_stamped_event
):
    old = make_card(gain_events=(make_stamped_event(0),))
    newer = make_card(gain_events=(make_stamped_event(5),))
    bank = (old, newer)
    first = bank_cycle_weight(old, bank, 1.0)
    assert bank_cycle_weight(old, bank, 1.0) == first
    assert first == pytest.approx(2.0 ** (-1 / 2))


def test_mutated_list_bank_is_not_served_stale(make_card, make_stamped_event):
    old = make_card(gain_events=(make_stamped_event(0),))
    bank = [old]
    assert bank_cycle_weight(old, bank, 1.0) == 1.0
    bank.append(make_card(gain_events=(make_stamped_event(1),)))
    assert bank_cycle_weight(old, bank, 1.0) == pytest.approx(2.0 ** (-1 / 2))


def test_distinct_equal_banks_do_not_collide(make_card, make_stamped_event):
    old = make_card(gain_events=(make_stamped_event(0),))
    bank_a = (old,)
    bank_b = (old, make_card(gain_events=(make_stamped_event(3),)))
    assert bank_cycle_weight(old, bank_a, 1.0) == 1.0
    assert bank_cycle_weight(old, bank_b, 1.0) == pytest.approx(2.0 ** (-1 / 2))
    assert bank_cycle_weight(old, bank_a, 1.0) == 1.0
