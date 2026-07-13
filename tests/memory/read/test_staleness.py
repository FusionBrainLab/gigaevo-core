"""Bank-cycle scalar and per-event weights over cached and uncached banks."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from gigaevo.memory.cards import ContextualGain, DecisionContext
from gigaevo.memory.read.staleness import bank_cycle_event_weights


@pytest.fixture
def make_stamped_event():
    def _make(hours: float, task_key: str = "") -> ContextualGain:
        return ContextualGain(
            context=DecisionContext(
                task_key=task_key,
                parent_metrics={},
                timestamp=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=hours),
            ),
            gain=0.1,
        )

    return _make


def test_weight_counts_strictly_newer_bank_events(make_card, make_stamped_event):
    old_event = make_stamped_event(0)
    newer_events = (make_stamped_event(1), make_stamped_event(2))
    old = make_card(gain_events=(old_event,))
    newer = make_card(gain_events=newer_events)
    bank = (old, newer)
    assert bank_cycle_event_weights((old_event,), bank, 1.0) == pytest.approx(
        (2.0 ** (-2 / 2),)
    )
    assert bank_cycle_event_weights(newer_events, bank, 1.0) == pytest.approx(
        (2.0 ** (-1 / 2), 1.0)
    )


def test_event_weights_age_each_ordered_event_from_its_own_stamp(
    make_card, make_stamped_event
):
    old = make_stamped_event(0, "task-a")
    middle = make_stamped_event(1, "task-a")
    unstamped = old.model_copy(
        update={"context": old.context.model_copy(update={"timestamp": None})}
    )
    foreign_future = make_stamped_event(100, "task-b")
    card = make_card(gain_events=(old, middle, unstamped, foreign_future))
    newer = make_card(gain_events=(make_stamped_event(2, "task-a"),), task_key="task-a")

    weights = bank_cycle_event_weights(
        (middle, old, unstamped), (card, newer), 1.0, task_key="task-a"
    )

    # Task-a population is two cards, so H=2.  Strictly-newer native stamps:
    # middle sees one, old sees two; subset order must be preserved.
    assert weights == pytest.approx((2.0 ** (-1 / 2), 2.0 ** (-2 / 2), 1.0))


def test_foreign_bank_events_cannot_change_native_event_weights(
    make_card, make_stamped_event
):
    native = make_stamped_event(0, "task-a")
    card = make_card(gain_events=(native,), task_key="task-a")
    native_bank = (card,)
    with_foreign = (
        card,
        make_card(
            gain_events=tuple(
                make_stamped_event(hour, "task-b") for hour in range(1, 101)
            ),
            task_key="task-b",
        ),
    )

    assert (
        bank_cycle_event_weights((native,), native_bank, 1.0, task_key="task-a")
        == bank_cycle_event_weights((native,), with_foreign, 1.0, task_key="task-a")
        == (1.0,)
    )


def test_weight_stable_across_repeated_reads_of_same_bank(
    make_card, make_stamped_event
):
    event = make_stamped_event(0)
    old = make_card(gain_events=(event,))
    newer = make_card(gain_events=(make_stamped_event(5),))
    bank = (old, newer)
    first = bank_cycle_event_weights((event,), bank, 1.0)
    assert bank_cycle_event_weights((event,), bank, 1.0) == first
    assert first == pytest.approx((2.0 ** (-1 / 2),))


def test_mutated_list_bank_is_not_served_stale(make_card, make_stamped_event):
    event = make_stamped_event(0)
    old = make_card(gain_events=(event,))
    bank = [old]
    assert bank_cycle_event_weights((event,), bank, 1.0) == (1.0,)
    bank.append(make_card(gain_events=(make_stamped_event(1),)))
    assert bank_cycle_event_weights((event,), bank, 1.0) == pytest.approx(
        (2.0 ** (-1 / 2),)
    )


def test_distinct_equal_banks_do_not_collide(make_card, make_stamped_event):
    event = make_stamped_event(0)
    old = make_card(gain_events=(event,))
    bank_a = (old,)
    bank_b = (old, make_card(gain_events=(make_stamped_event(3),)))
    assert bank_cycle_event_weights((event,), bank_a, 1.0) == (1.0,)
    assert bank_cycle_event_weights((event,), bank_b, 1.0) == pytest.approx(
        (2.0 ** (-1 / 2),)
    )
    assert bank_cycle_event_weights((event,), bank_a, 1.0) == (1.0,)


def test_foreign_events_do_not_age_native_evidence(make_card, make_stamped_event):
    old_a = make_card(gain_events=(make_stamped_event(0, "task-a"),))
    newer_b = make_card(gain_events=(make_stamped_event(10, "task-b"),))
    bank = (old_a, newer_b)

    assert bank_cycle_event_weights(
        old_a.gain_events, bank, 1.0, task_key="task-a"
    ) == (1.0,)


def test_foreign_traffic_does_not_change_native_half_life(
    make_card, make_stamped_event
):
    old_a = make_card(gain_events=(make_stamped_event(0, "task-a"),))
    newer_mixed = make_card(
        gain_events=(
            make_stamped_event(1, "task-a"),
            make_stamped_event(2, "task-a"),
            make_stamped_event(5, "task-b"),
        )
    )
    bank = (old_a, newer_mixed)

    # Population counts cards with native evidence (2), not native events (3):
    # the foreign stamp on newer_mixed must not move old_a's discount.
    assert bank_cycle_event_weights(
        old_a.gain_events, bank, 1.0, task_key="task-a"
    ) == pytest.approx((2.0 ** (-2 / 2),))


def test_eventless_foreign_authored_cards_do_not_inflate_population(
    make_card, make_stamped_event
):
    old_a = make_card(gain_events=(make_stamped_event(0, "task-a"),), task_key="task-a")
    newer_a = make_card(
        gain_events=(make_stamped_event(1, "task-a"),), task_key="task-a"
    )
    eventless_native = make_card(task_key="task-a")
    eventless_foreign = make_card(task_key="task-b")
    bank = (old_a, newer_a, eventless_native, eventless_foreign)

    # Population = 2 evented + 1 eventless task-a card; task-b's eventless
    # card must not slow task-a's decay.
    assert bank_cycle_event_weights(
        old_a.gain_events, bank, 1.0, task_key="task-a"
    ) == pytest.approx((2.0 ** (-1 / 3),))


def test_degenerate_native_population_has_unit_weight(make_card, make_stamped_event):
    card_b = make_card(gain_events=(make_stamped_event(0, "task-b"),))

    assert bank_cycle_event_weights(
        card_b.gain_events, (card_b,), 1.0, task_key="task-a"
    ) == (1.0,)


def test_stamp_cache_is_partitioned_by_task_key(make_card, make_stamped_event):
    old_a = make_card(gain_events=(make_stamped_event(0, "task-a"),))
    fresh_a_old_b = make_card(
        gain_events=(
            make_stamped_event(10, "task-a"),
            make_stamped_event(-10, "task-b"),
        )
    )
    bank = (old_a, fresh_a_old_b)

    weight_a = bank_cycle_event_weights(old_a.gain_events, bank, 1.0, task_key="task-a")
    weight_b = bank_cycle_event_weights(
        (fresh_a_old_b.gain_events[1],), bank, 1.0, task_key="task-b"
    )

    assert weight_a[0] < 1.0
    assert weight_b == (1.0,)
    assert (
        bank_cycle_event_weights(old_a.gain_events, bank, 1.0, task_key="task-a")
        == weight_a
    )
