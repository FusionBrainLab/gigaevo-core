"""Bank-cycle staleness: the single evidence-ageing mechanism.

For task ``T``, staleness ``s_i`` of event ``i`` is the number of native gain
events strictly newer than that event's own stamp — an internal rank count, no
wall clock. The half-life is ``half_life_cycles`` task-population cycles, so the
discount ``w_i = 2**(-s_i / H)`` is self-normalizing and foreign traffic cannot
age native evidence.

The posterior, bootstrap EV, probe support, and eviction support all consume
the same per-event vector, so a fresh event cannot revive older history.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Sequence
from datetime import UTC, datetime
import threading

from gigaevo.memory.cards import Card, ContextualGain
from gigaevo.memory.context.evidence import split_events_by_task


def stamp(value: datetime | None) -> datetime | None:
    """UTC-normalize a possibly-naive event timestamp (bank stamps are UTC)."""
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def bank_cycle_event_weights(
    events: Sequence[ContextualGain],
    bank: Sequence[Card],
    half_life_cycles: float,
    *,
    task_key: str = "",
) -> tuple[float, ...]:
    """Per-event discounts ``w_i = 2**(-s_i / H)`` in input order.

    ``s_i`` counts native bank stamps strictly newer than event ``i``'s own
    stamp and ``H`` is the task population times ``half_life_cycles``. An
    unstamped event, or an empty/degenerate task population, has unit weight.
    """
    if not events:
        return ()
    stamps, population = _task_bank_stamps(bank, task_key)
    half_life = population * half_life_cycles
    if half_life <= 0:
        return (1.0,) * len(events)
    weights: list[float] = []
    for event in events:
        event_stamp = stamp(event.context.timestamp)
        if event_stamp is None:
            weights.append(1.0)
            continue
        staleness = len(stamps) - bisect_right(stamps, event_stamp)
        weights.append(float(2.0 ** (-staleness / half_life)))
    return tuple(weights)


_STAMP_CACHE: OrderedDict[
    tuple[int, str], tuple[Sequence[Card], tuple[list[datetime], int]]
] = OrderedDict()
_STAMP_CACHE_MAX = 4
_STAMP_CACHE_LOCK = threading.Lock()


def _task_bank_stamps(
    bank: Sequence[Card], task_key: str
) -> tuple[list[datetime], int]:
    # Only immutable snapshots are safe to key by identity; the strong ref in
    # the cache entry keeps the id from being reused by a successor tuple.
    cacheable = isinstance(bank, tuple)
    cache_key = (id(bank), task_key)
    if cacheable:
        with _STAMP_CACHE_LOCK:
            entry = _STAMP_CACHE.get(cache_key)
            if entry is not None and entry[0] is bank:
                return entry[1]
    partitions = [split_events_by_task(card.gain_events, task_key) for card in bank]
    stamps = sorted(
        stamped
        for native, _ in partitions
        for event in native
        if (stamped := stamp(event.context.timestamp)) is not None
    )
    # Card-based population: cards with native evidence plus never-evented
    # cards authored by this task. Foreign traffic — events or card-authoring
    # floods — cannot shift the half-life, and a single-task bank keeps the
    # historical len(bank) denominator exactly.
    population = sum(
        1
        for card, (native, foreign) in zip(bank, partitions, strict=True)
        if native or (not foreign and card.task_key == task_key)
    )
    result = (stamps, population)
    if cacheable:
        with _STAMP_CACHE_LOCK:
            _STAMP_CACHE[cache_key] = (bank, result)
            while len(_STAMP_CACHE) > _STAMP_CACHE_MAX:
                _STAMP_CACHE.popitem(last=False)
    return result
