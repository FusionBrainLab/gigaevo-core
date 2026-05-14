"""Canonical event for the steady-state two-sema backpressure snapshot.

Why this event exists
---------------------
The engine exposes the two-sema model (``_producer_sema`` / ``_buffer_sema``)
plus the ``_in_flight`` set, all bounded by ``max_in_flight``. From the log
output of a running experiment there is *no way* to tell whether the cap is
actually being utilised — a user sees mutants land, but not the held-slot
count. A periodic structured snapshot answers the question
"is max_in_flight being reached?" directly, and lets the flow profiler render
a concurrency-over-time band against the dashboard.

The event is registered the same way as every other canonical event
(``__init_subclass__`` auto-register in
:class:`gigaevo.monitoring.events.BaseEvent`).
"""

from __future__ import annotations

import json

from pydantic import ValidationError
import pytest

from gigaevo.monitoring.events import CANONICAL_EVENTS, BackpressureSample


class TestBackpressureSampleRegistered:
    def test_event_in_registry(self) -> None:
        assert "BACKPRESSURE_SAMPLE" in CANONICAL_EVENTS
        assert CANONICAL_EVENTS["BACKPRESSURE_SAMPLE"] is BackpressureSample

    def test_event_class_metadata(self) -> None:
        # ClassVars are the contract — log_audit / alerts plumbing reads them.
        assert BackpressureSample.event == "BACKPRESSURE_SAMPLE"
        assert BackpressureSample.description
        assert BackpressureSample.health_question


class TestBackpressureSampleFields:
    def test_construct_with_all_required_fields(self) -> None:
        ev = BackpressureSample(
            producer_held=3,
            buffer_held=2,
            in_flight=2,
            max_in_flight=8,
            llm_active=1,
        )
        assert ev.producer_held == 3
        assert ev.buffer_held == 2
        assert ev.in_flight == 2
        assert ev.max_in_flight == 8
        assert ev.llm_active == 1

    def test_field_types_are_integers(self) -> None:
        # Held counts must be ints — float would make alert thresholds awkward.
        with pytest.raises(ValidationError):
            BackpressureSample(
                producer_held=1.5,  # type: ignore[arg-type]
                buffer_held=0,
                in_flight=0,
                max_in_flight=8,
                llm_active=0,
            )

    def test_negative_held_count_rejected(self) -> None:
        # A negative held count means we miscounted; loud crash > silent rubbish.
        with pytest.raises(ValidationError):
            BackpressureSample(
                producer_held=-1,
                buffer_held=0,
                in_flight=0,
                max_in_flight=8,
                llm_active=0,
            )

    def test_held_exceeding_cap_is_validation_error(self) -> None:
        # Held > cap can only happen if accounting is broken. We want the alert
        # path, not the silent-pass path, so reject at construction.
        with pytest.raises(ValidationError):
            BackpressureSample(
                producer_held=9,
                buffer_held=0,
                in_flight=0,
                max_in_flight=8,
                llm_active=0,
            )

    def test_in_flight_exceeding_cap_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BackpressureSample(
                producer_held=0,
                buffer_held=0,
                in_flight=9,
                max_in_flight=8,
                llm_active=0,
            )

    def test_llm_active_exceeding_producer_held_rejected(self) -> None:
        # llm_active must be <= producer_held since LLM tasks are a subset of producer tasks.
        with pytest.raises(ValidationError):
            BackpressureSample(
                producer_held=2,
                buffer_held=0,
                in_flight=0,
                max_in_flight=8,
                llm_active=3,
            )


class TestBackpressureSampleSerialization:
    def test_json_roundtrip_through_emit_format(self) -> None:
        ev = BackpressureSample(
            producer_held=4,
            buffer_held=5,
            in_flight=5,
            max_in_flight=8,
            llm_active=2,
            run_label="exp_a",
        )
        payload = ev.model_dump(mode="json")
        # Same shape `emit()` writes into the log line, minus the leading
        # "event" key that emit() prepends:
        wire = {"event": BackpressureSample.event, **payload}
        text = json.dumps(wire, ensure_ascii=False)
        decoded = json.loads(text)
        restored = BackpressureSample.model_validate(
            {k: v for k, v in decoded.items() if k != "event"}
        )
        assert restored.producer_held == 4
        assert restored.buffer_held == 5
        assert restored.in_flight == 5
        assert restored.llm_active == 2
        assert restored.run_label == "exp_a"
