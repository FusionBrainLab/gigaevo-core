"""Parser + aggregate tests for BACKPRESSURE_SAMPLE in flow_profiler.

The runner log carries one ``[BACKPRESSURE_SAMPLE] {json}`` line per
``loop_interval``. The flow profiler must:

1. Parse each line into a :class:`BackpressureSampleEvent` carrying
   ``(timestamp, producer_held, buffer_held, in_flight, max_in_flight)``.
2. Aggregate the time-series into a :class:`SaturationReport` that
   exposes peak held counts AND a per-side saturation fraction.

The saturation fraction directly answers the operator's question
"is max_in_flight actually being saturated?": near-1.0 means the cap is
the bottleneck (true saturation); well below 1.0 means something
upstream of the cap is the limiter.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from gigaevo.monitoring.flow_profiler import (
    BackpressureSampleEvent,
    SaturationReport,
    compute_saturation,
    parse_log,
)


def _write(tmp_path: Path, body: str) -> Path:
    log = tmp_path / "run.log"
    log.write_text(body)
    return log


BACKPRESSURE_LOG = (
    '2026-05-13 00:00:00.000 INFO [BACKPRESSURE_SAMPLE] {"event": '
    '"BACKPRESSURE_SAMPLE", "producer_held": 8, "buffer_held": 4, '
    '"in_flight": 4, "max_in_flight": 8, "llm_active": 4, "run_label": null}\n'
    '2026-05-13 00:00:01.000 INFO [BACKPRESSURE_SAMPLE] {"event": '
    '"BACKPRESSURE_SAMPLE", "producer_held": 8, "buffer_held": 8, '
    '"in_flight": 8, "max_in_flight": 8, "llm_active": 3, "run_label": null}\n'
    '2026-05-13 00:00:02.000 INFO [BACKPRESSURE_SAMPLE] {"event": '
    '"BACKPRESSURE_SAMPLE", "producer_held": 3, "buffer_held": 2, '
    '"in_flight": 2, "max_in_flight": 8, "llm_active": 2, "run_label": null}\n'
    '2026-05-13 00:00:03.000 INFO [BACKPRESSURE_SAMPLE] {"event": '
    '"BACKPRESSURE_SAMPLE", "producer_held": 8, "buffer_held": 0, '
    '"in_flight": 0, "max_in_flight": 8, "llm_active": 1, "run_label": null}\n'
)


class TestParseBackpressureSamples:
    def test_parse_emits_one_sample_per_line(self, tmp_path: Path) -> None:
        log = _write(tmp_path, BACKPRESSURE_LOG)
        _, _, _, samples = parse_log(log)
        assert len(samples) == 4
        first = samples[0]
        assert isinstance(first, BackpressureSampleEvent)
        assert first.producer_held == 8
        assert first.buffer_held == 4
        assert first.in_flight == 4
        assert first.max_in_flight == 8
        assert first.llm_active == 4
        assert first.timestamp == datetime(2026, 5, 13, 0, 0, 0)

    def test_parse_log_returns_empty_samples_when_no_event_present(
        self, tmp_path: Path
    ) -> None:
        log = _write(
            tmp_path,
            "2026-05-13 00:00:00.000 INFO [SteadyState] Start | producer_sema=8\n",
        )
        _, _, _, samples = parse_log(log)
        assert samples == []

    def test_parse_log_skips_malformed_json(self, tmp_path: Path) -> None:
        body = (
            "2026-05-13 00:00:00.000 INFO [BACKPRESSURE_SAMPLE] {bad json}\n"
            '2026-05-13 00:00:01.000 INFO [BACKPRESSURE_SAMPLE] {"event": '
            '"BACKPRESSURE_SAMPLE", "producer_held": 5, "buffer_held": 1, '
            '"in_flight": 1, "max_in_flight": 8, "run_label": null}\n'
        )
        log = _write(tmp_path, body)
        _, _, _, samples = parse_log(log)
        # The malformed line is silently skipped, but the good one parses.
        assert len(samples) == 1
        assert samples[0].producer_held == 5


class TestComputeSaturation:
    def test_saturation_pct_counts_samples_at_cap(self) -> None:
        samples = [
            BackpressureSampleEvent(
                timestamp=datetime(2026, 1, 1, 0, 0, i),
                producer_held=ph,
                buffer_held=bh,
                in_flight=bh,
                max_in_flight=8,
                llm_active=llm,
            )
            for i, (ph, bh, llm) in enumerate(
                [(8, 4, 5), (8, 8, 6), (3, 2, 2), (8, 0, 4)],
            )
        ]
        report = compute_saturation(samples)
        assert isinstance(report, SaturationReport)
        # 3 of 4 samples have producer_held == 8 → 75% producer saturation.
        assert report.producer_saturation_pct == pytest.approx(75.0)
        # 1 of 4 samples has buffer_held == 8 → 25% buffer saturation.
        assert report.buffer_saturation_pct == pytest.approx(25.0)
        assert report.peak_producer_held == 8
        assert report.peak_buffer_held == 8
        assert report.peak_llm_active == 6
        assert report.max_in_flight == 8
        assert report.sample_count == 4

    def test_empty_samples_yields_zero_saturation(self) -> None:
        report = compute_saturation([])
        assert report.sample_count == 0
        assert report.producer_saturation_pct == 0.0
        assert report.buffer_saturation_pct == 0.0
        assert report.peak_producer_held == 0
        assert report.peak_buffer_held == 0
        # Without samples we can't know the cap — report as 0.
        assert report.max_in_flight == 0

    def test_mixed_max_in_flight_uses_last_observed(self) -> None:
        # If max_in_flight ever changes mid-run (it shouldn't, but if e.g.
        # someone restarts with a different config), we report the LAST
        # observed value so the displayed saturation matches the steady-
        # state cap the operator currently cares about.
        s1 = BackpressureSampleEvent(
            timestamp=datetime(2026, 1, 1, 0, 0, 0),
            producer_held=4,
            buffer_held=4,
            in_flight=4,
            max_in_flight=4,
            llm_active=2,
        )
        s2 = BackpressureSampleEvent(
            timestamp=datetime(2026, 1, 1, 0, 0, 1),
            producer_held=8,
            buffer_held=8,
            in_flight=8,
            max_in_flight=8,
            llm_active=5,
        )
        report = compute_saturation([s1, s2])
        assert report.max_in_flight == 8
        # Saturation: producer_held==max_in_flight at the sample's own cap.
        # s1: 4 == 4 ✓, s2: 8 == 8 ✓ → 100%.
        assert report.producer_saturation_pct == pytest.approx(100.0)
        # Peak LLM-active is max of the samples.
        assert report.peak_llm_active == 5

    def test_llm_active_breakdown_in_producer_held(self) -> None:
        # LLM-active is a subset of producer_held. Verify that
        # peak_llm_active and peak_dag_active sum correctly.
        samples = [
            BackpressureSampleEvent(
                timestamp=datetime(2026, 1, 1, 0, 0, i),
                producer_held=ph,
                buffer_held=bh,
                in_flight=bh,
                max_in_flight=8,
                llm_active=llm,
            )
            for i, (ph, bh, llm) in enumerate(
                [(8, 4, 6), (8, 8, 3), (3, 2, 1), (8, 0, 8)],
            )
        ]
        report = compute_saturation(samples)
        # Peak LLM-active: max(6, 3, 1, 8) = 8.
        assert report.peak_llm_active == 8
        # DAG-active = producer_held - llm_active per sample; peaks at max(2, 5, 2, 0) = 5.
        assert report.peak_dag_active == 5
