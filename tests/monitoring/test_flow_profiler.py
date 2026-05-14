"""Unit tests for ``gigaevo.monitoring.flow_profiler``.

The CLI-level smoke tests live in ``tests/cli/test_profiler_cmd.py``.
This file targets the pure parsing and analysis primitives:

- ``parse_log`` now also extracts ``LLM_CALL`` canonical events and the
  ``archetype=`` / ``model=`` suffix on the ``[mutation] Task N: ...`` INFO
  line so the profiler can answer "are LLM stages utilizing LLM efficiently".
- ``compute_utilization`` aggregates intervals into LLM-vs-exec overlap stats
  (the headline efficiency number).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from gigaevo.monitoring.flow_profiler import (
    LLMCallEvent,
    UtilizationReport,
    classify_stage,
    compute_utilization,
    parse_log,
)

# --------------------------------------------------------------------------- #
# Synthetic log fixtures                                                      #
# --------------------------------------------------------------------------- #


def _write(tmp_path: Path, body: str) -> Path:
    log = tmp_path / "run.log"
    log.write_text(body)
    return log


LLM_CALL_LOG = (
    '2026-05-13 00:00:00.000 INFO [LLM_CALL] {"event": "LLM_CALL", "stage": '
    '"LineageAgent", "program_id": "aaaaaaaa-1111-2222-3333-444444444444", '
    '"endpoint": "", "model": "gpt-4", "attempt": 1, "ok": true, '
    '"latency_ms": 5000.0, "tokens_in": 100, "tokens_out": 50, '
    '"error_type": null, "run_label": null}\n'
    '2026-05-13 00:00:05.000 INFO [LLM_CALL] {"event": "LLM_CALL", "stage": '
    '"MutationAgent", "program_id": null, "endpoint": "", "model": "gpt-4", '
    '"attempt": 1, "ok": true, "latency_ms": 2500.0, "tokens_in": 50, '
    '"tokens_out": 20, "error_type": null, "run_label": null}\n'
    '2026-05-13 00:00:10.000 INFO [LLM_CALL] {"event": "LLM_CALL", "stage": '
    '"InsightsAgent", "program_id": "bbbbbbbb-1111-2222-3333-444444444444", '
    '"endpoint": "", "model": "gpt-4", "attempt": 2, "ok": false, '
    '"latency_ms": 1234.5, "tokens_in": 0, "tokens_out": 0, '
    '"error_type": "TimeoutError", "run_label": null}\n'
)

MUT_LINE_WITH_META = (
    "2026-05-13 00:00:00.000 INFO [mutation] Task 1: "
    "['aaaaaaaa'] → bbbbbbbb (model=gpt-4, archetype=Precision Optimization, "
    "prompt_id=default)\n"
)

MUT_LINE_LEGACY = (
    "2026-05-13 00:00:00.000 INFO [mutation] Task 1: ['aaaaaaaa'] -> bbbbbbbb\n"
)


# --------------------------------------------------------------------------- #
# parse_log: LLM_CALL extraction                                              #
# --------------------------------------------------------------------------- #


class TestParseLogLLMCallExtraction:
    def test_returns_four_tuple_with_backpressure_samples(self, tmp_path: Path):
        log = _write(tmp_path, LLM_CALL_LOG)
        result = parse_log(log)
        assert len(result) == 4, (
            "parse_log must return "
            "(programs, refreshes, llm_events, backpressure_samples)"
        )

    def test_collects_each_llm_call_event(self, tmp_path: Path):
        log = _write(tmp_path, LLM_CALL_LOG)
        _, _, llm_events, _ = parse_log(log)
        assert len(llm_events) == 3
        stages = [e.stage for e in llm_events]
        assert stages == ["LineageAgent", "MutationAgent", "InsightsAgent"]

    def test_preserves_duration_and_outcome(self, tmp_path: Path):
        log = _write(tmp_path, LLM_CALL_LOG)
        _, _, llm_events, _ = parse_log(log)
        # First Lineage call: 5000ms, ok
        assert llm_events[0].duration_ms == 5000.0
        assert llm_events[0].ok is True
        assert llm_events[0].error_type is None
        # Failed Insights call: 1234.5ms, not ok
        assert llm_events[2].ok is False
        assert llm_events[2].error_type == "TimeoutError"
        assert llm_events[2].duration_ms == 1234.5

    def test_end_timestamp_matches_log_line(self, tmp_path: Path):
        log = _write(tmp_path, LLM_CALL_LOG)
        _, _, llm_events, _ = parse_log(log)
        assert llm_events[0].end == datetime(2026, 5, 13, 0, 0, 0, 0)
        assert llm_events[1].end == datetime(2026, 5, 13, 0, 0, 5, 0)

    def test_short_program_id_extracted_when_present(self, tmp_path: Path):
        log = _write(tmp_path, LLM_CALL_LOG)
        _, _, llm_events, _ = parse_log(log)
        assert llm_events[0].program_id == "aaaaaaaa"
        assert llm_events[1].program_id is None  # mutation event has null
        assert llm_events[2].program_id == "bbbbbbbb"


# --------------------------------------------------------------------------- #
# parse_log: mutation line metadata extraction                                #
# --------------------------------------------------------------------------- #


class TestParseLogMutationMetadata:
    def test_extracts_archetype_and_model_from_mutation_line(self, tmp_path: Path):
        log = _write(tmp_path, MUT_LINE_WITH_META)
        programs, _, _, _ = parse_log(log)
        child = programs["bbbbbbbb"]
        assert child.mutation_archetype == "Precision Optimization"
        assert child.mutation_model == "gpt-4"

    def test_legacy_mutation_line_still_parses_without_metadata(self, tmp_path: Path):
        log = _write(tmp_path, MUT_LINE_LEGACY)
        programs, _, _, _ = parse_log(log)
        child = programs["bbbbbbbb"]
        assert child.mutation_archetype is None
        assert child.mutation_model is None
        # core fields still extracted
        assert child.parents == ("aaaaaaaa",)


# --------------------------------------------------------------------------- #
# classify_stage                                                              #
# --------------------------------------------------------------------------- #


class TestClassifyStage:
    def test_known_llm_stages_classified_as_llm(self):
        assert classify_stage("LineageStage") == "llm"
        assert classify_stage("InsightsStage") == "llm"
        # MutationAgent is the canonical LLM_CALL stage name for mutation;
        # it should also be classified as LLM when seen on an LLMCallEvent.
        assert classify_stage("LineageAgent") == "llm"
        assert classify_stage("InsightsAgent") == "llm"
        assert classify_stage("MutationAgent") == "llm"

    def test_program_execution_stages_classified_as_exec(self):
        assert classify_stage("CallProgramFunction") == "exec"
        assert classify_stage("CallValidatorFunction") == "exec"

    def test_other_stages_classified_as_orchestration(self):
        assert classify_stage("AncestorProgramIds") == "orchestration"
        assert classify_stage("DGTrackerStage") == "orchestration"
        assert classify_stage("MergeDictStage") == "orchestration"
        assert classify_stage("UnknownNewStage") == "orchestration"


# --------------------------------------------------------------------------- #
# compute_utilization: overlap math                                           #
# --------------------------------------------------------------------------- #


def _ts(offset_s: float) -> datetime:
    return datetime(2026, 5, 13) + timedelta(seconds=offset_s)


def _llm_evt(stage: str, end_offset_s: float, dur_ms: float) -> LLMCallEvent:
    return LLMCallEvent(
        stage=stage,
        program_id=None,
        end=_ts(end_offset_s),
        duration_ms=dur_ms,
        ok=True,
        model="gpt-4",
        error_type=None,
    )


class TestComputeUtilization:
    def test_empty_input_returns_zero_report(self):
        rep = compute_utilization({}, [], [])
        assert isinstance(rep, UtilizationReport)
        assert rep.overlap_s == 0.0
        assert rep.total_llm_s == 0.0
        assert rep.total_exec_s == 0.0
        assert rep.overlap_efficiency == 0.0

    def test_disjoint_llm_and_exec_have_zero_overlap(self):
        # LLM [0, 5s] then exec [10, 15s] — no overlap.
        llm = [_llm_evt("LineageAgent", end_offset_s=5.0, dur_ms=5000.0)]
        # Synthesize exec via a Program with a CallProgramFunction stage run.
        from gigaevo.monitoring.flow_profiler import Program, StageRun

        p = Program(short_id="aaaaaaaa")
        p.stage_runs.append(
            StageRun(
                stage="CallProgramFunction",
                start=_ts(10.0),
                end=_ts(15.0),
                decision="no_cache",
            )
        )
        rep = compute_utilization({"aaaaaaaa": p}, [], llm)
        assert rep.total_llm_s == 5.0
        assert rep.total_exec_s == 5.0
        assert rep.overlap_s == 0.0
        assert rep.overlap_efficiency == 0.0

    def test_full_overlap_yields_efficiency_one(self):
        # LLM [0, 5s] AND exec [0, 5s] — fully overlapped.
        llm = [_llm_evt("LineageAgent", end_offset_s=5.0, dur_ms=5000.0)]
        from gigaevo.monitoring.flow_profiler import Program, StageRun

        p = Program(short_id="aaaaaaaa")
        p.stage_runs.append(
            StageRun(
                stage="CallProgramFunction",
                start=_ts(0.0),
                end=_ts(5.0),
                decision="no_cache",
            )
        )
        rep = compute_utilization({"aaaaaaaa": p}, [], llm)
        assert rep.overlap_s == 5.0
        assert rep.overlap_efficiency == 1.0

    def test_partial_overlap_correct_slice(self):
        # LLM [0, 6s], exec [4, 10s] — overlap is [4, 6] = 2s.
        llm = [_llm_evt("LineageAgent", end_offset_s=6.0, dur_ms=6000.0)]
        from gigaevo.monitoring.flow_profiler import Program, StageRun

        p = Program(short_id="aaaaaaaa")
        p.stage_runs.append(
            StageRun(
                stage="CallProgramFunction",
                start=_ts(4.0),
                end=_ts(10.0),
                decision="no_cache",
            )
        )
        rep = compute_utilization({"aaaaaaaa": p}, [], llm)
        assert rep.overlap_s == 2.0
        # efficiency = overlap / min(llm, exec) = 2 / min(6, 6) = 1/3
        assert abs(rep.overlap_efficiency - (2.0 / 6.0)) < 1e-9

    def test_cached_skip_intervals_ignored_for_exec(self):
        # cached_skip stage runs are sub-millisecond; they should NOT
        # inflate exec time.
        from gigaevo.monitoring.flow_profiler import Program, StageRun

        p = Program(short_id="aaaaaaaa")
        p.stage_runs.append(
            StageRun(
                stage="CallProgramFunction",
                start=_ts(0.0),
                end=_ts(0.0),
                decision="cached_skip",
            )
        )
        rep = compute_utilization({"aaaaaaaa": p}, [], [])
        assert rep.total_exec_s == 0.0

    def test_orchestration_stages_excluded_from_both_sides(self):
        from gigaevo.monitoring.flow_profiler import Program, StageRun

        p = Program(short_id="aaaaaaaa")
        p.stage_runs.append(
            StageRun(
                stage="AncestorProgramIds",
                start=_ts(0.0),
                end=_ts(2.0),
                decision="no_cache",
            )
        )
        rep = compute_utilization({"aaaaaaaa": p}, [], [])
        assert rep.total_llm_s == 0.0
        assert rep.total_exec_s == 0.0

    def test_concurrent_llm_calls_unioned_not_doubled(self):
        # Two LLM calls both [0, 5s] — union is [0, 5s], not 10s.
        llm = [
            _llm_evt("LineageAgent", end_offset_s=5.0, dur_ms=5000.0),
            _llm_evt("InsightsAgent", end_offset_s=5.0, dur_ms=5000.0),
        ]
        rep = compute_utilization({}, [], llm)
        # total_llm_s is interval UNION, not sum
        assert rep.total_llm_s == 5.0


# --------------------------------------------------------------------------- #
# UtilizationReport: derived stats                                            #
# --------------------------------------------------------------------------- #


class TestUtilizationReportDerivedStats:
    def test_peak_concurrent_dags_inferred_from_intervals(self):
        # Two overlapping DAG runs → peak == 2.
        from gigaevo.monitoring.flow_profiler import Program

        p1 = Program(short_id="aaaaaaaa")
        p1.dag_starts.append(_ts(0.0))
        p1.dag_dones.append(_ts(10.0))
        p2 = Program(short_id="bbbbbbbb")
        p2.dag_starts.append(_ts(5.0))
        p2.dag_dones.append(_ts(15.0))
        rep = compute_utilization({"aaaaaaaa": p1, "bbbbbbbb": p2}, [], [])
        assert rep.peak_concurrent_dags == 2

    def test_mutation_archetype_counts_by_outcome(self):
        from gigaevo.monitoring.flow_profiler import Program

        p1 = Program(short_id="aaaaaaaa")
        p1.mutation_archetype = "Precision Optimization"
        p1.accepted = _ts(0.0)
        p2 = Program(short_id="bbbbbbbb")
        p2.mutation_archetype = "Precision Optimization"
        p2.rejected = _ts(0.0)
        p3 = Program(short_id="cccccccc")
        p3.mutation_archetype = "Algorithmic Redesign"
        p3.accepted = _ts(0.0)
        rep = compute_utilization(
            {"aaaaaaaa": p1, "bbbbbbbb": p2, "cccccccc": p3}, [], []
        )
        assert rep.archetype_counts["Precision Optimization"]["accepted"] == 1
        assert rep.archetype_counts["Precision Optimization"]["rejected"] == 1
        assert rep.archetype_counts["Algorithmic Redesign"]["accepted"] == 1
