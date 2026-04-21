"""Tests for the registry-backed log audit.

The audit parses a log file, tries to instantiate each canonical event via
`CANONICAL_EVENTS[name](**payload)`, and reports:
- Parse errors (non-JSON payload or unknown event name).
- Pydantic validation errors (missing field, bad type, invariant violation).
- Missing-by-gen errors when an event's `expected_after_gen` > 0 but no such
  event appears and the latest observed `gen` already exceeded that threshold.
"""

from __future__ import annotations

from gigaevo.experiment.log_audit import AuditReport, audit_log_text


def _make_log(lines: list[str]) -> str:
    return "\n".join(lines) + "\n"


class TestParseAndValidate:
    def test_well_formed_event_passes(self) -> None:
        log = _make_log(
            [
                '[TRACKER_WRITE] {"event": "TRACKER_WRITE", "pairs_count": 5, '
                '"positive_count": 2, "d_wins_added": 2, "g_resisted_added": 3, '
                '"d_faced_added": 5, "gen": 1}',
            ]
        )
        report = audit_log_text(log)
        assert report.failures == {}
        assert report.event_counts["TRACKER_WRITE"] == 1

    def test_missing_field_fails_with_pydantic_error(self) -> None:
        log = _make_log(
            [
                # missing pairs_count
                '[TRACKER_WRITE] {"event": "TRACKER_WRITE", "positive_count": 2, '
                '"d_wins_added": 2, "g_resisted_added": 3, "d_faced_added": 5}',
            ]
        )
        report = audit_log_text(log)
        assert "TRACKER_WRITE" in report.failures
        assert any("pairs_count" in msg for msg in report.failures["TRACKER_WRITE"])

    def test_positive_gt_pairs_invariant_fires(self) -> None:
        log = _make_log(
            [
                '[TRACKER_WRITE] {"event": "TRACKER_WRITE", "pairs_count": 1, '
                '"positive_count": 99, "d_wins_added": 99, "g_resisted_added": 0, '
                '"d_faced_added": 1}',
            ]
        )
        report = audit_log_text(log)
        assert "TRACKER_WRITE" in report.failures

    def test_unknown_event_name_flagged(self) -> None:
        log = _make_log(['[TOTALLY_MADE_UP] {"x": 1}'])
        report = audit_log_text(log)
        assert "TOTALLY_MADE_UP" in report.failures
        assert any("unknown" in m.lower() for m in report.failures["TOTALLY_MADE_UP"])

    def test_non_json_payload_flagged(self) -> None:
        log = _make_log(["[TRACKER_WRITE] not-json-here"])
        report = audit_log_text(log)
        # Parse error reported under event type from bracket — falls under
        # a generic "parse" key since we never got JSON.
        assert "_parse" in report.failures or "TRACKER_WRITE" in report.failures

    def test_line_not_matching_pattern_is_ignored(self) -> None:
        log = _make_log(
            [
                "2026-04-18 12:00:00.123 | INFO | startup banner",
                "some unrelated stderr noise",
                '[TRACKER_WRITE] {"event": "TRACKER_WRITE", "pairs_count": 0, '
                '"positive_count": 0, "d_wins_added": 0, "g_resisted_added": 0, '
                '"d_faced_added": 0}',
            ]
        )
        report = audit_log_text(log)
        assert report.failures == {}
        assert report.event_counts["TRACKER_WRITE"] == 1


class TestMissingByExpectedGen:
    def test_event_missing_past_expected_gen_is_flagged(self) -> None:
        # HOF_ROTATE has expected_after_gen=2 — if we observed gen=5 events
        # but never saw a HOF_ROTATE, the audit should complain.
        log = _make_log(
            [
                '[GENERATION_BOUNDARY] {"event": "GENERATION_BOUNDARY", "gen": 5}',
            ]
        )
        report = audit_log_text(log)
        assert "HOF_ROTATE" in report.missing_after_gen
        # HOF_FETCH has expected_after_gen=1 so it should also appear missing
        assert "HOF_FETCH" in report.missing_after_gen

    def test_event_present_is_not_missing(self) -> None:
        log = _make_log(
            [
                '[GENERATION_BOUNDARY] {"event": "GENERATION_BOUNDARY", "gen": 3}',
                '[HOF_ROTATE] {"event": "HOF_ROTATE", "label": "X", '
                '"old_hof_size": 1, "new_hof_size": 2, "gen": 2}',
            ]
        )
        report = audit_log_text(log)
        assert "HOF_ROTATE" not in report.missing_after_gen

    def test_no_generation_boundary_no_missing_check(self) -> None:
        # If we never observed a gen, we can't say anything is missing-by-gen.
        log = _make_log(
            [
                '[TRACKER_WRITE] {"event": "TRACKER_WRITE", "pairs_count": 0, '
                '"positive_count": 0, "d_wins_added": 0, "g_resisted_added": 0, '
                '"d_faced_added": 0}',
            ]
        )
        report = audit_log_text(log)
        assert report.missing_after_gen == {}


class TestReportFormat:
    def test_audit_report_is_hashable_dataclass(self) -> None:
        # Sanity: we return a typed object, not a magic dict.
        log = _make_log([])
        report = audit_log_text(log)
        assert isinstance(report, AuditReport)
        assert isinstance(report.failures, dict)
        assert isinstance(report.event_counts, dict)
        assert isinstance(report.missing_after_gen, dict)
