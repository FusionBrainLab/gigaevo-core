"""Schema tests for ProgramInsight enum fields and card attribution."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from gigaevo.llm.agents.insights import ProgramInsight


def _insight(**overrides) -> ProgramInsight:
    payload = {"type": "threshold_tuning", "tag": "rigid", "severity": "medium"}
    payload.update(overrides)
    return ProgramInsight(**payload)


class TestTagSeverityEnums:
    @pytest.mark.parametrize(
        "tag", ["beneficial", "harmful", "fragile", "rigid", "neutral"]
    )
    def test_valid_tags_accepted(self, tag: str) -> None:
        assert _insight(tag=tag).tag == tag

    @pytest.mark.parametrize("tag", ["optimization", "cache", "high", ""])
    def test_invalid_tags_rejected(self, tag: str) -> None:
        with pytest.raises(ValidationError):
            _insight(tag=tag)

    @pytest.mark.parametrize("severity", ["high", "medium", "low"])
    def test_valid_severities_accepted(self, severity: str) -> None:
        assert _insight(severity=severity).severity == severity

    @pytest.mark.parametrize("severity", ["urgent", "beneficial", ""])
    def test_invalid_severities_rejected(self, severity: str) -> None:
        with pytest.raises(ValidationError):
            _insight(severity=severity)


class TestMechanismSource:
    def test_defaults_empty(self) -> None:
        ins = _insight()
        assert ins.mechanism_source == ""
        assert ins.card_id == ""

    @pytest.mark.parametrize(
        "source",
        [
            "",
            "program",
            "metrics",
            "intra_memory",
            "memory_cards",
            "ancestral_trail",
            "evolutionary_statistics",
        ],
    )
    def test_valid_sources_accepted(self, source: str) -> None:
        assert _insight(mechanism_source=source).mechanism_source == source

    def test_invalid_source_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _insight(mechanism_source="global_bank")

    def test_card_attribution_round_trip(self) -> None:
        ins = _insight(mechanism_source="memory_cards", card_id="card-abc-123")
        assert ins.card_id == "card-abc-123"
