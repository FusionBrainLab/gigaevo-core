"""Fix 1: surface one legible per-card efficacy line to the mutator.

RED-phase TDD tests written before implementation. The mutator-facing render
(``EfficacyCardRenderer.render``) currently emits ``card.description`` only,
stripping the realized track record. These tests pin:

- ``format_card_efficacy``: MemoryCard -> endorsement line ONLY when the
  Beta-Binomial downside posterior is confident (``evolution_statistics["ALL"]``
  carries ``efficacy_confident``); non-confident / no-signal -> None.
  ProgramCard -> exemplar fitness.
- ``_render_card`` appends that line so the code-writing agent sees it.
"""

from __future__ import annotations

from gigaevo.memory.core import EfficacyCardRenderer
from gigaevo.memory.shared_memory.amem_gam_retriever import make_card_text
from gigaevo.memory.shared_memory.card_search import format_card_efficacy
from gigaevo.memory.shared_memory.models import (
    MemoryCard,
    MemoryCardExplanation,
    ProgramCard,
)

_DESC = "spatial target-encoding for capped target"


def _mcard(
    id: str = "m1",
    intro_events: int | None = None,
    median: float | None = None,
    downside: float | None = None,
    confident: bool | None = None,
    mechanism: str = "",
    adj_median: float | None = None,
) -> MemoryCard:
    all_block: dict = {}
    if intro_events is not None:
        all_block["intro_events"] = intro_events
    if median is not None:
        all_block["IntroGain_best_median"] = median
    if adj_median is not None:
        all_block["IntroGain_best_adj_median"] = adj_median
    if downside is not None:
        all_block["DownsideRate_best"] = downside
    if confident is not None:
        all_block["efficacy_confident"] = confident
    es = {"ALL": all_block} if all_block else {}
    return MemoryCard(
        id=id,
        description=_DESC,
        keywords=[],
        evolution_statistics=es,
        explanation=MemoryCardExplanation(summary=mechanism),
    )


def _pcard(id: str = "program-x", fitness: float | None = 0.85) -> ProgramCard:
    return ProgramCard(
        id=id,
        program_id=id.replace("program-", ""),
        description="CatBoost ensemble",
        keywords=[],
        fitness=fitness,
        connected_ideas=[],
    )


class TestFormatCardEfficacy:
    def test_non_confident_losing_card_is_silenced(self) -> None:
        # High-downside card (posterior not confident) emits no line at all.
        card = _mcard(intro_events=24, median=-0.0106, downside=0.71, confident=False)
        assert format_card_efficacy(card) is None

    def test_confident_winning_card_renders_endorsement_with_marker(self) -> None:
        card = _mcard(intro_events=3, median=0.012, downside=0.0, confident=True)
        assert format_card_efficacy(card) == (
            "efficacy: introduced in 3 children; "
            "median improvement +0.0120; downside 0% (confident)"
        )

    def test_confident_card_without_downside_omits_downside(self) -> None:
        card = _mcard(intro_events=3, median=0.012, confident=True)
        assert format_card_efficacy(card) == (
            "efficacy: introduced in 3 children; median improvement +0.0120 (confident)"
        )

    def test_confident_negative_median_renders_caution_not_endorsement(self) -> None:
        # A noise-band-confident posterior with a losing median must never read
        # as an endorsement to the mutator.
        card = _mcard(intro_events=4, median=-0.0338, downside=0.5, confident=True)
        assert format_card_efficacy(card) == (
            "efficacy: introduced in 4 children; "
            "median improvement -0.0338; downside 50% (caution: non-positive median)"
        )

    def test_confident_zero_median_renders_caution(self) -> None:
        card = _mcard(intro_events=3, median=0.0, confident=True)
        line = format_card_efficacy(card)
        assert line is not None
        assert "(confident)" not in line
        assert "(caution: non-positive median)" in line

    def test_intros_and_median_but_not_confident_returns_none(self) -> None:
        # Signal present but posterior not confident -> silenced.
        card = _mcard(intro_events=5, median=0.01, downside=0.2, confident=False)
        assert format_card_efficacy(card) is None

    def test_intros_and_median_missing_confident_flag_returns_none(self) -> None:
        # Absent efficacy_confident is treated as not-confident.
        card = _mcard(intro_events=5, median=0.01, downside=0.2)
        assert format_card_efficacy(card) is None

    def test_memory_card_zero_intros_returns_none(self) -> None:
        assert format_card_efficacy(_mcard(intro_events=0, median=None)) is None

    def test_memory_card_zero_intros_with_median_returns_none(self) -> None:
        # intros==0 must suppress the line even when a median is present.
        assert format_card_efficacy(_mcard(intro_events=0, median=0.01)) is None

    def test_memory_card_missing_all_block_returns_none(self) -> None:
        assert format_card_efficacy(_mcard()) is None

    def test_memory_card_intros_but_null_median_returns_none(self) -> None:
        assert (
            format_card_efficacy(_mcard(intro_events=5, median=None, confident=True))
            is None
        )

    def test_adjusted_median_preferred_over_raw_for_display_and_gate(self) -> None:
        # Cohort-adjusted median is the honest effect size: a card whose children
        # merely regress weak parents to the frontier has raw +0.0679 but adj 0.0
        # and must read as caution, never endorsement.
        card = _mcard(
            intro_events=18,
            median=0.0679,
            adj_median=0.0,
            downside=0.11,
            confident=True,
        )
        line = format_card_efficacy(card)
        assert line == (
            "efficacy: introduced in 18 children; "
            "median improvement vs cohort +0.0000; downside 11% "
            "(caution: non-positive median)"
        )

    def test_positive_adjusted_median_renders_endorsement(self) -> None:
        card = _mcard(
            intro_events=3,
            median=0.080,
            adj_median=0.030,
            downside=0.0,
            confident=True,
        )
        assert format_card_efficacy(card) == (
            "efficacy: introduced in 3 children; "
            "median improvement vs cohort +0.0300; downside 0% (confident)"
        )

    def test_legacy_block_without_adjusted_median_keeps_raw_wording(self) -> None:
        card = _mcard(intro_events=3, median=0.012, downside=0.0, confident=True)
        assert format_card_efficacy(card) == (
            "efficacy: introduced in 3 children; "
            "median improvement +0.0120; downside 0% (confident)"
        )

    def test_program_card_renders_fitness(self) -> None:
        assert format_card_efficacy(_pcard(fitness=0.852)) == (
            "efficacy: exemplar fitness 0.8520"
        )

    def test_program_card_null_fitness_returns_none(self) -> None:
        assert format_card_efficacy(_pcard(fitness=None)) is None

    def test_none_card_returns_none(self) -> None:
        assert format_card_efficacy(None) is None


class TestMutatorFacingRender:
    def test_render_card_appends_efficacy_for_confident_memory_card(self) -> None:
        rendered = EfficacyCardRenderer().render(
            _mcard(intro_events=3, median=0.012, downside=0.0, confident=True)
        )
        assert _DESC in rendered
        assert (
            "efficacy: introduced in 3 children; "
            "median improvement +0.0120; downside 0% (confident)"
        ) in rendered

    def test_render_card_silences_non_confident_memory_card(self) -> None:
        rendered = EfficacyCardRenderer().render(
            _mcard(intro_events=24, median=-0.0106, downside=0.71, confident=False)
        )
        assert rendered == _DESC
        assert "efficacy:" not in rendered

    def test_render_card_no_efficacy_line_when_no_signal(self) -> None:
        rendered = EfficacyCardRenderer().render(_mcard())
        assert rendered == _DESC
        assert "efficacy:" not in rendered

    def test_render_card_program_card_appends_fitness(self) -> None:
        rendered = EfficacyCardRenderer().render(_pcard(fitness=0.852))
        assert "efficacy: exemplar fitness 0.8520" in rendered

    def test_render_card_includes_mechanism_line(self) -> None:
        rendered = EfficacyCardRenderer().render(
            _mcard(mechanism="ratio features expose interactions to tree splits")
        )
        assert rendered == (
            f"{_DESC}\nmechanism: ratio features expose interactions to tree splits"
        )

    def test_render_card_mechanism_precedes_efficacy(self) -> None:
        rendered = EfficacyCardRenderer().render(
            _mcard(
                intro_events=3,
                median=0.012,
                downside=0.0,
                confident=True,
                mechanism="caps target leakage at fold boundaries",
            )
        )
        lines = rendered.splitlines()
        assert lines[0] == _DESC
        assert lines[1] == "mechanism: caps target leakage at fold boundaries"
        assert lines[2].startswith("efficacy:")

    def test_render_card_skips_mechanism_identical_to_description(self) -> None:
        rendered = EfficacyCardRenderer().render(_mcard(mechanism=_DESC))
        assert rendered == _DESC

    def test_render_card_coerced_explanation_summary(self) -> None:
        rendered = EfficacyCardRenderer().render(
            MemoryCard(
                id="m-expl",
                description=_DESC,
                explanation={"summary": "smooths sparse categories"},
            )
        )
        assert "mechanism: smooths sparse categories" in rendered


class TestRetrievalCorpusRender:
    """``make_card_text`` (GAM corpus) drops the dead ``usage`` / raw stats blobs
    and emits the same legible ``efficacy:`` line instead."""

    def test_confident_idea_record_emits_efficacy_line_not_raw_blobs(self) -> None:
        record = {
            "id": "9f0dfb8a",
            "description": _DESC,
            "evolution_statistics": {
                "ALL": {
                    "intro_events": 3,
                    "IntroGain_best_median": 0.012,
                    "DownsideRate_best": 0.0,
                    "efficacy_confident": True,
                }
            },
            "usage": {"total_used": 0, "median_delta_fitness": None},
        }
        text = make_card_text(record)
        assert (
            "efficacy: introduced in 3 children; "
            "median improvement +0.0120; downside 0% (confident)"
        ) in text
        assert "usage:" not in text
        assert "evolution_statistics:" not in text

    def test_non_confident_idea_record_silenced_and_no_raw_blobs(self) -> None:
        record = {
            "id": "9f0dfb8a",
            "description": _DESC,
            "evolution_statistics": {
                "ALL": {
                    "intro_events": 24,
                    "IntroGain_best_median": -0.0106,
                    "DownsideRate_best": 0.71,
                    "efficacy_confident": False,
                }
            },
            "usage": {"total_used": 0, "median_delta_fitness": None},
        }
        text = make_card_text(record)
        assert "efficacy:" not in text
        assert "usage:" not in text
        assert "evolution_statistics:" not in text

    def test_program_record_emits_efficacy_line(self) -> None:
        record = {
            "id": "program-abc",
            "description": "CatBoost ensemble",
            "category": "program",
            "fitness": 0.852,
        }
        text = make_card_text(record)
        assert "efficacy: exemplar fitness 0.8520" in text
        assert "usage:" not in text

    def test_record_without_signal_omits_efficacy_line(self) -> None:
        record = {"id": "m9", "description": _DESC}
        text = make_card_text(record)
        assert "efficacy:" not in text
        assert "usage:" not in text
        assert "evolution_statistics:" not in text

    def test_content_alias_resolves_into_description_line(self) -> None:
        text = make_card_text({"id": "m1", "content": "alias body"})
        assert "description: alias body" in text

    def test_string_explanation_renders_as_summary(self) -> None:
        text = make_card_text(
            {"id": "m2", "description": _DESC, "explanation": "why it works"}
        )
        assert "explanation_summary: why it works" in text

    def test_program_record_renders_program_fields_only(self) -> None:
        text = make_card_text(
            {
                "id": "program-abc",
                "category": "program",
                "program_id": "abc",
                "fitness": 0.852,
            }
        )
        assert "program_id: abc" in text
        assert "explanation_summary:" not in text


class TestGamPageMetaEnvelope:
    def test_reads_amem_id_and_ignores_vendor_payload(self) -> None:
        from gigaevo.memory.shared_memory.amem_gam_retriever import GamPageMeta

        meta = GamPageMeta.model_validate({"amem_id": "m1", "amem": {"id": "m1"}})
        assert meta.amem_id == "m1"

    def test_missing_or_null_amem_id_reads_as_empty(self) -> None:
        from gigaevo.memory.shared_memory.amem_gam_retriever import GamPageMeta

        assert GamPageMeta.model_validate({}).amem_id == ""
        assert GamPageMeta.model_validate({"amem_id": None}).amem_id == ""
