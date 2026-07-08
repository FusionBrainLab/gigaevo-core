"""Mutator-facing rendering: efficacy endorsement lines and the card block."""

from __future__ import annotations

from gigaevo.memory.cards import CardKind, CardStatsBlock
from gigaevo.memory.read.render import EfficacyCardRenderer, format_block_efficacy


def _block(**overrides) -> CardStatsBlock:
    params = {
        "posterior_a": 5.0,
        "posterior_b": 1.0,
        "intro_events": 4,
        "k_harm": 0,
        "efficacy_confident": True,
        "IntroGain_best_median": 0.0123,
    }
    params.update(overrides)
    return CardStatsBlock(**params)


class TestFormatBlockEfficacy:
    def test_program_card_renders_exemplar_fitness(self, make_card):
        card = make_card(kind=CardKind.PROGRAM, program_id="p1", fitness=0.87654)
        assert format_block_efficacy(card, None) == "efficacy: exemplar fitness 0.8765"

    def test_program_card_without_fitness_is_silent(self, make_card):
        card = make_card(kind=CardKind.PROGRAM, program_id="p1")
        assert format_block_efficacy(card, None) == ""

    def test_insight_without_block_is_silent(self, make_card):
        assert format_block_efficacy(make_card(), None) == ""

    def test_insight_without_intros_or_median_is_silent(self, make_card):
        card = make_card()
        assert format_block_efficacy(card, _block(intro_events=0)) == ""
        assert format_block_efficacy(card, _block(IntroGain_best_median=None)) == ""

    def test_non_confident_insight_is_silent(self, make_card):
        block = _block(efficacy_confident=False)
        assert format_block_efficacy(make_card(), block) == ""

    def test_confident_positive_median(self, make_card):
        line = format_block_efficacy(make_card(), _block())
        assert line == (
            "efficacy: introduced in 4 children; median improvement +0.0123 (confident)"
        )

    def test_confident_non_positive_median_carries_caution(self, make_card):
        line = format_block_efficacy(make_card(), _block(IntroGain_best_median=-0.05))
        assert line.endswith("(caution: non-positive median)")
        assert "-0.0500" in line

    def test_bootstrap_ev_renders_expected_improvement_not_median(self, make_card):
        line = format_block_efficacy(
            make_card(),
            _block(
                IntroGain_best_median=0.10,
                IntroGain_bootstrap_ev_mean=0.025,
                IntroGain_bootstrap_ev_lo20=0.01,
            ),
        )
        assert line == (
            "efficacy: introduced in 4 children; "
            "expected improvement +0.0250 (confident)"
        )

    def test_bootstrap_ev_caution_uses_expected_improvement_label(self, make_card):
        line = format_block_efficacy(
            make_card(),
            _block(IntroGain_best_median=0.10, IntroGain_bootstrap_ev_mean=-0.025),
        )
        assert line.endswith("(caution: non-positive expected improvement)")
        assert "median improvement" not in line


class TestEfficacyCardRenderer:
    def test_none_card_renders_empty(self):
        assert EfficacyCardRenderer().render(None) == ""

    def test_description_only_when_no_endorsement(self, make_card):
        card = make_card(description="use symmetry breaking")
        assert EfficacyCardRenderer().render(card, None) == "use symmetry breaking"

    def test_description_plus_endorsement(self, make_card):
        card = make_card(description="use symmetry breaking")
        text = EfficacyCardRenderer().render(card, _block())
        assert text.splitlines() == [
            "use symmetry breaking",
            "efficacy: introduced in 4 children; median improvement +0.0123 (confident)",
        ]

    def test_blank_description_yields_endorsement_only(self, make_card):
        card = make_card(description="   ")
        text = EfficacyCardRenderer().render(card, _block())
        assert text.startswith("efficacy:")
