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
    def test_foreign_help_rate_is_appended_without_a_magnitude(self, make_card):
        line = format_block_efficacy(
            make_card(),
            _block(foreign_help_events=1.5, foreign_total_events=2.5),
        )
        assert line.splitlines() == [
            "efficacy: introduced in 4 children; median improvement +0.0123 (confident)",
            "helped in 1.5 of 2.5 uses on other tasks",
        ]

    def test_foreign_only_evidence_renders_sign_rate_only(self, make_card):
        line = format_block_efficacy(
            make_card(),
            _block(
                intro_events=0,
                IntroGain_best_median=None,
                efficacy_confident=False,
                foreign_help_events=2,
                foreign_total_events=3,
            ),
        )
        assert line == "helped in 2 of 3 uses on other tasks"

    def test_zero_foreign_evidence_keeps_output_byte_identical(self, make_card):
        assert format_block_efficacy(make_card(), _block()) == (
            "efficacy: introduced in 4 children; median improvement +0.0123 (confident)"
        )

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

    def test_foreign_program_suppresses_fitness_and_appends_origin(self, make_card):
        card = make_card(
            kind=CardKind.PROGRAM,
            program_id="p1",
            fitness=0.87654,
            description="reuse staged search",
            task_key="origin-task",
        )

        text = EfficacyCardRenderer(task_key="current-task").render(card)

        assert text.splitlines() == [
            "reuse staged search",
            "evidence from a different task (origin-task)",
        ]
        assert "efficacy:" not in text

    def test_foreign_insight_keeps_efficacy_and_appends_origin(self, make_card):
        card = make_card(description="use symmetry breaking", task_key="origin-task")

        text = EfficacyCardRenderer(task_key="current-task").render(card, _block())

        assert text.splitlines() == [
            "use symmetry breaking",
            "efficacy: introduced in 4 children; median improvement +0.0123 (confident)",
            "evidence from a different task (origin-task)",
        ]

    def test_same_task_render_is_unchanged(self, make_card):
        card = make_card(description="use symmetry breaking", task_key="current-task")

        text = EfficacyCardRenderer(task_key="current-task").render(card, _block())

        assert text == (
            "use symmetry breaking\n"
            "efficacy: introduced in 4 children; median improvement +0.0123 (confident)"
        )

    def test_unstamped_renderer_ignores_card_task(self, make_card):
        card = make_card(description="use symmetry breaking", task_key="origin-task")

        text = EfficacyCardRenderer().render(card, _block())

        assert text == (
            "use symmetry breaking\n"
            "efficacy: introduced in 4 children; median improvement +0.0123 (confident)"
        )

    def test_legacy_card_has_no_foreign_provenance(self, make_card):
        card = make_card(description="use symmetry breaking", task_key="")

        text = EfficacyCardRenderer(task_key="current-task").render(card, _block())

        assert text == (
            "use symmetry breaking\n"
            "efficacy: introduced in 4 children; median improvement +0.0123 (confident)"
        )
