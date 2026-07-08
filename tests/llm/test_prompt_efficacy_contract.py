"""Contract between the card renderer and the prompts that consume its output.

The mutation-suggestion analyst describes the per-card ``efficacy:`` line it will
be handed; the mutator acts on the analyst's ranking. Both prompts are
``.format()``-rendered at agent construction. These tests guard the two silent
ways a prompt edit breaks the system:

1. **Drift** — the analyst describing a stale/removed efficacy field (e.g. the
   long-gone "downside rate") so it reasons about input it never receives. The
   expected tokens are derived from the real ``format_block_efficacy`` so the
   prompt and the renderer cannot drift apart unnoticed.
2. **Unescaped brace** — a literal ``{``/``}`` that was not doubled crashes
   ``.format()`` (KeyError/ValueError = silent run death at startup).
"""

from __future__ import annotations

from gigaevo.memory.cards import Card, CardKind, CardStatsBlock
from gigaevo.memory.read.render import format_block_efficacy
from gigaevo.prompts import MutationSuggestionsPrompts, load_prompt


def _mcard() -> Card:
    return Card(id="m1", description="d")


def _confident_positive_line() -> str:
    return format_block_efficacy(
        _mcard(),
        CardStatsBlock(
            intro_events=3, IntroGain_best_median=0.012, efficacy_confident=True
        ),
    )


def _confident_expected_line() -> str:
    return format_block_efficacy(
        _mcard(),
        CardStatsBlock(
            intro_events=3,
            IntroGain_best_median=0.020,
            IntroGain_bootstrap_ev_mean=0.012,
            IntroGain_bootstrap_ev_lo20=0.006,
            efficacy_confident=True,
        ),
    )


def _caution_line() -> str:
    return format_block_efficacy(
        _mcard(),
        CardStatsBlock(
            intro_events=4, IntroGain_best_median=-0.034, efficacy_confident=True
        ),
    )


def _exemplar_line() -> str:
    return format_block_efficacy(
        Card(
            id="program-x",
            kind=CardKind.PROGRAM,
            program_id="x",
            description="d",
            fitness=0.85,
        ),
        None,
    )


class TestEfficacyDescriptionMatchesRenderer:
    """The analyst must describe the line it will actually be handed."""

    def test_prompt_describes_confident_endorsement_tokens(self) -> None:
        prompt = MutationSuggestionsPrompts.system()
        line = _confident_positive_line()
        for token in ("introduced in", "median improvement", "(confident)"):
            assert token in line, f"renderer no longer emits {token!r}"
            assert token in prompt, f"prompt omits renderer token {token!r}"

    def test_prompt_describes_bootstrap_expected_improvement_tokens(self) -> None:
        prompt = MutationSuggestionsPrompts.system()
        line = _confident_expected_line()
        for token in ("expected improvement", "(confident)"):
            assert token in line, f"renderer no longer emits {token!r}"
            assert token in prompt, f"prompt omits renderer token {token!r}"

    def test_prompt_describes_caution_token(self) -> None:
        assert "(caution: non-positive median)" in _caution_line()
        assert "(caution: non-positive median)" in MutationSuggestionsPrompts.system()
        assert (
            "(caution: non-positive expected improvement)"
            in MutationSuggestionsPrompts.system()
        )

    def test_prompt_describes_exemplar_token(self) -> None:
        assert "exemplar fitness" in _exemplar_line()
        assert "exemplar fitness" in MutationSuggestionsPrompts.system()

    def test_prompt_drops_removed_downside_field(self) -> None:
        # The refactor removed the downside-rate field from the rendered line;
        # the analyst must not describe input it never receives.
        assert "downside" not in _confident_positive_line()
        assert "downside" not in _caution_line()
        assert "downside" not in MutationSuggestionsPrompts.system().lower()


class TestSystemPromptsFormatCleanly:
    """Literal braces must stay doubled — ``.format()`` with the agent's real
    fields must not raise."""

    def test_mutation_suggestions_system_formats(self) -> None:
        rendered = load_prompt("mutation_suggestions", "system").format(
            task_description="T", metrics_description="M", max_insights=5
        )
        for placeholder in (
            "{task_description}",
            "{metrics_description}",
            "{max_insights}",
        ):
            assert placeholder not in rendered

    def test_mutation_system_formats(self) -> None:
        rendered = load_prompt("mutation", "system").format(
            task_description="T", metrics_description="M"
        )
        for placeholder in ("{task_description}", "{metrics_description}"):
            assert placeholder not in rendered
