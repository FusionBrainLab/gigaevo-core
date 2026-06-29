"""Fix 1: surface one legible per-card efficacy line to the mutator.

The mutator-facing render (``EfficacyCardRenderer.render``) and the context-free
``format_card_efficacy`` emit the realized track record. These pin:

- ``format_card_efficacy``: MemoryCard -> endorsement line ONLY when the
  Beta-Binomial downside posterior reputation computes from the card's gain
  events is confident; non-confident / no-signal -> None. ProgramCard -> its
  exemplar capture fitness (source-blind, block-independent).
- ``format_block_efficacy``: the rendering branch, given an explicit stats block.
- ``EfficacyCardRenderer.render`` appends that line so the code-writing agent
  sees it (the caller resolves the block via reputation, as in the read pipeline).
"""

from __future__ import annotations

from gigaevo.memory.context import ContextualGain, DecisionContext
from gigaevo.memory.core import EfficacyCardRenderer
from gigaevo.memory.shared_memory.amem_gam_retriever import (
    build_gam_store,
    make_card_text,
)
from gigaevo.memory.shared_memory.card_search import (
    format_block_efficacy,
    format_card_efficacy,
)
from gigaevo.memory.shared_memory.models import (
    CardStatsBlock,
    MemoryCard,
    ProgramCard,
)

_DESC = "spatial target-encoding for capped target"


def _g(gain: float) -> ContextualGain:
    return ContextualGain(
        context=DecisionContext(parent_metrics={"min_area": 0.5}), gain=gain
    )


def _events(gains: list[float]) -> list[ContextualGain]:
    return [_g(g) for g in gains]


def _mcard(
    id: str = "m1",
    gain_events: list[ContextualGain] | None = None,
) -> MemoryCard:
    return MemoryCard(
        id=id,
        description=_DESC,
        keywords=[],
        gain_events=gain_events,
    )


def _pcard(
    id: str = "program-x",
    fitness: float | None = 0.85,
    gain_events: list[ContextualGain] | None = None,
) -> ProgramCard:
    return ProgramCard(
        id=id,
        program_id=id.replace("program-", ""),
        description="CatBoost ensemble",
        keywords=[],
        fitness=fitness,
        gain_events=gain_events,
    )


# N equal positive gains -> Beta(N+1, 1): confident, median = the gain.
def _confident_wins(n: int, gain: float = 0.012) -> list[ContextualGain]:
    return _events([gain] * n)


class TestFormatCardEfficacy:
    def test_non_confident_losing_card_is_silenced(self) -> None:
        # All-negative gains -> harmful posterior, not confidently helpful.
        card = _mcard(gain_events=_events([-0.0106] * 24))
        assert format_card_efficacy(card) is None

    def test_confident_winning_card_renders_endorsement_with_marker(self) -> None:
        card = _mcard(gain_events=_confident_wins(3))
        assert format_card_efficacy(card) == (
            "efficacy: introduced in 3 children; median improvement +0.0120 (confident)"
        )

    def test_confident_zero_median_renders_caution(self) -> None:
        card = _mcard(gain_events=_events([0.0] * 3))
        line = format_card_efficacy(card)
        assert line is not None
        assert "(confident)" not in line
        assert "(caution: non-positive median)" in line

    def test_signal_present_but_not_confident_returns_none(self) -> None:
        # A single positive event: median > 0 but the pessimistic lower bound on
        # P(help) stays under the threshold, so the card stays silent.
        card = _mcard(gain_events=_events([0.01]))
        assert format_card_efficacy(card) is None

    def test_memory_card_no_events_returns_none(self) -> None:
        assert format_card_efficacy(_mcard(gain_events=None)) is None

    def test_memory_card_empty_events_returns_none(self) -> None:
        assert format_card_efficacy(_mcard(gain_events=[])) is None

    def test_program_card_renders_exemplar_fitness(self) -> None:
        # Source-blind: a program card surfaces its own capture fitness, never a
        # posterior — the rendering is block-independent.
        assert format_card_efficacy(_pcard(fitness=0.852)) == (
            "efficacy: exemplar fitness 0.8520"
        )

    def test_program_card_null_fitness_returns_none(self) -> None:
        assert format_card_efficacy(_pcard(fitness=None)) is None

    def test_none_card_returns_none(self) -> None:
        assert format_card_efficacy(None) is None


class TestFormatBlockEfficacy:
    """The rendering branch given an explicit stats block (the locality the
    auction bid on); reputation feeds it the card's resolved ``card_stats``."""

    def test_confident_positive_median_renders_endorsement(self) -> None:
        block = CardStatsBlock(
            intro_events=3, IntroGain_best_median=0.012, efficacy_confident=True
        )
        assert format_block_efficacy(_mcard(), block) == (
            "efficacy: introduced in 3 children; median improvement +0.0120 (confident)"
        )

    def test_confident_negative_median_renders_caution_not_endorsement(self) -> None:
        # A noise-band-confident posterior with a losing median must never read
        # as an endorsement to the mutator.
        block = CardStatsBlock(
            intro_events=4, IntroGain_best_median=-0.0338, efficacy_confident=True
        )
        assert format_block_efficacy(_mcard(), block) == (
            "efficacy: introduced in 4 children; "
            "median improvement -0.0338 (caution: non-positive median)"
        )

    def test_non_confident_block_is_silenced(self) -> None:
        block = CardStatsBlock(
            intro_events=5, IntroGain_best_median=0.01, efficacy_confident=False
        )
        assert format_block_efficacy(_mcard(), block) is None

    def test_zero_intros_returns_none(self) -> None:
        block = CardStatsBlock(
            intro_events=0, IntroGain_best_median=0.01, efficacy_confident=True
        )
        assert format_block_efficacy(_mcard(), block) is None

    def test_null_median_returns_none(self) -> None:
        block = CardStatsBlock(
            intro_events=5, IntroGain_best_median=None, efficacy_confident=True
        )
        assert format_block_efficacy(_mcard(), block) is None

    def test_missing_block_returns_none(self) -> None:
        assert format_block_efficacy(_mcard(), None) is None

    def test_program_card_renders_exemplar_fitness_block_independent(self) -> None:
        block = CardStatsBlock(
            intro_events=4, IntroGain_best_median=0.012, efficacy_confident=True
        )
        assert format_block_efficacy(_pcard(fitness=0.852), block) == (
            "efficacy: exemplar fitness 0.8520"
        )

    def test_program_card_null_fitness_returns_none(self) -> None:
        block = CardStatsBlock(
            intro_events=4, IntroGain_best_median=0.012, efficacy_confident=True
        )
        assert format_block_efficacy(_pcard(fitness=None), block) is None


class TestMutatorFacingRender:
    def test_render_card_appends_efficacy_for_confident_memory_card(self) -> None:
        block = CardStatsBlock(
            intro_events=3, IntroGain_best_median=0.012, efficacy_confident=True
        )
        rendered = EfficacyCardRenderer().render(_mcard(), block)
        assert _DESC in rendered
        assert (
            "efficacy: introduced in 3 children; median improvement +0.0120 (confident)"
        ) in rendered

    def test_render_card_silences_non_confident_memory_card(self) -> None:
        block = CardStatsBlock(
            intro_events=24, IntroGain_best_median=-0.0106, efficacy_confident=False
        )
        rendered = EfficacyCardRenderer().render(_mcard(), block)
        assert rendered == _DESC
        assert "efficacy:" not in rendered

    def test_render_card_no_efficacy_line_when_no_block(self) -> None:
        rendered = EfficacyCardRenderer().render(_mcard(), None)
        assert rendered == _DESC
        assert "efficacy:" not in rendered

    def test_render_card_program_card_appends_exemplar_fitness(self) -> None:
        rendered = EfficacyCardRenderer().render(_pcard(fitness=0.852), None)
        assert "efficacy: exemplar fitness 0.8520" in rendered

    def test_render_card_program_card_null_fitness_silenced(self) -> None:
        rendered = EfficacyCardRenderer().render(_pcard(fitness=None), None)
        assert "efficacy:" not in rendered


class TestRetrievalCorpusRender:
    """``make_card_text`` (GAM corpus) drops the dead ``usage`` blob and emits the
    same legible ``efficacy:`` line, computed from the card's gain events."""

    def test_confident_idea_record_emits_efficacy_line_not_raw_blobs(self) -> None:
        record = {
            "id": "9f0dfb8a",
            "description": _DESC,
            "gain_events": [_g(0.012).model_dump() for _ in range(3)],
            "usage": {"total_used": 0, "median_delta_fitness": None},
        }
        text = make_card_text(record)
        assert (
            "efficacy: introduced in 3 children; median improvement +0.0120 (confident)"
        ) in text
        assert "usage:" not in text
        assert "evolution_statistics:" not in text

    def test_non_confident_idea_record_silenced_and_no_raw_blobs(self) -> None:
        record = {
            "id": "9f0dfb8a",
            "description": _DESC,
            "gain_events": [_g(-0.0106).model_dump() for _ in range(24)],
            "usage": {"total_used": 0, "median_delta_fitness": None},
        }
        text = make_card_text(record)
        assert "efficacy:" not in text
        assert "usage:" not in text
        assert "evolution_statistics:" not in text

    def test_program_record_emits_exemplar_fitness_line(self) -> None:
        record = {
            "id": "program-abc",
            "description": "CatBoost ensemble",
            "category": "program",
            "fitness": 0.852,
            "gain_events": [_g(0.012).model_dump() for _ in range(4)],
        }
        text = make_card_text(record)
        assert "efficacy: exemplar fitness 0.8520" in text
        assert "usage:" not in text

    def test_program_record_capture_fitness_only_emits_exemplar_fitness(self) -> None:
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

    def test_idea_record_renders_explanation_summary(self) -> None:
        text = make_card_text(
            {
                "id": "m1",
                "description": "spectral gap widening",
                "explanation_summary": "why the gap widens under this lever",
            }
        )
        assert "explanation_summary: why the gap widens under this lever" in text


class TestGamMemoryAbstract:
    """The GAM long-term-memory abstract is the always-on planning-context
    channel (``_planning`` injects each as ``Page {i}: {abstract}``), distinct
    from page-content retrieval. Idea-card abstracts must carry the why-text;
    program exemplar abstracts stay description-only (their explanation rides
    page metadata, matching the page-text projection)."""

    def test_idea_abstract_includes_explanation_summary(self, tmp_path) -> None:
        memory_store, _, _ = build_gam_store(
            [
                {
                    "id": "m1",
                    "description": "widen the spectral gap before pruning",
                    "explanation_summary": (
                        "pruning collapses the gap that carries the signal"
                    ),
                }
            ],
            tmp_path,
        )
        (abstract,) = memory_store.load().abstracts
        assert "pruning collapses the gap that carries the signal" in abstract
        assert "widen the spectral gap before pruning" in abstract

    def test_program_abstract_stays_description_only(self, tmp_path) -> None:
        memory_store, _, _ = build_gam_store(
            [
                {
                    "id": "program-abc",
                    "category": "program",
                    "program_id": "abc",
                    "description": "CatBoost ensemble",
                    "explanation_summary": "ensembling lowers variance",
                }
            ],
            tmp_path,
        )
        (abstract,) = memory_store.load().abstracts
        assert abstract == "CatBoost ensemble"


class TestGamAbstractStoreOverwrite:
    """Each rebuild overwrites the planning-abstract store with ONLY the current
    cards' abstracts, mirroring the page store. Edited or removed cards must not
    leave a stale abstract behind in the always-on planning channel."""

    def test_rebuild_drops_stale_abstract_when_card_edited(self, tmp_path) -> None:
        build_gam_store(
            [
                {
                    "id": "m1",
                    "description": "widen the gap",
                    "explanation_summary": "old",
                }
            ],
            tmp_path,
        )
        memory_store, _, _ = build_gam_store(
            [
                {
                    "id": "m1",
                    "description": "widen the gap",
                    "explanation_summary": "new",
                }
            ],
            tmp_path,
        )
        assert memory_store.load().abstracts == ["widen the gap — new"]

    def test_rebuild_drops_abstract_of_removed_card(self, tmp_path) -> None:
        build_gam_store(
            [
                {"id": "m1", "description": "keep me", "explanation_summary": "a"},
                {"id": "m2", "description": "drop me", "explanation_summary": "b"},
            ],
            tmp_path,
        )
        memory_store, _, _ = build_gam_store(
            [{"id": "m1", "description": "keep me", "explanation_summary": "a"}],
            tmp_path,
        )
        assert memory_store.load().abstracts == ["keep me — a"]


class TestGamPageMetaEnvelope:
    def test_reads_amem_id_and_ignores_vendor_payload(self) -> None:
        from gigaevo.memory.shared_memory.amem_gam_retriever import GamPageMeta

        meta = GamPageMeta.model_validate({"amem_id": "m1", "amem": {"id": "m1"}})
        assert meta.amem_id == "m1"

    def test_missing_or_null_amem_id_reads_as_empty(self) -> None:
        from gigaevo.memory.shared_memory.amem_gam_retriever import GamPageMeta

        assert GamPageMeta.model_validate({}).amem_id == ""
        assert GamPageMeta.model_validate({"amem_id": None}).amem_id == ""
