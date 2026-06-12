"""Typed evolution-statistics contract: cards carry CardStatsBlock /
EvolutionStatistics models, and every consumer reads typed fields — no
Mapping/getattr dual paths anywhere in the card decision paths."""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
from pydantic import ValidationError
import pytest

from gigaevo.memory.core.admitter import harm_statistics
from gigaevo.memory.core.auctioneer import (
    AuctionBid,
    AuctionCandidate,
    ThompsonAuctioneer,
)
from gigaevo.memory.core.budgeter import TopThetaBudgeter
from gigaevo.memory.core.evictor import HarmEvictor
from gigaevo.memory.core.idea_stats import IdeaStats
from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.core.selection import MemorySelection
from gigaevo.memory.shared_memory.card_search import format_card_efficacy
from gigaevo.memory.shared_memory.injection_posterior import InjectionOutcome
from gigaevo.memory.shared_memory.models import (
    CardAlias,
    CardStatsBlock,
    EvolutionStatistics,
    MemoryCard,
    ProgramCard,
    Quartile,
)
from gigaevo.memory.write_pipeline import (
    fold_best_idea_metrics,
    parse_best_ideas,
    stamp_card_posterior,
)

BANK_STATS = {
    "ALL": {
        "intro_events": 7,
        "IntroGain_best_p10": -0.01,
        "IntroGain_best_median": 0.004,
        "IntroGain_best_adj_median": 0.006,
        "IntroGain_best_rel_median": 0.1,
        "IntroGain_best_p90": 0.02,
        "DownsideRate_best": 0.142857,
        "TailRisk_best_median": -0.002,
        "posterior_a": 7.0,
        "posterior_b": 2.0,
        "p_help_mean": 0.7778,
        "p_help_lo20": 0.62,
        "efficacy_confident": True,
        "SiblingWinRate": None,
    },
    "Q1": {
        "intro_events": 2,
        "posterior_a": 2.0,
        "posterior_b": 2.0,
        "p_help_mean": 0.5,
        "p_help_lo20": 0.3,
        "efficacy_confident": False,
    },
    "best_ideas_snapshot": {
        "IntroGain_best_median": 0.004,
        "SiblingDelta_median": None,
    },
}


class TestEvolutionStatisticsModel:
    def test_bank_shaped_dict_roundtrips_losslessly(self):
        stats = EvolutionStatistics.model_validate(BANK_STATS)
        assert stats.model_dump() == BANK_STATS

    def test_typed_reads_on_all_block(self):
        stats = EvolutionStatistics.model_validate(BANK_STATS)
        assert stats.ALL is not None
        assert stats.ALL.posterior_a == 7.0
        assert stats.ALL.posterior_b == 2.0
        assert stats.ALL.intro_events == 7
        assert stats.ALL.efficacy_confident is True
        assert stats.ALL.IntroGain_best_adj_median == 0.006
        assert stats.ALL.DownsideRate_best == 0.142857
        assert stats.ALL.TailRisk_best_median == -0.002
        assert stats.Q1 is not None and stats.Q1.efficacy_confident is False

    def test_metric_vocabulary_is_first_class_and_described(self):
        metric_fields = set(IdeaStats.model_fields) - {
            "idea_id",
            "quartile",
            "description",
        }
        assert metric_fields <= set(CardStatsBlock.model_fields)
        assert IdeaStats.model_config["extra"] == "forbid"
        undescribed = [
            f"{model.__name__}.{name}"
            for model in (CardStatsBlock, IdeaStats)
            for name, field in model.model_fields.items()
            if not field.description
        ]
        assert not undescribed

    def test_quartile_is_a_first_class_type(self):
        assert [q.value for q in Quartile] == ["Q1", "Q2", "Q3", "Q4", "ALL"]
        assert Quartile.quarters() == (
            Quartile.Q1,
            Quartile.Q2,
            Quartile.Q3,
            Quartile.Q4,
        )
        assert IdeaStats.model_fields["quartile"].annotation is Quartile
        row = IdeaStats(idea_id="i1", quartile="ALL")
        assert row.quartile is Quartile.ALL

    def test_stats_blocks_reject_undeclared_keys(self):
        assert CardStatsBlock.model_config["extra"] == "forbid"
        assert EvolutionStatistics.model_config["extra"] == "forbid"

    def test_with_block_and_from_blocks_build_typed_statistics(self):
        block = CardStatsBlock(posterior_a=3.0, posterior_b=1.0, intro_events=2)
        stats = EvolutionStatistics.from_blocks({Quartile.Q2: block})
        assert stats.Q2 == block and stats.ALL is None
        replaced = stats.with_block(Quartile.ALL, block)
        assert replaced.ALL == block and replaced.Q2 == block
        assert stats.ALL is None

    def test_idea_stats_projects_to_stats_block_without_identity_keys(self):
        row = IdeaStats(
            idea_id="i1",
            quartile=Quartile.ALL,
            description="d",
            intro_events=3,
            IntroGain_best_median=float("nan"),
            posterior_a=3.0,
        )
        block = row.to_stats_block()
        assert isinstance(block, CardStatsBlock)
        assert block.IntroGain_best_median is None
        assert block.posterior_a == 3.0
        assert block.model_extra in (None, {})

    def test_harm_statistics_carries_no_identity_keys(self):
        row = IdeaStats(idea_id="i1", quartile=Quartile.ALL, posterior_a=1.0)
        stats = harm_statistics(row)
        assert stats.ALL is not None
        assert stats.ALL.model_extra in (None, {})

    def test_absent_blocks_stay_absent_in_dump(self):
        stats = EvolutionStatistics.model_validate({"ALL": {"intro_events": 1}})
        dumped = stats.model_dump()
        assert "Q1" not in dumped
        assert "best_ideas_snapshot" not in dumped

    def test_empty_statistics_dump_to_empty_dict(self):
        assert EvolutionStatistics().model_dump() == {}

    def test_cards_coerce_raw_dict_statistics(self):
        card = MemoryCard(id="idea-1", evolution_statistics=BANK_STATS)
        assert isinstance(card.evolution_statistics, EvolutionStatistics)
        assert card.evolution_statistics.ALL.p_help_lo20 == 0.62
        prog = ProgramCard(id="program-p1", program_id="p1")
        assert prog.evolution_statistics.ALL is None


class TestReputationTypedReads:
    def test_card_posterior_reads_stamped_block(self):
        rep = BetaBinomialReputation()
        card = MemoryCard(id="idea-1", evolution_statistics=BANK_STATS)
        assert rep.card_posterior(card) == (7.0, 2.0)

    def test_card_posterior_cold_prior_without_all_block(self):
        rep = BetaBinomialReputation()
        assert rep.card_posterior(MemoryCard(id="idea-2")) == rep.cold_prior

    def test_card_posterior_cold_prior_when_posterior_fields_missing(self):
        rep = BetaBinomialReputation()
        card = MemoryCard(
            id="idea-3", evolution_statistics={"ALL": {"intro_events": 4}}
        )
        assert rep.card_posterior(card) == rep.cold_prior

    def test_is_confidently_harmful_takes_typed_statistics(self):
        rep = BetaBinomialReputation()
        harmful = EvolutionStatistics(
            ALL=CardStatsBlock(posterior_a=1.0, posterior_b=9.0, intro_events=8)
        )
        helpful = EvolutionStatistics(
            ALL=CardStatsBlock(posterior_a=9.0, posterior_b=1.0, intro_events=8)
        )
        thin = EvolutionStatistics(
            ALL=CardStatsBlock(posterior_a=1.0, posterior_b=9.0, intro_events=2)
        )
        assert rep.is_confidently_harmful(harmful) is True
        assert rep.is_confidently_harmful(helpful) is False
        assert rep.is_confidently_harmful(thin) is False
        assert rep.is_confidently_harmful(EvolutionStatistics()) is False
        assert rep.is_confidently_harmful(None) is False


class TestEvictorTypedReads:
    def test_should_evict_reads_card_statistics(self):
        evictor = HarmEvictor(reputation=BetaBinomialReputation())
        harmful = MemoryCard(
            id="bad",
            evolution_statistics={
                "ALL": {"posterior_a": 1.0, "posterior_b": 9.0, "intro_events": 8}
            },
        )
        assert evictor.should_evict(harmful) is True
        assert evictor.should_evict(MemoryCard(id="cold")) is False
        assert evictor.sweep({"bad": harmful, "cold": MemoryCard(id="cold")}) == ["bad"]


class TestEfficacyRenderingTypedReads:
    def test_confident_memory_card_renders_endorsement(self):
        card = MemoryCard(id="idea-1", evolution_statistics=BANK_STATS)
        line = format_card_efficacy(card)
        assert line is not None
        assert "introduced in 7 children" in line
        assert "vs cohort" in line
        assert "(confident)" in line

    def test_non_confident_card_stays_silent(self):
        stats = {
            "ALL": {
                "intro_events": 2,
                "IntroGain_best_median": 0.01,
                "efficacy_confident": False,
            }
        }
        assert (
            format_card_efficacy(MemoryCard(id="idea-2", evolution_statistics=stats))
            is None
        )

    def test_program_card_renders_exemplar_fitness(self):
        card = ProgramCard(id="program-p1", program_id="p1", fitness=0.8712)
        assert format_card_efficacy(card) == "efficacy: exemplar fitness 0.8712"
        assert (
            format_card_efficacy(ProgramCard(id="program-p2", program_id="p2")) is None
        )


class TestStampCardPosteriorTyped:
    def test_stamp_replaces_all_block_and_keeps_snapshot(self):
        card = MemoryCard(
            id="idea-1",
            evolution_statistics={
                "best_ideas_snapshot": {"IntroGain_best_median": 0.004}
            },
        )
        posterior = CardStatsBlock(
            posterior_a=5.0, posterior_b=1.0, intro_events=4, efficacy_confident=True
        )
        stamped = stamp_card_posterior(card, {"idea-1": posterior})
        assert stamped.evolution_statistics.ALL == posterior
        assert (
            stamped.evolution_statistics.best_ideas_snapshot.IntroGain_best_median
            == 0.004
        )
        assert card.evolution_statistics.ALL is None

    def test_cards_without_posterior_pass_through_cold(self):
        card = MemoryCard(id="idea-2")
        assert stamp_card_posterior(card, {}) is card

    def test_compute_injection_posteriors_returns_typed_blocks(self):
        rep = BetaBinomialReputation()
        rng = np.random.default_rng(0)
        programs = [
            InjectionOutcome(
                id=f"p{i}",
                fitness=0.5 + 0.01 * i + float(rng.normal(0, 0.001)),
                parents=[f"p{i - 1}"] if i else [],
                selected_ids=["idea-1"] if i % 2 else [],
            )
            for i in range(30)
        ]
        posteriors = rep.compute_injection_posteriors(programs)
        assert posteriors
        assert all(isinstance(b, CardStatsBlock) for b in posteriors.values())


def _bid(card_id: str, theta: float) -> AuctionBid:
    return AuctionBid(
        card_id=card_id,
        posterior_a=2.0,
        posterior_b=2.0,
        theta=theta,
        baseline_a=3.0,
        baseline_b=3.0,
        baseline_theta=0.4,
        selected=theta > 0.4,
    )


class TestTypedAuctionSlate:
    def test_auction_emits_typed_bids(self):
        candidates = [
            AuctionCandidate(card_id="a", posterior_a=4.0, posterior_b=1.0),
            AuctionCandidate(card_id="b", posterior_a=1.0, posterior_b=4.0),
        ]
        winners, slate = ThompsonAuctioneer().run(candidates, np.random.default_rng(7))
        assert [bid.card_id for bid in slate] == ["a", "b"]
        for bid in slate:
            assert isinstance(bid, AuctionBid)
            assert bid.selected == (bid.theta > bid.baseline_theta)
            assert (bid.card_id in winners) == bid.selected

    def test_budgeter_caps_by_bid_theta(self):
        slate = [_bid("a", 0.2), _bid("b", 0.9), _bid("c", 0.5)]
        assert TopThetaBudgeter().cap(["a", "b", "c"], slate, 2) == ["b", "c"]

    def test_selection_slate_is_typed(self):
        sel = MemorySelection(cards=["text"], card_ids=["a"], slate=[_bid("a", 0.5)])
        assert isinstance(sel.slate[0], AuctionBid)

    def test_auction_models_are_strict_and_described(self):
        for model in (AuctionCandidate, AuctionBid):
            assert model.model_config.get("extra") == "forbid"
            undocumented = [
                name
                for name, field in model.model_fields.items()
                if not field.description
            ]
            assert not undocumented, undocumented


class TestCardAlias:
    def test_aliases_validate_into_typed_entries(self):
        card = MemoryCard(
            id="x",
            aliases=[
                {
                    "key": "x-update",
                    "description": "old wording",
                    "programs": ["p1"],
                    "explanations": ["why it was replaced"],
                }
            ],
        )
        alias = card.aliases[0]
        assert isinstance(alias, CardAlias)
        assert alias.key == "x-update"
        assert alias.description == "old wording"

    def test_alias_rejects_undeclared_keys(self):
        with pytest.raises(ValidationError):
            CardAlias(key="k", description="d", wrong=1)

    def test_alias_fields_are_described(self):
        undocumented = [
            name
            for name, field in CardAlias.model_fields.items()
            if not field.description
        ]
        assert not undocumented, undocumented


class TestTypedBestIdeas:
    def test_fold_stamps_typed_snapshot(self):
        row = IdeaStats(
            idea_id="i1",
            quartile=Quartile.ALL,
            description="seed desc",
            IntroGain_best_median=0.004,
        )
        folded = fold_best_idea_metrics(MemoryCard(id="i1"), row)
        snapshot = folded.evolution_statistics.best_ideas_snapshot
        assert isinstance(snapshot, CardStatsBlock)
        assert snapshot.IntroGain_best_median == 0.004
        assert folded.description == "seed desc"

    def test_parse_best_ideas_returns_idea_stats_rows(self, tmp_path):
        row = IdeaStats(
            idea_id="i1",
            quartile=Quartile.ALL,
            intro_events=3,
            IntroGain_best_median=float("nan"),
        )
        path = tmp_path / "best_ideas.json"
        path.write_text(
            json.dumps([{"timestamp": "t", "best_ideas": [row.as_json_row()]}]),
            encoding="utf-8",
        )
        idea_ids, rows_by_id = parse_best_ideas(path)
        assert idea_ids == ["i1"]
        parsed = rows_by_id["i1"]
        assert isinstance(parsed, IdeaStats)
        assert parsed.intro_events == 3
        assert parsed.IntroGain_best_median is None


class TestNoDictPlumbingInCardPaths:
    """Source-scan guard: the card decision paths must not re-grow Mapping/
    getattr dual-path fetches (prior directive: typed fields, mypy-checkable)."""

    CARD_PATH_FILES = [
        "gigaevo/memory/core/reputation.py",
        "gigaevo/memory/core/evictor.py",
        "gigaevo/memory/core/renderer.py",
        "gigaevo/memory/core/deduplicator.py",
        "gigaevo/memory/core/write_pipeline.py",
        "gigaevo/memory/core/read_pipeline.py",
        "gigaevo/memory/core/protocols.py",
        "gigaevo/memory/shared_memory/card_search.py",
        "gigaevo/memory/shared_memory/card_conversion.py",
        "gigaevo/memory/shared_memory/injection_posterior.py",
        "gigaevo/memory/shared_memory/card_loader.py",
        "gigaevo/memory/shared_memory/amem_gam_retriever.py",
        "gigaevo/memory/ideas_tracker/models.py",
        "gigaevo/memory/ideas_tracker/idea_bank.py",
        "gigaevo/memory/ideas_tracker/analyzers.py",
        "gigaevo/memory/write_pipeline.py",
    ]
    FORBIDDEN = (
        r"getattr\(card",
        r"getattr\(normalized",
        r"isinstance\(card, Mapping\)",
        r"isinstance\(card, dict\)",
        r"\bcard\.get\(",
        r"evolution_statistics\.get\(",
        r"\.get\(\"",
    )

    def test_no_dual_path_card_field_fetches(self):
        repo_root = Path(__file__).resolve().parents[2]
        offenders = []
        for rel in self.CARD_PATH_FILES:
            text = (repo_root / rel).read_text(encoding="utf-8")
            offenders.extend(
                f"{rel}: {pattern}"
                for pattern in self.FORBIDDEN
                if re.search(pattern, text)
            )
        assert not offenders, f"dict plumbing crept back into card paths: {offenders}"
