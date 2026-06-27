"""Typed card-stats contract: cards carry only raw gain events, reputation
computes every CardStatsBlock from them at read time, and every consumer reads
typed fields — no Mapping/getattr dual paths anywhere in the card decision paths."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
from pydantic import ValidationError
import pytest

from gigaevo.memory.context import ContextualGain, DecisionContext
from gigaevo.memory.core.auctioneer import (
    AuctionBid,
    AuctionCandidate,
    ThompsonAuctioneer,
)
from gigaevo.memory.core.budgeter import TopThetaBudgeter
from gigaevo.memory.core.evictor import HarmEvictor
from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.core.selection import MemorySelection
from gigaevo.memory.shared_memory.card_search import format_card_efficacy
from gigaevo.memory.shared_memory.models import (
    CardStatsBlock,
    DecisionMetrics,
    MemoryCard,
    ProgramCard,
)


def _events(gains: list[float]) -> list[ContextualGain]:
    return [
        ContextualGain(
            context=DecisionContext(parent_metrics={"min_area": 0.5}), gain=g
        )
        for g in gains
    ]


# Six wins and one loss: the MAD noise band collapses to 0, so exactly k_harm = 1
# of n = 7 events fall below threshold -> Beta(7, 2) downside posterior, +0.01
# median magnitude, confidently helpful.
CONFIDENT_EVENTS = _events([0.01] * 6 + [-0.5])


class TestStatsVocabulary:
    def test_metric_vocabulary_is_first_class_and_described(self):
        decision = set(DecisionMetrics.model_fields)
        assert set(CardStatsBlock.model_fields) == decision
        undescribed = [
            f"{model.__name__}.{name}"
            for model in (DecisionMetrics, CardStatsBlock)
            for name, field in model.model_fields.items()
            if not field.description
        ]
        assert not undescribed

    def test_stats_blocks_reject_undeclared_keys(self):
        assert CardStatsBlock.model_config["extra"] == "forbid"
        with pytest.raises(ValidationError):
            CardStatsBlock(SiblingWinRate=0.5)


class TestReputationTypedReads:
    def test_card_posterior_computes_from_gain_events(self):
        rep = BetaBinomialReputation()
        card = MemoryCard(id="idea-1", gain_events=CONFIDENT_EVENTS)
        assert rep.card_posterior(card) == (7.0, 2.0)

    def test_card_posterior_cold_prior_without_events(self):
        rep = BetaBinomialReputation()
        assert rep.card_posterior(MemoryCard(id="idea-2")) == rep.cold_prior

    def test_card_posterior_cold_prior_on_empty_events(self):
        rep = BetaBinomialReputation()
        assert rep.card_posterior(MemoryCard(id="idea-3", gain_events=[])) == (
            rep.cold_prior
        )

    def test_is_confidently_harmful_takes_typed_statistics(self):
        rep = BetaBinomialReputation()
        harmful = CardStatsBlock(posterior_a=1.0, posterior_b=9.0, intro_events=8)
        helpful = CardStatsBlock(posterior_a=9.0, posterior_b=1.0, intro_events=8)
        thin = CardStatsBlock(posterior_a=1.0, posterior_b=9.0, intro_events=2)
        assert rep.is_confidently_harmful(harmful) is True
        assert rep.is_confidently_harmful(helpful) is False
        assert rep.is_confidently_harmful(thin) is False
        assert rep.is_confidently_harmful(CardStatsBlock()) is False
        assert rep.is_confidently_harmful(None) is False


class TestEvictorTypedReads:
    def test_should_evict_reads_card_statistics(self):
        evictor = HarmEvictor(reputation=BetaBinomialReputation())
        harmful = MemoryCard(id="bad", gain_events=_events([-0.5] * 8))
        assert evictor.should_evict(harmful) is True
        assert evictor.should_evict(MemoryCard(id="cold")) is False
        assert evictor.sweep({"bad": harmful, "cold": MemoryCard(id="cold")}) == ["bad"]


class TestEfficacyRenderingTypedReads:
    def test_confident_memory_card_renders_endorsement(self):
        card = MemoryCard(id="idea-1", gain_events=CONFIDENT_EVENTS)
        line = format_card_efficacy(card)
        assert line is not None
        assert "introduced in 7 children" in line
        assert "median improvement +0.0100" in line
        assert "(confident)" in line
        assert "vs cohort" not in line

    def test_non_confident_card_stays_silent(self):
        # A single positive event: median > 0 but the pessimistic lower bound on
        # P(help) stays under the threshold, so the card stays silent.
        card = MemoryCard(id="idea-2", gain_events=_events([0.01]))
        assert format_card_efficacy(card) is None

    def test_program_card_renders_exemplar_fitness(self):
        card = ProgramCard(id="program-p1", program_id="p1", fitness=0.8712)
        assert format_card_efficacy(card) == "efficacy: exemplar fitness 0.8712"
        assert (
            format_card_efficacy(ProgramCard(id="program-p2", program_id="p2")) is None
        )


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


class TestNoDictPlumbingInCardPaths:
    """Source-scan guard: the card decision paths must not re-grow Mapping/
    getattr dual-path fetches (prior directive: typed fields, mypy-checkable)."""

    CARD_PATH_FILES = [
        "gigaevo/memory/core/reputation.py",
        "gigaevo/memory/core/evictor.py",
        "gigaevo/memory/core/renderer.py",
        "gigaevo/memory/core/read_pipeline.py",
        "gigaevo/memory/core/protocols.py",
        "gigaevo/memory/shared_memory/card_search.py",
        "gigaevo/memory/shared_memory/card_conversion.py",
        "gigaevo/memory/shared_memory/injection_posterior.py",
        "gigaevo/memory/shared_memory/amem_gam_retriever.py",
        "gigaevo/memory/ideas_tracker/models.py",
        "gigaevo/memory/efficacy/stamping.py",
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
