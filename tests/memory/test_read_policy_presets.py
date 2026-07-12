"""Public memory/read_policy presets compose coherent bandit stacks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "config"
_BASE = ["problem.name=_test_"]


def _compose(*overrides: str) -> Any:
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(_CONFIG_DIR.absolute()), version_base=None
    ):
        return compose(config_name="config", overrides=[*_BASE, *overrides])


def _contains_behavior_space_ref(node: Any) -> bool:
    if isinstance(node, str):
        return "${ref:behavior_space}" in node
    if isinstance(node, dict):
        return any(_contains_behavior_space_ref(value) for value in node.values())
    if isinstance(node, list):
        return any(_contains_behavior_space_ref(value) for value in node)
    return False


def _cold_priors(node: Any) -> list[Any]:
    if isinstance(node, dict):
        values = []
        if "cold_prior" in node:
            values.append(node["cold_prior"])
        for value in node.values():
            values.extend(_cold_priors(value))
        return values
    if isinstance(node, list):
        values = []
        for value in node:
            values.extend(_cold_priors(value))
        return values
    return []


def test_full_defaults_to_adaptive_contextual_bootstrap_with_lineage():
    cfg = _compose("memory=full")
    memory = cfg.memory
    raw_memory = OmegaConf.to_container(memory, resolve=False)

    assert (
        memory.reputation._target_
        == "gigaevo.memory.read.reputation.BootstrapReputation"
    )
    assert (
        memory.reputation.inner._target_
        == "gigaevo.memory.read.reputation.BDProximityReputation"
    )
    assert (
        memory.auction._target_
        == "gigaevo.memory.read.auction.BootstrapThompsonAuctioneer"
    )
    assert memory.budget._target_ == "gigaevo.memory.read.auction.TopBidBudgeter"
    assert (
        memory.reader.shortlister._target_
        == "gigaevo.memory.read.fused.BootstrapFusedRankingShortlister"
    )
    assert (
        memory.context_model._target_
        == "gigaevo.memory.context.models.BDCellMemoryContext"
    )
    assert (
        memory.prior._target_ == "gigaevo.memory.read.prior.EmpiricalBayesMemoryPrior"
    )
    assert (
        memory.no_card_evidence._target_
        == "gigaevo.memory.context.no_card.JsonNoCardEvidenceStore"
    )
    assert memory.probe_policy._target_ == "gigaevo.memory.read.probe.ColdProbePolicy"
    assert (
        memory.reader.candidate_projector._target_
        == "gigaevo.memory.read.projection.AuctionCandidateProjector"
    )
    assert memory.excluder._target_ == "gigaevo.memory.read.exclusion.LineageExcluder"
    assert memory.auction.ev_floor_quantile == pytest.approx(0.765)
    assert memory.evictor._target_ == "gigaevo.memory.write.eviction.CompositeEvictor"
    assert (
        memory.evictor.evictors[0]._target_
        == "gigaevo.memory.write.eviction.BirthFailureEvictor"
    )
    assert (
        memory.evictor.evictors[1]._target_
        == "gigaevo.memory.write.eviction.HarmEvictor"
    )
    assert (
        memory.evictor.evictors[2]._target_
        == "gigaevo.memory.write.eviction.PolicyNonViableEvictor"
    )
    assert raw_memory["writer"]["baseline_estimator"] == "${ref:memory.context_model}"
    assert raw_memory["writer"]["no_card_recorder"] == "${ref:memory.no_card_evidence}"
    assert memory.neutral_gain == pytest.approx(0.0)
    assert list(memory.baseline_prior) == [3.0, 3.0]
    assert list(memory.auction.baseline_prior) == list(memory.baseline_prior)
    assert list(memory.no_card_evidence.seed_prior) == list(memory.baseline_prior)
    assert memory.evictor.evictors[2].neutral_gain == pytest.approx(0.0)
    assert memory.evidence.min_effective_events == 3
    assert memory.eviction_safety.min_effective_events == 3
    assert memory.reputation.confident_min_events == 3
    assert memory.reputation.inner.harm_min_events == 3
    assert memory.evictor.evictors[1].skip_contextual_without_context is True
    assert memory.evictor.evictors[2].min_effective_events == 3
    assert memory.evictor.evictors[2].skip_contextual_without_context is True


def test_ev_floor_quantile_is_bootstrap_policy_local():
    recommended = _compose("memory=full")
    median_legacy = _compose("memory=full", "memory/read_policy=median_ev_legacy")
    probability_legacy = _compose(
        "memory=full", "memory/read_policy=probability_legacy"
    )

    assert recommended.memory.auction.ev_floor_quantile == pytest.approx(0.765)
    assert "ev_floor_quantile" not in median_legacy.memory.auction
    assert "ev_floor_quantile" not in probability_legacy.memory.auction


def test_reader_defaults_to_portable_bootstrap_with_lineage():
    cfg = _compose("memory=reader")
    memory = cfg.memory
    raw_reputation = OmegaConf.to_container(memory.reputation, resolve=False)

    assert (
        memory.reputation._target_
        == "gigaevo.memory.read.reputation.BootstrapReputation"
    )
    assert (
        memory.reputation.inner._target_
        == "gigaevo.memory.read.reputation.BetaBinomialReputation"
    )
    assert (
        memory.context_model._target_
        == "gigaevo.memory.context.models.GlobalMemoryContext"
    )
    assert memory.probe_policy._target_ == "gigaevo.memory.read.probe.ColdProbePolicy"
    assert not _contains_behavior_space_ref(raw_reputation)
    assert memory.excluder._target_ == "gigaevo.memory.read.exclusion.LineageExcluder"


def test_writer_default_evictor_uses_memory_neutral_gain():
    cfg = _compose("memory=writer")
    memory = cfg.memory
    raw_memory = OmegaConf.to_container(memory, resolve=False)

    assert memory.neutral_gain == pytest.approx(0.0)
    assert (
        memory.evictor.evictors[2]._target_
        == "gigaevo.memory.write.eviction.PolicyNonViableEvictor"
    )
    assert (
        memory.context_model._target_
        == "gigaevo.memory.context.models.GlobalMemoryContext"
    )
    assert (
        memory.no_card_evidence._target_
        == "gigaevo.memory.context.no_card.JsonNoCardEvidenceStore"
    )
    assert raw_memory["writer"]["baseline_estimator"] == "${ref:memory.context_model}"
    assert memory.evictor.evictors[2].neutral_gain == pytest.approx(memory.neutral_gain)
    assert list(memory.baseline_prior) == [3.0, 3.0]
    assert list(memory.no_card_evidence.seed_prior) == list(memory.baseline_prior)
    assert memory.evidence.min_effective_events == 3
    assert memory.eviction_safety.min_effective_events == 3
    assert memory.reputation.harm_min_events == 3
    assert memory.evictor.evictors[1].skip_contextual_without_context is True
    assert memory.evictor.evictors[2].min_effective_events == 3
    assert memory.evictor.evictors[2].skip_contextual_without_context is True


def test_portable_policy_contains_no_behavior_space_ref():
    cfg = _compose("memory=full", "memory/read_policy=portable")
    raw_reputation = OmegaConf.to_container(cfg.memory.reputation, resolve=False)
    assert not _contains_behavior_space_ref(raw_reputation)


@pytest.mark.parametrize(
    ("policy", "reputation", "auction", "budget", "shortlister"),
    [
        (
            "adaptive",
            "gigaevo.memory.read.reputation.BootstrapReputation",
            "gigaevo.memory.read.auction.BootstrapThompsonAuctioneer",
            "gigaevo.memory.read.auction.TopBidBudgeter",
            "gigaevo.memory.read.fused.BootstrapFusedRankingShortlister",
        ),
        (
            "portable",
            "gigaevo.memory.read.reputation.BootstrapReputation",
            "gigaevo.memory.read.auction.BootstrapThompsonAuctioneer",
            "gigaevo.memory.read.auction.TopBidBudgeter",
            "gigaevo.memory.read.fused.BootstrapFusedRankingShortlister",
        ),
        (
            "median_ev_legacy",
            "gigaevo.memory.read.reputation.BDProximityReputation",
            "gigaevo.memory.read.auction.EVThompsonAuctioneer",
            "gigaevo.memory.read.auction.TopBidBudgeter",
            "gigaevo.memory.read.fused.FusedRankingShortlister",
        ),
        (
            "probability_legacy",
            "gigaevo.memory.read.reputation.BetaBinomialReputation",
            "gigaevo.memory.read.auction.ThompsonAuctioneer",
            "gigaevo.memory.read.auction.TopThetaBudgeter",
            "gigaevo.memory.read.fused.FusedRankingShortlister",
        ),
    ],
)
def test_read_policy_core_pairings(policy, reputation, auction, budget, shortlister):
    cfg = _compose("memory=full", f"memory/read_policy={policy}")
    memory = cfg.memory

    assert memory.reputation._target_ == reputation
    assert memory.auction._target_ == auction
    assert memory.budget._target_ == budget
    assert memory.reader.shortlister._target_ == shortlister
    assert memory.excluder._target_ == "gigaevo.memory.read.exclusion.LineageExcluder"


@pytest.mark.parametrize(
    "policy",
    [
        "adaptive",
        "portable",
        "median_ev_legacy",
        "probability_legacy",
        "contextual_bootstrap_decay",
        "portable_bootstrap_decay",
        "decay_median_ev_legacy",
    ],
)
def test_read_policy_cold_priors_match_auction_baseline(policy):
    cfg = _compose("memory=full", f"memory/read_policy={policy}")
    raw_reputation = OmegaConf.to_container(cfg.memory.reputation, resolve=False)
    for cold_prior in _cold_priors(raw_reputation):
        assert cold_prior == "${memory.baseline_prior}"
    assert list(cfg.memory.auction.baseline_prior) == list(cfg.memory.baseline_prior)


def test_novelty_auction_override_composes_on_full():
    # `+` because the auction group enters through read_policy's nested defaults.
    cfg = _compose("memory=full", "+memory/auction=thompson_bootstrap_novelty")
    memory = cfg.memory

    assert (
        memory.auction._target_
        == "gigaevo.memory.read.auction.NoveltyDiscountedBootstrapAuctioneer"
    )
    assert memory.auction.novelty_power == pytest.approx(0.5)
    assert memory.auction.ev_floor_quantile == pytest.approx(0.765)
    # The rest of the read stack is untouched by the auction swap.
    assert memory.budget._target_ == "gigaevo.memory.read.auction.TopBidBudgeter"
    assert (
        memory.reader.candidate_projector._target_
        == "gigaevo.memory.read.projection.AuctionCandidateProjector"
    )
