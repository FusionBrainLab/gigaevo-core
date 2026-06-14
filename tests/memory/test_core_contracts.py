"""Structural contracts for gigaevo.memory.core: Protocol conformance, Hydra
config-group instantiation, and the no-hardcoded-constants guard (redesign §6.3/6.5/6.7).
"""

from __future__ import annotations

import ast
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest

import gigaevo.memory.core as core
from gigaevo.memory.core.admitter import PermissiveAdmitter, SignBasedAdmitter
from gigaevo.memory.core.auctioneer import ThompsonAuctioneer
from gigaevo.memory.core.budgeter import TopThetaBudgeter
from gigaevo.memory.core.card_selector import LLMCardSelector
from gigaevo.memory.core.deduplicator import LLMDeduplicator, NullDeduplicator
from gigaevo.memory.core.evictor import HarmEvictor
from gigaevo.memory.core.protocols import (
    Auctioneer,
    Budgeter,
    CardRenderer,
    CardRetriever,
    CardShortlister,
    Deduplicator,
    Evictor,
    MemoryAdmitter,
    ReputationModel,
)
from gigaevo.memory.core.renderer import EfficacyCardRenderer
from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.core.retriever import GamRetriever

REPO_ROOT = Path(core.__file__).resolve().parents[3]
CONFIG_MEMORY = REPO_ROOT / "config" / "memory"


class TestProtocolConformance:
    @pytest.mark.parametrize("impl", [SignBasedAdmitter(), PermissiveAdmitter()])
    def test_admitters(self, impl):
        assert isinstance(impl, MemoryAdmitter)

    def test_reputation(self):
        assert isinstance(BetaBinomialReputation(), ReputationModel)

    def test_auctioneer(self):
        assert isinstance(ThompsonAuctioneer(), Auctioneer)

    def test_retriever(self):
        assert isinstance(GamRetriever(backend=None), CardRetriever)

    def test_shortlister(self):
        assert isinstance(LLMCardSelector(), CardShortlister)

    def test_budgeter(self):
        assert isinstance(TopThetaBudgeter(), Budgeter)

    def test_renderer(self):
        assert isinstance(EfficacyCardRenderer(), CardRenderer)

    @pytest.mark.parametrize("impl", [NullDeduplicator(), LLMDeduplicator()])
    def test_deduplicators(self, impl):
        assert isinstance(impl, Deduplicator)

    def test_evictor(self):
        assert isinstance(HarmEvictor(), Evictor)


class TestHydraComposition:
    def _load(self, *parts: str):
        return OmegaConf.load(CONFIG_MEMORY.joinpath(*parts))

    def test_reputation_group(self):
        obj = instantiate(self._load("reputation", "beta_binomial.yaml"))
        assert obj == BetaBinomialReputation()

    @pytest.mark.parametrize(
        ("leaf", "cls"),
        [
            ("sign_based.yaml", SignBasedAdmitter),
        ],
    )
    def test_admitter_group(self, leaf, cls):
        # _partial_ leaf: MemorySystem completes it with the shared reputation.
        rep = instantiate(self._load("reputation", "beta_binomial.yaml"))
        obj = instantiate(self._load("admitter", leaf))(reputation=rep)
        assert isinstance(obj, cls)
        assert obj.reputation == BetaBinomialReputation()

    def test_auction_group(self):
        obj = instantiate(self._load("auction", "thompson.yaml"))
        assert isinstance(obj, ThompsonAuctioneer)
        assert tuple(obj.baseline_prior) == (3.0, 3.0)

    def test_selector_group(self):
        obj = instantiate(self._load("selector", "llm.yaml"))
        assert isinstance(obj, LLMCardSelector)

    def test_budget_group(self):
        obj = instantiate(self._load("budget", "top_theta.yaml"))
        assert isinstance(obj, TopThetaBudgeter)

    def test_dedup_none_group(self):
        obj = instantiate(self._load("dedup", "none.yaml"))
        assert isinstance(obj, NullDeduplicator)

    def test_dedup_llm_group(self):
        obj = instantiate(self._load("dedup", "llm.yaml"))
        assert isinstance(obj, LLMDeduplicator)
        cfg = obj.config
        assert cfg.enabled is True
        assert cfg.top_k_per_query == 10
        assert cfg.final_top_n == 10
        assert cfg.min_final_score == 0.05
        assert cfg.llm_max_retries == 2
        assert cfg.weights.description == 0.35
        assert cfg.weights.explanation_summary == 0.2
        assert cfg.weights.description_explanation_summary == 0.3
        assert cfg.weights.description_task_description_summary == 0.15

    def test_evictor_harm_group(self):
        # _partial_ leaf: MemorySystem completes it with the shared reputation.
        rep = instantiate(self._load("reputation", "beta_binomial.yaml"))
        obj = instantiate(self._load("evictor", "harm.yaml"))(reputation=rep)
        assert isinstance(obj, HarmEvictor)
        assert obj.reputation == BetaBinomialReputation()

    def test_retriever_gam_group(self):
        obj = instantiate(self._load("retriever", "gam.yaml"))
        assert isinstance(obj, GamRetriever)
        assert obj.enable_bm25 is False
        assert obj.max_iters == 3
        assert list(obj.allowed_tools) == ["page_index", "vector"]
        assert isinstance(obj.top_k_by_tool, dict)
        # keyword is a dead tool (not in allowed_tools); its top_k entry was
        # dropped so the config carries no budget for a path that never runs.
        assert "keyword" not in obj.top_k_by_tool
        assert obj.top_k_by_tool["vector"] == 3
        assert obj.top_k_by_tool["page_index"] == 5
        assert obj.top_k_by_tool["vector_task_description"] == 0


class TestNoHardcodedConstants:
    """Decision expressions in the core modules must reference dataclass fields,
    never literals. Allowed inside function bodies: 0/1 identity values, the
    structural event-count tiers (2, 3), and sort sentinels (99, 1e18). Real
    thresholds (0.01, 0.4, 0.8, 0.2, 15, prior 3.0, ...) may appear only as
    dataclass field defaults or module-level named constants."""

    ALLOWED = {0, 1, 2, 3, 99, 1e18, 0.0, 1.0}

    @pytest.mark.parametrize(
        "module",
        [
            "reputation.py",
            "admitter.py",
            "auctioneer.py",
            "budgeter.py",
            "card_selector.py",
            "read_pipeline.py",
            "renderer.py",
            "retriever.py",
            "deduplicator.py",
            "evictor.py",
            "write_pipeline.py",
        ],
    )
    def test_no_literal_thresholds_in_function_bodies(self, module):
        source = (Path(core.__file__).parent / module).read_text()
        tree = ast.parse(source)
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Constant)
                    and isinstance(sub.value, (int, float))
                    and not isinstance(sub.value, bool)
                    and sub.value not in self.ALLOWED
                ):
                    offenders.append((module, sub.lineno, sub.value))
        assert not offenders, f"hardcoded numerics in decision code: {offenders}"
