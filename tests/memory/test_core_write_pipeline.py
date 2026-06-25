"""Phase 3 write-pipeline equivalence: HarmEvictor vs the legacy harm gate,
deduplicator seams, and MemoryWritePipeline.ingest as a behavioral twin of
AmemGamMemory.save_card (redesign §2/§6.1/§6.4).
"""

from __future__ import annotations

from typing import Any

import pytest

from gigaevo.memory.context import ContextualGain, DecisionContext
from gigaevo.memory.core.deduplicator import LLMDeduplicator, NullDeduplicator
from gigaevo.memory.core.evictor import HarmEvictor
from gigaevo.memory.core.reputation import BetaBinomialReputation
from gigaevo.memory.core.write_pipeline import MemoryWritePipeline
from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card
from gigaevo.memory.shared_memory.card_dedup import DedupAction, DedupDecision
from gigaevo.memory.shared_memory.card_update_dedup import CardUpdateDedupConfig
from tests.fakes.agentic_memory import make_test_memory


def _events(gains: list[float]) -> list[dict[str, Any]]:
    return [
        ContextualGain(
            context=DecisionContext(parent_metrics={"min_area": 0.5}), gain=g
        ).model_dump()
        for g in gains
    ]


# Equal-sign gains collapse the MAD noise band to 0, so every loss falls below
# threshold: six losses -> Beta(1, 7) k_harm=6 (confidently harmful); six wins ->
# Beta(7, 1) (proven); two losses -> Beta(1, 3) intro=2 (too thin to evict).
_HARMFUL_EVENTS = _events([-0.5] * 6)
_PROVEN_EVENTS = _events([0.01] * 6)
_THIN_EVENTS = _events([-0.5] * 2)


def _idea(
    card_id: str, description: str, events: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    card: dict[str, Any] = {
        "id": card_id,
        "category": "general",
        "description": description,
    }
    if events is not None:
        card["gain_events"] = events
    return card


def _program(program_id: str) -> dict[str, Any]:
    return {
        "id": f"program-{program_id}",
        "category": "program",
        "program_id": program_id,
        "description": "exemplar program",
        "fitness": 0.5,
        "code": "def fit_predict(): ...",
    }


def _add_decision(reason: str = "stub") -> DedupDecision:
    return DedupDecision(action="add", reason=reason, duplicate_of="", merges=[])


class _StubDedup:
    def __init__(self, decision: DedupDecision) -> None:
        self.decision = decision
        self.calls: list[str] = []

    def reconcile(self, card: Any, bank: Any) -> DedupDecision:
        self.calls.append(str(card.id))
        if not bank:
            return _add_decision("empty bank")
        return self.decision


class _StubEngine:
    def __init__(
        self,
        *,
        enabled: bool = True,
        llm_service: Any = None,
        decision: DedupDecision | None = None,
    ) -> None:
        self.config = CardUpdateDedupConfig(enabled=enabled)
        self.llm_service = llm_service
        self.decision = decision or _add_decision("engine")
        self.calls: list[str] = []

    def run_dedup_on_incoming_card(self, card: Any) -> DedupDecision:
        self.calls.append(str(card.id))
        return self.decision


class TestHarmEvictor:
    @pytest.mark.parametrize(
        "events",
        [
            _HARMFUL_EVENTS,
            _PROVEN_EVENTS,
            _THIN_EVENTS,
            [],
            None,
        ],
    )
    def test_should_evict_matches_reputation_predicate(self, events):
        evictor = HarmEvictor()
        card = normalize_memory_card(_idea("idea-x", "desc", events))
        rep = BetaBinomialReputation()
        expected = rep.is_confidently_harmful(rep.card_stats(card, None))
        assert evictor.should_evict(card) is expected

    def test_reputation_is_injectable(self):
        lenient = HarmEvictor(reputation=BetaBinomialReputation(harm_min_events=99))
        card = normalize_memory_card(_idea("idea-x", "d", _HARMFUL_EVENTS))
        assert lenient.should_evict(card) is False

    def test_sweep_returns_only_harmful_ids(self):
        bank = {
            "idea-good": normalize_memory_card(_idea("idea-good", "g", _PROVEN_EVENTS)),
            "idea-bad": normalize_memory_card(_idea("idea-bad", "b", _HARMFUL_EVENTS)),
            "idea-cold": normalize_memory_card(_idea("idea-cold", "c")),
        }
        assert HarmEvictor().sweep(bank) == ["idea-bad"]

    def test_sweep_empty_bank(self):
        assert HarmEvictor().sweep({}) == []


class TestDeduplicators:
    _CARD = normalize_memory_card(_idea("idea-new", "candidate"))
    _BANK = {"idea-1": normalize_memory_card(_idea("idea-1", "existing"))}

    def test_null_always_add(self):
        decision = NullDeduplicator().reconcile(self._CARD, self._BANK)
        assert decision.action is DedupAction.ADD
        assert decision.merges == []

    def test_llm_without_engine_adds(self):
        assert (
            LLMDeduplicator().reconcile(self._CARD, self._BANK).action
            is DedupAction.ADD
        )

    def test_llm_disabled_engine_adds(self):
        engine = _StubEngine(enabled=False, llm_service=object())
        dedup = LLMDeduplicator(engine=engine)
        assert dedup.reconcile(self._CARD, self._BANK).action is DedupAction.ADD
        assert engine.calls == []

    def test_llm_empty_bank_adds(self):
        engine = _StubEngine(llm_service=object())
        dedup = LLMDeduplicator(engine=engine)
        assert dedup.reconcile(self._CARD, {}).action is DedupAction.ADD
        assert engine.calls == []

    def test_llm_missing_llm_adds(self):
        engine = _StubEngine(llm_service=None)
        dedup = LLMDeduplicator(engine=engine)
        assert dedup.reconcile(self._CARD, self._BANK).action is DedupAction.ADD
        assert dedup.reconcile(self._CARD, self._BANK).action is DedupAction.ADD
        assert engine.calls == []

    def test_llm_delegates_to_ready_engine(self):
        sentinel = DedupDecision(
            action="discard", reason="dup", duplicate_of="idea-1", merges=[]
        )
        engine = _StubEngine(llm_service=object(), decision=sentinel)
        dedup = LLMDeduplicator(engine=engine)
        assert dedup.reconcile(self._CARD, self._BANK) is sentinel
        assert engine.calls == ["idea-new"]


def _twins(tmp_path, **overrides):
    legacy = make_test_memory(tmp_path / "legacy", **overrides)
    target = make_test_memory(tmp_path / "target", **overrides)
    return legacy, target


def _pipeline(mem, dedup=None) -> MemoryWritePipeline:
    return MemoryWritePipeline(
        store=mem,
        evictor=HarmEvictor(),
        deduplicator=dedup if dedup is not None else LLMDeduplicator(engine=mem.dedup),
    )


def _bank_dump(mem) -> dict[str, dict[str, Any]]:
    return {cid: card.model_dump() for cid, card in mem.card_store.cards.items()}


def _assert_twins(legacy, target) -> None:
    assert target.get_card_write_stats() == legacy.get_card_write_stats()
    assert _bank_dump(target) == _bank_dump(legacy)


def _run_both(legacy, pipeline, cards) -> None:
    for card in cards:
        assert pipeline.ingest(dict(card)) == legacy.save_card(dict(card))


class TestMemoryWritePipelineEquivalence:
    def test_add_update_program_and_harm_sequence(self, tmp_path):
        legacy, target = _twins(tmp_path)
        pipeline = _pipeline(target)
        _run_both(
            legacy,
            pipeline,
            [
                _idea("idea-1", "use gradient clipping"),
                _idea("idea-1", "use gradient clipping with max-norm"),
                _program("p1"),
                _idea("idea-bad", "harmful newcomer", _HARMFUL_EVENTS),
                _idea("idea-1", "went harmful", _HARMFUL_EVENTS),
            ],
        )
        _assert_twins(legacy, target)
        stats = target.get_card_write_stats()
        assert stats["processed"] == 5
        assert stats["rejected"] == 2
        assert "idea-1" not in target.card_store.cards
        assert "program-p1" in target.card_store.cards

    def _dedup_twins(self, tmp_path, decision: DedupDecision):
        legacy, target = _twins(tmp_path, card_update_dedup_config={"enabled": True})
        legacy.dedup.llm_service = object()
        legacy.dedup.run_dedup_on_incoming_card = lambda card: decision
        pipeline = _pipeline(target, dedup=_StubDedup(decision))
        seed = _idea("idea-1", "existing idea")
        assert pipeline.ingest(dict(seed)) == legacy.save_card(dict(seed))
        return legacy, target, pipeline

    def test_dedup_discard_known_duplicate(self, tmp_path):
        decision = DedupDecision(
            action="discard", reason="dup", duplicate_of="idea-1", merges=[]
        )
        legacy, target, pipeline = self._dedup_twins(tmp_path, decision)
        incoming = _idea("idea-2", "same idea reworded")
        assert pipeline.ingest(dict(incoming)) == legacy.save_card(dict(incoming))
        _assert_twins(legacy, target)
        assert target.get_card_write_stats()["rejected"] == 1
        assert "idea-2" not in target.card_store.cards

    def test_dedup_discard_phantom_duplicate(self, tmp_path):
        decision = DedupDecision(
            action="discard", reason="dup", duplicate_of="ghost", merges=[]
        )
        legacy, target, pipeline = self._dedup_twins(tmp_path, decision)
        incoming = _idea("idea-2", "same idea reworded")
        assert pipeline.ingest(dict(incoming)) == legacy.save_card(dict(incoming))
        _assert_twins(legacy, target)
        assert "idea-2" not in target.card_store.cards

    def test_dedup_update_with_merges(self, tmp_path):
        merged = normalize_memory_card(_idea("idea-1", "merged description"))
        decision = DedupDecision(
            action="update",
            reason="merge",
            duplicate_of="",
            merges=[("idea-1", merged)],
        )
        legacy, target, pipeline = self._dedup_twins(tmp_path, decision)
        incoming = _idea("idea-2", "extra details")
        assert pipeline.ingest(dict(incoming)) == legacy.save_card(dict(incoming))
        _assert_twins(legacy, target)
        stats = target.get_card_write_stats()
        assert stats["updated"] == 1
        assert stats["updated_target_cards"] == 1
        assert target.card_store.cards["idea-1"].description == "merged description"

    def test_dedup_update_empty_merges_falls_to_add(self, tmp_path):
        decision = DedupDecision(
            action="update", reason="merge", duplicate_of="", merges=[]
        )
        legacy, target, pipeline = self._dedup_twins(tmp_path, decision)
        incoming = _idea("idea-2", "extra details")
        assert pipeline.ingest(dict(incoming)) == legacy.save_card(dict(incoming))
        _assert_twins(legacy, target)
        assert "idea-2" in target.card_store.cards

    def test_dedup_disabled_adds_without_engine_call(self, tmp_path):
        legacy, target = _twins(tmp_path)
        pipeline = _pipeline(target)
        _run_both(
            legacy,
            pipeline,
            [_idea("idea-1", "first"), _idea("idea-2", "second")],
        )
        _assert_twins(legacy, target)
        assert target.get_card_write_stats()["added"] == 2


class TestSaveCardDelegation:
    def test_memory_exposes_wired_write_pipeline(self, tmp_path):
        mem = make_test_memory(tmp_path)
        assert isinstance(mem.write_pipeline, MemoryWritePipeline)
        assert isinstance(mem.write_pipeline._evictor, HarmEvictor)
        assert isinstance(mem.write_pipeline._dedup, LLMDeduplicator)
        assert mem.write_pipeline._dedup.engine is mem.dedup
        assert mem.write_pipeline._store is mem

    def test_save_card_delegates_to_pipeline(self, tmp_path):
        mem = make_test_memory(tmp_path)
        seen: list[Any] = []

        def fake_ingest(card):
            seen.append(card)
            return "sentinel-id"

        mem.write_pipeline.ingest = fake_ingest
        assert mem.save_card(_idea("idea-1", "via shim")) == "sentinel-id"
        assert len(seen) == 1

    def test_save_card_behavior_unchanged_through_shim(self, tmp_path):
        mem = make_test_memory(tmp_path)
        cid = mem.save_card(_idea("idea-1", "real ingest"))
        assert cid == "idea-1"
        assert mem.save_card(_idea("idea-bad", "harmful", _HARMFUL_EVENTS)) == ""
        stats = mem.get_card_write_stats()
        assert stats["processed"] == 2
        assert stats["added"] == 1
        assert stats["rejected"] == 1
        assert "idea-bad" not in mem.card_store.cards


class TestWritePipelineSweep:
    def test_sweep_evicts_harmful_cards_from_store(self, tmp_path):
        mem = make_test_memory(tmp_path)
        pipeline = _pipeline(mem)
        pipeline.ingest(_idea("idea-good", "keep me", _PROVEN_EVENTS))
        pipeline.ingest(_idea("idea-cold", "no stats yet"))
        mem.card_store.cards["idea-good"].gain_events = _HARMFUL_EVENTS
        assert pipeline.sweep() == ["idea-good"]
        assert "idea-good" not in mem.card_store.cards
        assert "idea-cold" in mem.card_store.cards

    def test_sweep_noop_on_healthy_bank(self, tmp_path):
        mem = make_test_memory(tmp_path)
        pipeline = _pipeline(mem)
        pipeline.ingest(_idea("idea-good", "keep me", _PROVEN_EVENTS))
        assert pipeline.sweep() == []
        assert "idea-good" in mem.card_store.cards


class TestWriteSideInjection:
    def test_injected_evictor_and_deduplicator_reach_write_pipeline(self, tmp_path):
        evictor = HarmEvictor(reputation=BetaBinomialReputation(harm_min_events=2))
        dedup = NullDeduplicator()
        mem = make_test_memory(tmp_path, evictor=evictor, deduplicator=dedup)
        assert mem.write_pipeline._evictor is evictor
        assert mem.write_pipeline._dedup is dedup

    def test_injected_llm_deduplicator_gets_engine_and_drives_engine_config(
        self, tmp_path
    ):
        cfg = CardUpdateDedupConfig(enabled=False)
        dedup = LLMDeduplicator(config=cfg)
        mem = make_test_memory(tmp_path, deduplicator=dedup)
        assert dedup.engine is mem.dedup
        assert mem.dedup.config is cfg

    def test_default_write_side_construction_unchanged(self, tmp_path):
        mem = make_test_memory(tmp_path)
        assert isinstance(mem.write_pipeline._evictor, HarmEvictor)
        assert isinstance(mem.write_pipeline._dedup, LLMDeduplicator)
        assert mem.write_pipeline._dedup.engine is mem.dedup
        assert mem.dedup.config is mem.config.dedup
