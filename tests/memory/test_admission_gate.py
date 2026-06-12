"""Posterior-aware admission gate in ``save_card``.

A card whose injection posterior is confidently harmful (>=3 intro events and
even the optimistic 80th-percentile read of P(not harmful) below 0.5) must not
enter the card store; a resident card whose refreshed posterior turns
confidently harmful on re-save is evicted. Thin or healthy evidence admits as
before (the auction treats thin cards as near-cold).
"""

from __future__ import annotations

from gigaevo.memory.efficacy import beta_binomial_posterior
from tests.fakes.agentic_memory import make_test_memory


def _harmful_stats() -> dict:
    return {"ALL": beta_binomial_posterior([-0.01, -0.02, -0.03, -0.04])}


def _healthy_stats() -> dict:
    return {"ALL": beta_binomial_posterior([0.01, 0.02, 0.03, 0.04])}


class TestAdmissionGate:
    def test_confidently_harmful_new_card_rejected(self, tmp_path) -> None:
        mem = make_test_memory(tmp_path)
        before = mem.card_store.write_stats.get("rejected", 0)
        cid = mem.save_card(
            {
                "id": "bad-1",
                "description": "harmful lever",
                "evolution_statistics": _harmful_stats(),
            }
        )
        assert cid == "bad-1"
        assert "bad-1" not in mem.card_store.cards
        assert mem.card_store.write_stats["rejected"] == before + 1

    def test_healthy_card_admitted(self, tmp_path) -> None:
        mem = make_test_memory(tmp_path)
        mem.save_card(
            {
                "id": "good-1",
                "description": "useful lever",
                "evolution_statistics": _healthy_stats(),
            }
        )
        assert "good-1" in mem.card_store.cards

    def test_thin_all_harm_evidence_still_admitted(self, tmp_path) -> None:
        mem = make_test_memory(tmp_path)
        mem.save_card(
            {
                "id": "thin-1",
                "description": "unproven lever",
                "evolution_statistics": {
                    "ALL": beta_binomial_posterior([-0.01, -0.02])
                },
            }
        )
        assert "thin-1" in mem.card_store.cards

    def test_card_without_stats_admitted(self, tmp_path) -> None:
        mem = make_test_memory(tmp_path)
        mem.save_card({"id": "cold-1", "description": "fresh idea"})
        assert "cold-1" in mem.card_store.cards

    def test_resident_card_evicted_when_posterior_turns_harmful(self, tmp_path) -> None:
        mem = make_test_memory(tmp_path)
        mem.save_card(
            {
                "id": "idea-1",
                "description": "lever",
                "evolution_statistics": _healthy_stats(),
            }
        )
        assert "idea-1" in mem.card_store.cards
        mem.save_card(
            {
                "id": "idea-1",
                "description": "lever",
                "evolution_statistics": _harmful_stats(),
            }
        )
        assert "idea-1" not in mem.card_store.cards

    def test_program_card_with_harmful_posterior_rejected(self, tmp_path) -> None:
        # ProgramCards bypass dedup but must not bypass the gate.
        mem = make_test_memory(tmp_path)
        mem.save_card(
            {
                "id": "program-abc",
                "program_id": "abc",
                "category": "program",
                "description": "exemplar",
                "fitness": 0.5,
                "evolution_statistics": _harmful_stats(),
            }
        )
        assert "program-abc" not in mem.card_store.cards
