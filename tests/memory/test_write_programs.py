import json
from typing import Any

from pydantic import Field

from gigaevo.memory.backend_factory import MemoryBackendFactory
from gigaevo.memory.core import BetaBinomialReputation
from gigaevo.memory.core.deduplicator import NullDeduplicator
from gigaevo.memory.core.evictor import HarmEvictor
from gigaevo.memory.shared_memory.card_conversion import normalize_memory_card
from gigaevo.memory.shared_memory.injection_posterior import beta_binomial_posterior
from gigaevo.memory.shared_memory.models import ProgramCard
from gigaevo.memory.write_pipeline import load_memory_cards
from gigaevo.memory.write_pipeline import main as write_main
from tests.fakes.agentic_memory import make_test_memory


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _minimal_corpus(tmp_path):
    """banks / best_ideas / programs trio with prog-9 as the single top card."""
    banks_path = tmp_path / "banks.json"
    best_ideas_path = tmp_path / "best_ideas.json"
    programs_path = tmp_path / "programs.json"
    _write_json(
        banks_path,
        [{"timestamp": "t", "active_bank": [], "inactive_bank": []}],
    )
    _write_json(
        best_ideas_path,
        [{"timestamp": "t", "best_ideas": []}],
    )
    _write_json(
        programs_path,
        [
            {
                "timestamp": "t",
                "programs": [
                    {
                        "id": f"prog-{idx}",
                        "fitness": float(idx),
                        "is_valid": 1.0,
                        "code": f"def run_code():\n    return {idx}\n",
                    }
                    for idx in range(10)
                ],
            }
        ],
    )
    return banks_path, best_ideas_path, programs_path


class _StubWriteMemory:
    def __init__(self):
        self.swept = False

    def save_card(self, card):
        return "card-1"

    def get_card(self, card_id):
        return None

    def get_card_write_stats(self):
        return {}

    def rebuild(self):
        pass

    def sweep_harmful(self):
        self.swept = True
        return []

    def close(self):
        pass


class _RecordingFactory(MemoryBackendFactory):
    recorded: dict[str, Any] = Field(default_factory=dict)
    memory: Any = None

    def backend_class(self):
        return _StubWriteMemory

    def build(self, *, checkpoint_dir=None, gam=None, evictor=None, deduplicator=None):
        self.recorded.update(evictor=evictor, deduplicator=deduplicator)
        self.memory = _StubWriteMemory()
        return self.memory


def test_main_passes_components_to_write_backend_and_sweeps(tmp_path):
    """memory/evictor + memory/dedup groups must reach the WRITE backend
    (the one that ingests), and main() must sweep after the ingest loop."""
    banks_path, best_ideas_path, programs_path = _minimal_corpus(tmp_path)
    evictor = HarmEvictor()
    deduplicator = NullDeduplicator()
    factory = _RecordingFactory()

    snapshot = write_main(
        banks_path=banks_path,
        best_ideas_path=best_ideas_path,
        programs_path=programs_path,
        backend=factory,
        evictor=evictor,
        deduplicator=deduplicator,
    )

    assert snapshot is not None
    assert factory.recorded["evictor"] is evictor
    assert factory.recorded["deduplicator"] is deduplicator
    assert factory.memory.swept


def test_memory_facade_sweep_harmful_empty_bank(tmp_path):
    memory = make_test_memory(tmp_path)
    try:
        assert memory.sweep_harmful() == []
    finally:
        memory.close()


def test_load_memory_cards_adds_top_program_cards(tmp_path):
    banks_path = tmp_path / "banks.json"
    best_ideas_path = tmp_path / "best_ideas.json"
    programs_path = tmp_path / "programs.json"

    _write_json(
        banks_path,
        [
            {
                "timestamp": "2026-03-23 00:00:00",
                "active_bank": [
                    {
                        "id": "idea-1",
                        "description": "Use simulated annealing for local refinement.",
                        "task_description": "Solve the task.",
                        "task_description_summary": "Solve the task efficiently.",
                        "programs": ["prog-1", "prog-9"],
                    },
                    {
                        "id": "idea-2",
                        "description": "Add a boundary-aware repair step.",
                        "task_description": "Solve the task.",
                        "task_description_summary": "Solve the task efficiently.",
                        "programs": ["prog-9"],
                    },
                ],
                "inactive_bank": [],
            }
        ],
    )
    _write_json(
        best_ideas_path,
        [
            {
                "timestamp": "2026-03-23 00:00:00",
                "best_ideas": [
                    {"idea_id": "idea-1", "description": "Use simulated annealing."}
                ],
            }
        ],
    )
    _write_json(
        programs_path,
        [
            {
                "timestamp": "2026-03-23 00:00:00",
                "programs": [
                    {
                        "id": f"prog-{idx}",
                        "fitness": float(idx),
                        "is_valid": 1.0,
                        "generation": idx,
                        "strategy": "hybrid",
                        "task_description": "Solve the task.",
                        "task_description_summary": "Solve the task efficiently.",
                        "code": f"def run_code():\n    return {idx}\n",
                    }
                    for idx in range(10)
                ],
            }
        ],
    )

    cards = load_memory_cards(
        banks_path,
        best_ideas_path=best_ideas_path,
        programs_path=programs_path,
        best_programs_percent=5.0,
    )

    idea_cards = [card for card in cards if card.category != "program"]
    program_cards = [card for card in cards if card.category == "program"]

    assert len(idea_cards) == 1
    assert len(program_cards) == 1

    program_card = program_cards[0]
    assert program_card.program_id == "prog-9"
    assert program_card.fitness == 9.0
    assert program_card.task_description_summary == "Solve the task efficiently."
    assert "def run_code()" in program_card.code
    assert len(program_card.connected_ideas) == 2
    assert program_card.connected_ideas[0].idea_id == "idea-1"
    assert program_card.connected_ideas[1].idea_id == "idea-2"
    assert isinstance(program_card, ProgramCard)


def test_program_cards_bypass_idea_dedup(tmp_path):
    memory = make_test_memory(tmp_path, card_update_dedup_config={"enabled": True})
    memory.save_card(
        {
            "id": "idea-1",
            "category": "general",
            "description": "Repair invalid candidates before scoring.",
            "task_description": "Solve task.",
            "task_description_summary": "Solve task.",
        }
    )
    memory.llm_service = object()

    def _unexpected_call(*args, **kwargs):
        raise AssertionError("Program cards should not use idea-card dedup.")

    memory.dedup.score_duplicate_candidates = _unexpected_call  # type: ignore[method-assign]

    card_id = memory.save_card(
        {
            "id": "program-prog-1",
            "category": "program",
            "program_id": "prog-1",
            "fitness": 12.5,
            "task_description": "Solve task.",
            "task_description_summary": "Solve task.",
            "description": "Top evolved program for Solve task. (fitness=12.5).",
            "code": "def run_code():\n    return 1\n",
            "connected_ideas": [
                {
                    "idea_id": "idea-1",
                    "description": "Repair invalid candidates before scoring.",
                }
            ],
        }
    )

    stored = memory.get_card(card_id)
    assert card_id == "program-prog-1"
    assert stored is not None
    assert stored.program_id == "prog-1"
    assert stored.fitness == 12.5
    assert len(stored.connected_ideas) == 1
    assert stored.connected_ideas[0].idea_id == "idea-1"


def test_normalize_program_card_is_minimal_shape():
    card = normalize_memory_card(
        {
            "id": "program-prog-1",
            "category": "program",
            "program_id": "prog-1",
            "task_description": "Solve task.",
            "task_description_summary": "Solve task.",
            "description": "Top evolved program.",
            "fitness": 12.5,
            "code": "def run_code():\n    return 1\n",
            "connected_ideas": [{"idea_id": "idea-1"}],
            "links": ["idea-1"],
            "strategy": "unused",
        }
    )

    assert card.id == "program-prog-1"
    assert card.category == "program"
    assert card.program_id == "prog-1"
    assert card.fitness == 12.5


def test_load_memory_cards_stamps_posterior_on_program_card(tmp_path):
    banks_path, best_ideas_path, programs_path = _minimal_corpus(tmp_path)
    harm_post = beta_binomial_posterior([-0.01, -0.02, -0.03])

    cards = load_memory_cards(
        banks_path,
        best_ideas_path=best_ideas_path,
        programs_path=programs_path,
        best_programs_percent=5.0,
        card_posterior={"program-prog-9": harm_post},
    )

    program_card = next(c for c in cards if c.category == "program")
    assert program_card.program_id == "prog-9"
    all_block = program_card.evolution_statistics["ALL"]
    assert (all_block["posterior_a"], all_block["posterior_b"]) == (1.0, 4.0)
    assert all_block["k_harm"] == 3


def test_card_posterior_reads_stamped_program_card(tmp_path):
    # End-to-end seam: the auction's _card_posterior must draw the stamped
    # downside posterior off a minted ProgramCard (not COLD Beta(1, 1)).
    banks_path, best_ideas_path, programs_path = _minimal_corpus(tmp_path)
    harm_post = beta_binomial_posterior([-0.01, -0.02, -0.03])

    cards = load_memory_cards(
        banks_path,
        best_ideas_path=best_ideas_path,
        programs_path=programs_path,
        best_programs_percent=5.0,
        card_posterior={"program-prog-9": harm_post},
    )
    program_card = next(c for c in cards if c.category == "program")

    assert BetaBinomialReputation().card_posterior(program_card) == (1.0, 4.0)


def test_posterior_bearing_program_outside_top_slice_is_still_built(tmp_path):
    # A card accrues an injection posterior only after it has been injected
    # downstream, by which point it has usually fallen out of the top-fitness
    # slice; build it anyway so its signal reaches the auction.
    banks_path, best_ideas_path, programs_path = _minimal_corpus(tmp_path)
    harm_post = beta_binomial_posterior([-0.01, -0.02, -0.03])

    cards = load_memory_cards(
        banks_path,
        best_ideas_path=best_ideas_path,
        programs_path=programs_path,
        best_programs_percent=5.0,
        card_posterior={"program-prog-3": harm_post},
    )

    program_cards = {c.program_id: c for c in cards if c.category == "program"}
    assert "prog-9" in program_cards
    assert "prog-3" in program_cards
    stamped = program_cards["prog-3"].evolution_statistics["ALL"]
    assert (stamped["posterior_a"], stamped["posterior_b"]) == (1.0, 4.0)


def test_program_card_without_posterior_is_cold(tmp_path):
    banks_path, best_ideas_path, programs_path = _minimal_corpus(tmp_path)

    cards = load_memory_cards(
        banks_path,
        best_ideas_path=best_ideas_path,
        programs_path=programs_path,
        best_programs_percent=5.0,
    )
    program_card = next(c for c in cards if c.category == "program")

    assert program_card.evolution_statistics == {}
    assert BetaBinomialReputation().card_posterior(program_card) == (1.0, 1.0)


def _idea_corpus(tmp_path):
    """banks / best_ideas pair with idea-1 selected as a best idea."""
    banks_path = tmp_path / "banks.json"
    best_ideas_path = tmp_path / "best_ideas.json"
    _write_json(
        banks_path,
        [
            {
                "timestamp": "t",
                "active_bank": [
                    {
                        "id": "idea-1",
                        "description": "Use simulated annealing for local refinement.",
                        "task_description": "Solve the task.",
                        "task_description_summary": "Solve the task efficiently.",
                        "programs": [],
                    }
                ],
                "inactive_bank": [],
            }
        ],
    )
    _write_json(
        best_ideas_path,
        [
            {
                "timestamp": "t",
                "best_ideas": [
                    {
                        "idea_id": "idea-1",
                        "description": "Use simulated annealing.",
                        "intro_gain_median": 0.01,
                    }
                ],
            }
        ],
    )
    return banks_path, best_ideas_path


def test_load_memory_cards_stamps_idea_card_posterior(tmp_path):
    # The selector injects idea cards too; their ids accrue injection events
    # exactly like program-<uuid> ids, so their posterior must not be dropped.
    banks_path, best_ideas_path = _idea_corpus(tmp_path)
    harm_post = beta_binomial_posterior([-0.01, -0.02, -0.03])

    cards = load_memory_cards(
        banks_path,
        best_ideas_path=best_ideas_path,
        card_posterior={"idea-1": harm_post},
    )

    idea_card = next(c for c in cards if c.category != "program")
    assert idea_card.id == "idea-1"
    all_block = idea_card.evolution_statistics["ALL"]
    assert (all_block["posterior_a"], all_block["posterior_b"]) == (1.0, 4.0)
    assert BetaBinomialReputation().card_posterior(idea_card) == (1.0, 4.0)


def test_idea_posterior_stamp_preserves_best_ideas_snapshot(tmp_path):
    # The "ALL" posterior merges alongside the best_ideas_snapshot metrics
    # already living in evolution_statistics, never replacing them.
    banks_path, best_ideas_path = _idea_corpus(tmp_path)
    harm_post = beta_binomial_posterior([-0.01, -0.02, -0.03])

    cards = load_memory_cards(
        banks_path,
        best_ideas_path=best_ideas_path,
        card_posterior={"idea-1": harm_post},
    )

    idea_card = next(c for c in cards if c.category != "program")
    stats = idea_card.evolution_statistics
    assert stats["best_ideas_snapshot"] == {"intro_gain_median": 0.01}
    assert stats["ALL"]["k_harm"] == 3


def test_idea_card_without_posterior_keeps_stats_untouched(tmp_path):
    banks_path, best_ideas_path = _idea_corpus(tmp_path)

    cards = load_memory_cards(
        banks_path,
        best_ideas_path=best_ideas_path,
        card_posterior={"idea-other": beta_binomial_posterior([0.01])},
    )

    idea_card = next(c for c in cards if c.category != "program")
    assert "ALL" not in idea_card.evolution_statistics
    assert BetaBinomialReputation().card_posterior(idea_card) == (1.0, 1.0)
