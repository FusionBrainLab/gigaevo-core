"""End-to-end integration scenarios for the memory system.

Each test simulates a realistic multi-step workflow:
  Run 1: evolution produces programs → IdeaTracker extracts ideas → write to memory
  Run 2: memory loaded → MemorySelectorAgent queries → cards injected into mutation

All tests use local-only mode (no API, no network, no Redis).
"""

import asyncio
from dataclasses import dataclass, field
import json
from unittest.mock import MagicMock

from gigaevo.memory.ideas_tracker.components.data_components import (
    IncomingIdeas,
    RecordBank,
)
from gigaevo.memory.memory_write_example import load_memory_cards
from gigaevo.memory.shared_memory.memory import AmemGamMemory
from gigaevo.memory.shared_memory.models import ProgramCard
from tests.fakes.agentic_memory import make_test_memory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memory(tmp_path, **overrides):
    return make_test_memory(tmp_path, **overrides)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


@dataclass
class FakeProgram:
    code: str = "def solve(): return 1"
    metadata: dict = field(default_factory=dict)


# ===========================================================================
# Scenario 1: Full two-run cycle (fill + use)
# ===========================================================================


class TestScenarioTwoRunCycle:
    """Simulate the canonical two-run workflow:
    Run 1: programs → ideas → memory cards → persist
    Run 2: load memory → search → inject into mutation
    """

    def _create_ideas_from_programs(self, tmp_path):
        """Simulate IdeaTracker: programs → ideas bank → best_ideas → memory cards."""
        # Programs from evolution run (simulated Redis export)
        programs = [
            {
                "id": "prog-1",
                "fitness": 92.0,
                "generation": 15,
                "code": "def solve(x):\n    return sorted(x, key=lambda t: t[1])\n",
                "task_description": "Multi-hop fact verification",
                "task_description_summary": "HoVer verification",
            },
            {
                "id": "prog-2",
                "fitness": 88.5,
                "generation": 12,
                "code": "def solve(x):\n    return [hop for hop in x if hop['score'] > 0.5]\n",
                "task_description": "Multi-hop fact verification",
                "task_description_summary": "HoVer verification",
            },
            {
                "id": "prog-3",
                "fitness": 78.0,
                "generation": 8,
                "code": "def solve(x):\n    return x[:3]\n",
                "task_description": "Multi-hop fact verification",
                "task_description_summary": "HoVer verification",
            },
        ]

        # Simulate IdeaTracker classification: extract ideas from improvements
        ideas_bank = RecordBank(list_max_ideas=20)
        ideas_bank.add_idea(
            "Sort evidence by relevance score before hop selection",
            "prog-1",
            generation=15,
            category="retrieval",
            strategy="exploitation",
            task_description="Multi-hop fact verification",
            change_motivation="Sorting improves evidence chain quality",
        )
        ideas_bank.add_idea(
            "Filter low-confidence hops using threshold 0.5",
            "prog-2",
            generation=12,
            category="filtering",
            strategy="exploration",
            task_description="Multi-hop fact verification",
            change_motivation="Reduces noise in multi-hop chains",
        )
        ideas_bank.add_idea(
            "Limit retrieval depth to 3 hops maximum",
            "prog-3",
            generation=8,
            category="retrieval",
            strategy="exploitation",
            task_description="Multi-hop fact verification",
            change_motivation="Prevents over-retrieval degradation",
        )

        # Write banks.json and best_ideas.json (IdeaTracker output)
        all_ideas = ideas_bank.all_ideas_cards()
        active_bank = []
        for idea in all_ideas:
            active_bank.append(
                {
                    "id": idea.id,
                    "description": idea.description,
                    "task_description": idea.task_description,
                    "task_description_summary": "HoVer verification",
                    "programs": idea.programs,
                    "explanation": idea.explanation,
                    "strategy": idea.strategy,
                    "category": idea.category,
                }
            )

        banks_path = tmp_path / "logs" / "banks.json"
        _write_json(banks_path, [{"active_bank": active_bank}])

        best_ideas = [
            {"idea_id": idea.id, "description": idea.description} for idea in all_ideas
        ]
        best_ideas_path = tmp_path / "logs" / "best_ideas.json"
        _write_json(best_ideas_path, [{"best_ideas": best_ideas}])

        programs_path = tmp_path / "logs" / "programs.json"
        _write_json(programs_path, [{"programs": programs}])

        return banks_path, best_ideas_path, programs_path, ideas_bank

    def test_full_two_run_cycle(self, tmp_path):
        """Run 1: fill memory. Run 2: load and search."""
        banks, best_ideas, programs, _ = self._create_ideas_from_programs(tmp_path)

        # --- Run 1: Write to memory ---
        cards = load_memory_cards(
            banks,
            best_ideas,
            programs_path=programs,
            best_programs_percent=50.0,
        )
        assert len(cards) >= 3  # 3 ideas + at least 1 program

        mem = _make_memory(tmp_path)
        for card in cards:
            mem.save_card(card)

        stats = mem.get_card_write_stats()
        assert stats["processed"] == len(cards)
        assert stats["added"] == len(cards)
        assert stats["rejected"] == 0

        # Verify persistence
        assert mem.config.index_file.exists()

        # --- Run 2: Load and search ---
        mem2 = _make_memory(tmp_path)
        assert len(mem2.card_store.cards) == len(cards)

        # Search for retrieval ideas — verify by card ID presence
        idea_ids = [uid for uid in mem2.card_store.cards]
        result = mem2.search("evidence retrieval sorting relevance")
        # At least one idea card ID should appear in the formatted result
        matched_ids = [uid for uid in idea_ids if uid in result]
        assert len(matched_ids) > 0, (
            f"No card IDs found in search result: {result[:200]}"
        )

        # Search for filtering ideas
        result = mem2.search("confidence threshold filtering hops")
        matched_ids = [uid for uid in idea_ids if uid in result]
        assert len(matched_ids) > 0, (
            f"No card IDs found in search result: {result[:200]}"
        )

    def test_memory_guides_mutation(self, tmp_path):
        """Full cycle: fill memory → MemorySelectorAgent.select() → cards in mutation."""
        from gigaevo.llm.agents.memory_selector import MemorySelectorAgent

        banks, best_ideas, _, _ = self._create_ideas_from_programs(tmp_path)
        cards = load_memory_cards(banks, best_ideas)

        # Fill memory
        mem = _make_memory(tmp_path)
        for card in cards:
            mem.save_card(card)

        # Reload (simulating new process)
        mem2 = _make_memory(tmp_path)

        # Create selector with real memory
        selector = MemorySelectorAgent.__new__(MemorySelectorAgent)
        selector._search_lock = asyncio.Lock()
        selector._backend_error = None
        selector.memory = mem2

        # Simulate mutation query
        parent = FakeProgram(
            code="def solve(x):\n    return x\n",
            metadata={"mutation_context": "Basic identity function"},
        )

        loop = asyncio.new_event_loop()
        try:
            selection = loop.run_until_complete(
                selector.select(
                    input=[parent],
                    mutation_mode="rewrite",
                    task_description="Multi-hop fact verification for HoVer dataset",
                    metrics_description="fitness: label accuracy on validation set",
                    memory_text="",
                    max_cards=3,
                )
            )
        finally:
            loop.close()

        # Should get results from memory
        assert len(selection.cards) > 0 or len(selection.card_ids) > 0


# ===========================================================================
# Scenario 2: Incremental memory growth across generations
# ===========================================================================


class TestScenarioIncrementalGrowth:
    """Simulate multiple IdeaTracker runs adding ideas over time."""

    def test_memory_grows_across_tracker_runs(self, tmp_path):
        mem = _make_memory(tmp_path)

        # Generation 5: first batch of ideas
        gen5_ideas = [
            {
                "id": "idea-g5-1",
                "description": "Use beam search for chain construction",
                "task_description": "Multi-hop QA",
                "task_description_summary": "QA",
            },
            {
                "id": "idea-g5-2",
                "description": "Increase retrieval context window",
                "task_description": "Multi-hop QA",
                "task_description_summary": "QA",
            },
        ]
        for idea in gen5_ideas:
            mem.save_card(idea)
        assert len(mem.card_store.cards) == 2

        # Persist and reload (simulating process restart between generations)
        mem2 = _make_memory(tmp_path)
        assert len(mem2.card_store.cards) == 2

        # Generation 10: second batch
        gen10_ideas = [
            {
                "id": "idea-g10-1",
                "description": "Apply chain-of-thought prompting",
                "task_description": "Multi-hop QA",
                "task_description_summary": "QA",
            },
            {
                "id": "idea-g10-2",
                "description": "Dynamic hop count based on query complexity",
                "task_description": "Multi-hop QA",
                "task_description_summary": "QA",
            },
        ]
        for idea in gen10_ideas:
            mem2.save_card(idea)
        assert len(mem2.card_store.cards) == 4

        # Generation 15: update existing idea
        mem3 = _make_memory(tmp_path)
        assert len(mem3.card_store.cards) == 4
        mem3.save_card(
            {
                "id": "idea-g5-1",
                "description": "Use beam search with width=3 for chain construction",
            }
        )
        # Updated, not duplicated
        assert len(mem3.card_store.cards) == 4
        card = mem3.get_card("idea-g5-1")
        assert "width=3" in card.description

    def test_ideas_and_programs_accumulate(self, tmp_path):
        """Ideas from gen 5 + programs from gen 10 coexist."""
        mem = _make_memory(tmp_path)

        # Ideas from early generations
        mem.save_card(
            {
                "id": "idea-1",
                "description": "sort by relevance",
                "task_description": "HoVer",
            }
        )
        mem.save_card(
            {
                "id": "idea-2",
                "description": "filter low confidence",
                "task_description": "HoVer",
            }
        )

        # Programs from later generations
        mem.save_card(
            {
                "category": "program",
                "program_id": "prog-best-1",
                "description": "Top program for HoVer",
                "fitness": 95.0,
                "code": "def solve(): return 42",
                "task_description": "HoVer",
            }
        )

        # Reload and verify
        mem2 = _make_memory(tmp_path)
        assert len(mem2.card_store.cards) == 3

        # Search finds ideas
        result = mem2.search("relevance sorting")
        assert "idea-1" in result

        # Program card stored — save_card uses the ID as-is since it's provided
        # The card was saved with category="program", program_id="prog-best-1"
        # but the explicit id was "program-prog-best-1" (we set it, not auto-prefixed)
        # Let's verify by checking all cards
        prog_cards = [
            c for c in mem2.card_store.cards.values() if isinstance(c, ProgramCard)
        ]
        assert len(prog_cards) == 1
        assert prog_cards[0].fitness == 95.0


# ===========================================================================
# Scenario 3: Dedup across multiple write sessions
# ===========================================================================


class TestScenarioDedup:
    """Simulate dedup when re-running IdeaTracker with overlapping ideas."""

    def test_same_id_updates_not_duplicates(self, tmp_path):
        """Re-saving with same ID updates the card, doesn't create duplicate."""
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "idea-1", "description": "SA optimization v1"})
        mem.save_card({"id": "idea-1", "description": "SA optimization v2 (improved)"})

        assert len(mem.card_store.cards) == 1
        assert "v2" in mem.get_card("idea-1").description

        stats = mem.get_card_write_stats()
        assert stats["updated"] == 1
        assert stats["added"] == 1

    def test_llm_dedup_rejects_similar_card(self, tmp_path):
        """With LLM dedup enabled, similar new card is rejected."""
        mem = _make_memory(tmp_path, card_update_dedup_config={"enabled": True})
        mem.save_card({"id": "idea-1", "description": "Use simulated annealing"})

        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            json.dumps({"action": "discard", "duplicate_of": "idea-1"}),
            {},
            None,
            None,
        )
        mem.llm_service = mock_llm
        mem.dedup.score_candidates = MagicMock(
            return_value=[{"card_id": "idea-1", "final_score": 0.9}]
        )

        result_id = mem.save_card({"description": "Apply SA for local search"})
        assert result_id == "idea-1"
        assert len(mem.card_store.cards) == 1
        assert mem.get_card_write_stats()["rejected"] == 1

    def test_llm_dedup_updates_existing_card(self, tmp_path):
        """LLM says 'update' → existing card gets new explanation merged."""
        mem = _make_memory(tmp_path, card_update_dedup_config={"enabled": True})
        mem.save_card(
            {
                "id": "idea-1",
                "description": "SA optimization",
                "explanation": {
                    "explanations": ["original insight"],
                    "summary": "SA works",
                },
            }
        )

        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            json.dumps(
                {
                    "action": "update",
                    "updates": [
                        {
                            "card_id": "idea-1",
                            "update_explanation": True,
                            "explanation_append": "Also effective for multi-hop chains",
                        }
                    ],
                }
            ),
            {},
            None,
            None,
        )
        mem.llm_service = mock_llm
        mem.dedup.score_candidates = MagicMock(
            return_value=[{"card_id": "idea-1", "final_score": 0.85}]
        )

        result_id = mem.save_card({"description": "SA for chain optimization"})
        assert result_id == "idea-1"
        card = mem.get_card("idea-1")
        assert "multi-hop chains" in str(card.explanation)
        stats = mem.get_card_write_stats()
        assert stats["updated"] == 1
        assert stats["updated_target_cards"] == 1


# ===========================================================================
# Scenario 4: Memory write pipeline end-to-end
# ===========================================================================


class TestScenarioWritePipeline:
    """Full write pipeline: banks.json → load_memory_cards → save → search."""

    def _setup_tracker_output(self, tmp_path, n_ideas=5, n_programs=3):
        """Generate realistic IdeaTracker output files."""
        active_bank = []
        for i in range(n_ideas):
            active_bank.append(
                {
                    "id": f"idea-{i}",
                    "description": f"Optimization technique {i}: improve convergence via method_{i}",
                    "task_description": "Multi-hop fact verification on HoVer",
                    "task_description_summary": "HoVer verification",
                    "programs": [f"prog-{i}", f"prog-{i + 10}"],
                    "explanation": {
                        "explanations": [f"Method {i} explanation"],
                        "summary": f"Method {i} works",
                    },
                    "strategy": "exploitation" if i % 2 == 0 else "exploration",
                    "category": "retrieval" if i < 3 else "filtering",
                    "keywords": [f"method_{i}", "optimization"],
                }
            )

        banks = tmp_path / "banks.json"
        _write_json(banks, [{"active_bank": active_bank}])

        best_ideas = [
            {"idea_id": f"idea-{i}", "description": f"Technique {i}"}
            for i in range(n_ideas)
        ]
        best = tmp_path / "best_ideas.json"
        _write_json(best, [{"best_ideas": best_ideas}])

        programs = []
        for i in range(n_programs):
            programs.append(
                {
                    "id": f"prog-{i}",
                    "fitness": 90.0 - i * 5,
                    "generation": 10 + i,
                    "code": f"def solve_{i}(x):\n    return x[:{i + 1}]\n",
                    "task_description": "Multi-hop fact verification on HoVer",
                    "task_description_summary": "HoVer verification",
                }
            )
        progs = tmp_path / "programs.json"
        _write_json(progs, [{"programs": programs}])

        return banks, best, progs

    def test_full_pipeline_ideas_only(self, tmp_path):
        banks, best, _ = self._setup_tracker_output(tmp_path, n_ideas=5, n_programs=0)
        cards = load_memory_cards(banks, best)
        assert len(cards) == 5

        mem = _make_memory(tmp_path)
        for card in cards:
            mem.save_card(card)

        mem2 = _make_memory(tmp_path)
        assert len(mem2.card_store.cards) == 5

        # Search for specific method
        result = mem2.search("method_3 optimization")
        assert "idea-3" in result

    def test_full_pipeline_with_programs(self, tmp_path):
        banks, best, progs = self._setup_tracker_output(tmp_path)
        cards = load_memory_cards(
            banks, best, programs_path=progs, best_programs_percent=100.0
        )

        idea_cards = [c for c in cards if not isinstance(c, ProgramCard)]
        prog_cards = [c for c in cards if isinstance(c, ProgramCard)]
        assert len(idea_cards) == 5
        assert len(prog_cards) == 3

        mem = _make_memory(tmp_path)
        for card in cards:
            mem.save_card(card)

        stats = mem.get_card_write_stats()
        assert stats["processed"] == 8
        assert stats["added"] == 8

    def test_pipeline_repeated_writes_idempotent(self, tmp_path):
        """Re-running the write pipeline with same IDs updates, doesn't duplicate."""
        banks, best, _ = self._setup_tracker_output(tmp_path, n_ideas=3)
        cards = load_memory_cards(banks, best)

        mem = _make_memory(tmp_path)
        for card in cards:
            mem.save_card(card)
        assert len(mem.card_store.cards) == 3

        # Re-run (same IDs)
        for card in cards:
            mem.save_card(card)
        assert len(mem.card_store.cards) == 3  # No duplicates

        stats = mem.get_card_write_stats()
        assert stats["added"] == 3
        assert stats["updated"] == 3  # Second run = updates


# ===========================================================================
# Scenario 5: IncomingIdeas → RecordBank → memory write
# ===========================================================================


class TestScenarioIdeasToMemory:
    """Simulate ideas classification → bank → memory card write."""

    def test_incoming_ideas_classified_and_saved(self, tmp_path):
        """Classify incoming ideas, add to bank, then save to memory."""
        # Programs produce improvements (raw LLM output)
        raw_improvements = [
            {
                "description": "Sort evidence by relevance",
                "explanation": "Better chain quality",
            },
            {
                "description": "Filter noise from retrieval",
                "explanation": "Reduces false positives",
            },
            {
                "description": "Sort evidence by relevance",
                "explanation": "Duplicate of first",
            },
        ]

        # Classify via IncomingIdeas
        incoming = IncomingIdeas(raw_improvements)
        assert incoming.new_ideas_count == 3

        # First idea: new
        # Second idea: new
        # Third idea: duplicate of first (classifier would mark it)
        incoming.update_idea(3, target_idea_id="some-existing-id", rewrite=False)
        assert incoming.new_ideas_count == 2
        assert incoming.present_ideas_count == 1

        # Add new ideas to bank
        bank = RecordBank(list_max_ideas=20)
        for idea in incoming.ideas:
            if not idea["classified"]:
                bank.add_idea(
                    description=idea["description"],
                    linked_program="prog-1",
                    generation=5,
                    category="retrieval",
                    strategy="exploitation",
                    task_description="HoVer verification",
                    change_motivation=idea["change_motivation"],
                )

        assert len(bank.uuids) == 2

        # Write bank ideas to memory
        mem = _make_memory(tmp_path)
        for card in bank.all_ideas_cards():
            mem.save_card(
                {
                    "id": card.id,
                    "description": card.description,
                    "task_description": card.task_description,
                    "task_description_summary": "HoVer",
                    "strategy": card.strategy,
                    "category": card.category,
                    "explanation": card.explanation,
                    "programs": card.programs,
                }
            )

        assert len(mem.card_store.cards) == 2

        # Reload and search
        mem2 = _make_memory(tmp_path)
        assert len(mem2.card_store.cards) == 2
        result = mem2.search("evidence relevance sorting")
        assert "Sort evidence" in result or "relevance" in result.lower()


# ===========================================================================
# Scenario 6: Delete and rebuild
# ===========================================================================


class TestScenarioDeleteAndRebuild:
    """Memory card lifecycle: create → verify → delete → verify gone."""

    def test_delete_removes_from_search(self, tmp_path):
        mem = _make_memory(tmp_path)
        mem.save_card(
            {"id": "idea-keep", "description": "Use beam search", "keywords": ["beam"]}
        )
        mem.save_card(
            {
                "id": "idea-remove",
                "description": "Use random search",
                "keywords": ["random"],
            }
        )

        # Both searchable
        assert "idea-keep" in mem.search("beam search")
        assert "idea-remove" in mem.search("random search")

        # Delete one
        mem.delete("idea-remove")

        # Removed from search
        result = mem.search("random search")
        assert "idea-remove" not in result

        # Other still works
        assert "idea-keep" in mem.search("beam search")

        # Persists across reload
        mem2 = _make_memory(tmp_path)
        assert mem2.get_card("idea-remove") is None
        assert mem2.get_card("idea-keep") is not None

    def test_delete_all_then_repopulate(self, tmp_path):
        mem = _make_memory(tmp_path)
        ids = []
        for i in range(5):
            mem.save_card({"id": f"c{i}", "description": f"idea {i}"})
            ids.append(f"c{i}")

        for cid in ids:
            mem.delete(cid)
        assert len(mem.card_store.cards) == 0

        # Repopulate
        mem.save_card({"id": "fresh", "description": "brand new idea"})
        mem2 = _make_memory(tmp_path)
        assert len(mem2.card_store.cards) == 1
        assert mem2.get_card("fresh") is not None


# ===========================================================================
# Scenario 7: Cross-task memory isolation
# ===========================================================================


class TestScenarioCrossTask:
    """Different tasks use separate memory namespaces via checkpoint_path."""

    def test_separate_checkpoint_dirs_isolate_memory(self, tmp_path):
        from gigaevo.memory.shared_memory.memory_config import MemoryConfig

        hover_cfg = MemoryConfig(checkpoint_path=tmp_path / "hover_mem")
        hotpot_cfg = MemoryConfig(checkpoint_path=tmp_path / "hotpot_mem")

        mem_hover = AmemGamMemory(config=hover_cfg)
        mem_hotpot = AmemGamMemory(config=hotpot_cfg)

        mem_hover.save_card({"id": "hover-1", "description": "HoVer retrieval idea"})
        mem_hotpot.save_card({"id": "hotpot-1", "description": "HotpotQA chain idea"})

        # Each has only its own cards
        assert mem_hover.get_card("hover-1") is not None
        assert mem_hover.get_card("hotpot-1") is None
        assert mem_hotpot.get_card("hotpot-1") is not None
        assert mem_hotpot.get_card("hover-1") is None

        # Reload verifies isolation
        mem_hover2 = AmemGamMemory(config=hover_cfg)
        assert len(mem_hover2.card_store.cards) == 1
        assert mem_hover2.get_card("hover-1") is not None


# ===========================================================================
# Scenario 8: Error and corruption recovery
# ===========================================================================


class TestScenarioErrorRecovery:
    """Test behavior when things go wrong: corrupt files, missing fields."""

    def test_truncated_index_json_recovers_empty(self, tmp_path):
        """Crash mid-write → truncated JSON → silent empty start."""
        mem_dir = tmp_path / "mem"
        mem_dir.mkdir(parents=True)

        # Write valid data first
        mem = _make_memory(tmp_path)
        mem.save_card({"id": "c1", "description": "important idea"})
        assert mem.config.index_file.exists()

        # Corrupt the index file (simulating crash mid-write)
        mem.config.index_file.write_text('{"memory_cards": {"c1": {"id": "c1", "des')

        # Reload: data is lost (known bug, documented)
        mem2 = _make_memory(tmp_path)
        assert len(mem2.card_store.cards) == 0

    def test_save_card_with_minimal_fields(self, tmp_path):
        """Cards with only description should still work."""
        mem = _make_memory(tmp_path)
        card_id = mem.save_card({"description": "just a description"})
        card = mem.get_card(card_id)
        assert card is not None
        assert card.description == "just a description"
        assert card.category == "general"

    def test_save_card_with_empty_dict(self, tmp_path):
        """Empty dict should produce a valid card with auto-generated ID."""
        mem = _make_memory(tmp_path)
        card_id = mem.save_card({})
        assert card_id.startswith("mem-")
        card = mem.get_card(card_id)
        assert card is not None

    def test_search_on_empty_memory(self, tmp_path):
        """Search on empty memory returns no-results message, not crash."""
        mem = _make_memory(tmp_path)
        result = mem.search("anything")
        assert "No relevant memories found" in result

    def test_delete_nonexistent_returns_false(self, tmp_path):
        """Deleting non-existent card is a no-op, returns False."""
        mem = _make_memory(tmp_path)
        assert mem.delete("nonexistent") is False

    def test_concurrent_instances_same_checkpoint(self, tmp_path):
        """Two instances reading same checkpoint: both see the same data.
        Write from one, the other still has stale in-memory state."""
        mem1 = _make_memory(tmp_path)
        mem1.save_card({"id": "c1", "description": "from mem1"})

        mem2 = _make_memory(tmp_path)
        assert mem2.get_card("c1") is not None  # Reads persisted data

        # mem1 adds another card
        mem1.save_card({"id": "c2", "description": "also from mem1"})

        # mem2 doesn't see c2 (stale in-memory state)
        assert mem2.get_card("c2") is None

        # But a fresh load does
        mem3 = _make_memory(tmp_path)
        assert mem3.get_card("c2") is not None

    def test_large_memory_search_relevance(self, tmp_path):
        """With 100 cards, search returns the most relevant, not random."""
        mem = _make_memory(tmp_path, search_limit=3)
        for i in range(100):
            mem.save_card(
                {
                    "id": f"noise-{i}",
                    "description": f"generic optimization idea number {i}",
                }
            )
        # Add a very specific card
        mem.save_card(
            {
                "id": "target",
                "description": "quantum annealing for protein folding",
                "keywords": ["quantum", "annealing", "protein"],
            }
        )

        result = mem.search("quantum annealing protein")
        assert "target" in result
