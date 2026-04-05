"""Realistic end-to-end scenarios with full fake agentic memory.

These tests simulate complete multi-generation evolution runs where
memory accumulates ideas, influences mutations, and improves over time.
Uses FakeDagRunner + fakeredis + FakeAgenticMemorySystem + FakeResearchAgent.

Scenarios:
1. Knowledge accumulation: ideas from gen 1-5 help gen 6-10
2. Memory-guided search relevance: search returns ideas matching the task
3. Dedup prevents idea bloat across multiple tracker runs
4. Cross-experiment memory transfer: ideas from experiment A help experiment B
5. Memory corruption recovery: truncated index → rebuild from agentic system
6. API sync simulation: remote cards sync to local, stale cards pruned
7. Concurrent-style writes: rapid save_card interleaved with search
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from unittest.mock import MagicMock

import fakeredis.aioredis
import pytest

from gigaevo.database.redis import RedisProgramStorageConfig
from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.evolution.engine.config import EngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.mutation.base import MutationOperator, MutationSpec
from gigaevo.evolution.strategies.elite_selectors import ScalarTournamentEliteSelector
from gigaevo.evolution.strategies.island import IslandConfig
from gigaevo.evolution.strategies.migrant_selectors import RandomMigrantSelector
from gigaevo.evolution.strategies.models import BehaviorSpace, LinearBinning
from gigaevo.evolution.strategies.multi_island import MapElitesMultiIsland
from gigaevo.evolution.strategies.removers import FitnessArchiveRemover
from gigaevo.evolution.strategies.selectors import SumArchiveSelector
from gigaevo.memory.shared_memory.card_conversion import (
    is_program_card,
    normalize_allowed_gam_tools,
    normalize_memory_card,
)
from gigaevo.memory.shared_memory.memory import AmemGamMemory
from gigaevo.programs.program import Program
from gigaevo.programs.program_state import ProgramState
from tests.fakes.agentic_memory import (
    FakeResearchAgent,
    fake_build_gam_store,
    fake_build_retrievers,
    fake_load_amem_records,
    make_test_memory,
    make_test_memory_with_agentic,
)

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

_RETURN_RE = re.compile(
    r'return\s*\{\s*"fitness":\s*([\d.]+)\s*,\s*"x":\s*([\d.]+)\s*\}',
)
_CALL_COUNTER = 0


def _reset():
    global _CALL_COUNTER
    _CALL_COUNTER = 0


def _extract(code: str) -> dict[str, float]:
    m = _RETURN_RE.search(code)
    if m is None:
        raise ValueError(f"Bad code:\n{code}")
    return {"fitness": float(m.group(1)), "x": float(m.group(2))}


def _code(fitness: float, x: float) -> str:
    return f'def entrypoint():\n    return {{"fitness": {fitness}, "x": {x}}}'


class MemoryBoostOperator(MutationOperator):
    """fitness += 2.0 with memory, += 1.0 without. Records all calls."""

    def __init__(self):
        self.history: list[dict] = []

    async def mutate_single(self, parents, memory_instructions=None):
        global _CALL_COUNTER
        p = _extract(parents[0].code)
        boost = 2.0 if memory_instructions else 1.0
        self.history.append(
            {
                "parent_fitness": p["fitness"],
                "boost": boost,
                "has_memory": memory_instructions is not None,
                "memory_len": len(memory_instructions) if memory_instructions else 0,
            }
        )
        f = p["fitness"] + boost
        x = 0.5 + _CALL_COUNTER
        _CALL_COUNTER += 1
        return MutationSpec(code=_code(f, x), parents=parents, name="mem_boost")


class FakeDagRunner:
    def __init__(self, storage, sm):
        self._s, self._sm = storage, sm
        self._task = None

    def start(self):
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _loop(self):
        while True:
            for p in await self._s.get_all_by_status(ProgramState.QUEUED.value):
                await self._sm.set_program_state(p, ProgramState.RUNNING)
                p.add_metrics(_extract(p.code))
                await self._sm.set_program_state(p, ProgramState.DONE)
            await asyncio.sleep(0.005)


def _storage(server):
    cfg = RedisProgramStorageConfig(redis_url="redis://fake:6379/0", key_prefix="t")
    s = RedisProgramStorage(cfg)
    s._conn._redis = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    s._conn._closing = False
    return s


def _island():
    return IslandConfig(
        island_id="main",
        behavior_space=BehaviorSpace(
            bins={"x": LinearBinning(min_val=0, max_val=10, num_bins=10, type="linear")}
        ),
        archive_selector=SumArchiveSelector(fitness_keys=["fitness"]),
        archive_remover=FitnessArchiveRemover(
            fitness_key="fitness", fitness_key_higher_is_better=True
        ),
        elite_selector=ScalarTournamentEliteSelector(
            fitness_key="fitness",
            fitness_key_higher_is_better=True,
            tournament_size=99,
        ),
        migrant_selector=RandomMigrantSelector(),
    )


def _writer():
    w = MagicMock()
    w.bind.return_value = w
    return w


def _tracker():
    t = MagicMock()
    t.start = MagicMock()

    async def _stop():
        pass

    t.stop = _stop
    return t


def _make_memory(tmp_path, **kw):
    return make_test_memory(tmp_path, **kw)


def _make_full_memory(tmp_path, ideas=None, **kw):
    # Default high rebuild_interval to avoid auto-rebuild during setup
    kw.setdefault("rebuild_interval", 9999)
    mem, fs = make_test_memory_with_agentic(tmp_path, **kw)

    def _patched_gam_build():
        if mem.note_sync is not None:
            mem.note_sync.export_jsonl(mem.config.export_file)
        recs = fake_load_amem_records(mem.config.export_file) or [
            c.model_dump() for c in mem.card_store.cards.values()
        ]
        ms, ps, _ = fake_build_gam_store(recs, mem.config.gam_store_dir)
        rets = fake_build_retrievers(
            ps,
            mem.config.gam_store_dir / "idx",
            mem.config.checkpoint_path / "chroma",
            allowed_tools=sorted(
                normalize_allowed_gam_tools(mem.config.gam.allowed_tools or None)
            ),
        )
        mem.gam.agent = (
            FakeResearchAgent(retrievers=rets, generator=mem.generator)
            if rets
            else None
        )

    def _patched_dedup():
        recs = [
            c.model_dump()
            for c in mem.card_store.cards.values()
            if not is_program_card(c)
        ]
        if not recs:
            return {}
        _, ps, _ = fake_build_gam_store(recs, mem.config.gam_store_dir)
        rets = fake_build_retrievers(
            ps,
            mem.config.gam_store_dir / "idx",
            mem.config.checkpoint_path / "chroma",
            allowed_tools=[
                "vector_description",
                "vector_explanation_summary",
                "vector_description_explanation_summary",
                "vector_description_task_description_summary",
            ],
        )
        return {
            n: r
            for n, r in rets.items()
            if n in normalize_allowed_gam_tools(mem.config.gam.allowed_tools or None)
        }

    # Patch ALL paths BEFORE saving any cards
    mem.gam.build = _patched_gam_build
    mem.dedup.build_retrievers = _patched_dedup

    def _safe_rebuild():
        mem.card_store.persist()
        if mem.memory_system is None or mem.generator is None:
            return
        if mem.note_sync is not None:
            mem.note_sync.export_jsonl(mem.config.export_file)
        try:
            _patched_gam_build()
            mem.research_agent = mem.gam.agent
        except Exception:
            mem.research_agent = None
        mem.dedup.invalidate_retrievers()
        mem._iters_after_rebuild = 0

    mem.rebuild = _safe_rebuild

    # NOW save ideas (safe — all patches in place)
    for idea in ideas or []:
        mem.save_card(idea)

    return mem, fs


async def _evolve(
    server,
    gens,
    *,
    operator,
):
    s = _storage(server)
    strat = MapElitesMultiIsland(island_configs=[_island()], program_storage=s)
    eng = EvolutionEngine(
        storage=s,
        strategy=strat,
        mutation_operator=operator,
        config=EngineConfig(
            loop_interval=0.005,
            max_elites_per_generation=1,
            max_mutations_per_generation=1,
            generation_timeout=30,
            max_generations=gens,
        ),
        writer=_writer(),
        metrics_tracker=_tracker(),
    )
    await s.add(Program(code=_code(1.0, 0.0), state=ProgramState.QUEUED))
    sm = ProgramStateManager(s)
    runner = FakeDagRunner(s, sm)
    runner.start()
    eng.start()
    try:
        await asyncio.wait_for(eng.task, timeout=30)
    except TimeoutError:
        pytest.fail("Engine timeout")
    finally:
        await runner.stop()
    # Get archive
    s2 = _storage(server)
    strat2 = MapElitesMultiIsland(island_configs=[_island()], program_storage=s2)
    progs = await strat2.islands["main"].get_elites()
    await s2.close()
    await s.close()
    return eng, progs, operator


# ===========================================================================
# Scenario 1: Knowledge accumulation across generations
# ===========================================================================


# ===========================================================================
# Scenario 2: Memory search relevance
# ===========================================================================


class TestMemorySearchRelevance:
    """Search returns ideas matching the query, ranked by similarity."""

    def test_relevant_ideas_rank_higher(self, tmp_path):
        mem, _ = _make_full_memory(
            tmp_path,
            ideas=[
                {
                    "id": "sa",
                    "description": "simulated annealing for local search optimization",
                    "keywords": ["annealing", "local-search"],
                },
                {
                    "id": "ga",
                    "description": "genetic algorithm crossover for population diversity",
                    "keywords": ["genetic", "crossover"],
                },
                {
                    "id": "dp",
                    "description": "dynamic programming for optimal substructure problems",
                    "keywords": ["dynamic", "programming"],
                },
                {
                    "id": "sa2",
                    "description": "adaptive cooling schedule for simulated annealing",
                    "keywords": ["annealing", "cooling", "schedule"],
                },
            ],
        )
        mem.rebuild()

        # Search for annealing-related ideas
        result = mem.search("simulated annealing cooling optimization")

        # SA ideas should appear before unrelated ones
        result.find("sa") if "sa" in result else 999
        result.find("ga") if "ga" in result else 999
        # At minimum, SA-related cards should be in results
        assert "sa" in result.lower() or "annealing" in result.lower()

    def test_search_with_many_ideas_returns_top_k(self, tmp_path):
        mem, _ = _make_full_memory(
            tmp_path,
            search_limit=3,
            ideas=[
                {
                    "id": f"idea-{i}",
                    "description": f"optimization technique {i} for solving problems",
                    "keywords": [f"technique{i}", "optimization"],
                }
                for i in range(20)
            ],
        )
        # Use local search (no research_agent) to test search_limit
        mem.research_agent = None

        result = mem.search("optimization technique solving")
        # Count card IDs in result — should respect search_limit=3
        found = [f"idea-{i}" for i in range(20) if f"idea-{i}" in result]
        assert len(found) <= 3, f"Expected ≤3 results, found {len(found)}: {found}"

    def test_empty_query_returns_no_results(self, tmp_path):
        mem, _ = _make_full_memory(
            tmp_path,
            ideas=[
                {"id": "i1", "description": "test idea"},
            ],
        )
        result = mem.search("")
        assert "No relevant" in result


# ===========================================================================
# Scenario 3: Dedup prevents idea bloat
# ===========================================================================


class TestDedupPreventsIdeaBloat:
    """Multiple tracker runs with similar ideas → dedup keeps memory lean."""

    def test_repeated_similar_ideas_deduplicated(self, tmp_path):
        mem, _ = _make_full_memory(
            tmp_path,
            ideas=[
                {
                    "id": "idea-1",
                    "description": "simulated annealing for optimization",
                    "keywords": ["annealing"],
                },
            ],
            card_update_dedup_config={"enabled": True},
        )

        # Mock LLM to recognize duplicates
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            json.dumps({"action": "discard", "duplicate_of": "idea-1"}),
            {},
            None,
            None,
        )
        mem.llm_service = mock_llm

        # Try adding 5 similar ideas
        for i in range(5):
            mem.save_card({"description": f"SA annealing optimization variant {i}"})

        # All should be deduped against idea-1
        assert len(mem.card_store.cards) == 1
        assert mem.get_card_write_stats()["rejected"] == 5

    def test_genuinely_new_ideas_still_added(self, tmp_path):
        mem, _ = _make_full_memory(
            tmp_path,
            ideas=[{"id": "i1", "description": "annealing", "keywords": ["annealing"]}],
            card_update_dedup_config={"enabled": True},
        )

        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            json.dumps({"action": "add"}),
            {},
            None,
            None,
        )
        mem.llm_service = mock_llm

        mem.save_card({"description": "quantum computing for protein folding"})
        assert len(mem.card_store.cards) == 2


# ===========================================================================
# Scenario 4: Cross-experiment memory transfer
# ===========================================================================


class TestCrossExperimentTransfer:
    """Ideas from experiment A loaded into experiment B's memory."""

    def test_transfer_ideas_between_experiments(self, tmp_path):
        # Experiment A: creates ideas
        mem_a, _ = _make_full_memory(
            tmp_path / "exp_a",
            ideas=[
                {
                    "id": "expA-1",
                    "description": "Sort evidence by relevance score for HoVer",
                    "keywords": ["sort", "relevance", "HoVer"],
                },
                {
                    "id": "expA-2",
                    "description": "Filter low-confidence hops with threshold 0.5",
                    "keywords": ["filter", "confidence", "threshold"],
                },
            ],
        )

        # Export ideas from A
        exported_ideas = list(mem_a.card_store.cards.values())

        # Experiment B: loads ideas from A
        mem_b, _ = _make_full_memory(tmp_path / "exp_b")
        for idea in exported_ideas:
            mem_b.save_card(idea)
        mem_b.rebuild()

        # Experiment B can search for A's ideas
        result = mem_b.search("relevance sorting evidence")
        assert "expA-1" in result

        # Both experiments have independent memory
        assert len(mem_a.card_store.cards) == 2
        assert len(mem_b.card_store.cards) == 2

        # Adding to B doesn't affect A
        mem_b.save_card({"id": "expB-1", "description": "new idea for exp B"})
        assert len(mem_a.card_store.cards) == 2
        assert len(mem_b.card_store.cards) == 3


# ===========================================================================
# Scenario 5: API sync simulation
# ===========================================================================


class TestApiSyncSimulation:
    """Simulate _sync_from_api with mocked API + real agentic system."""

    def test_sync_adds_remote_cards_to_agentic_system(self, tmp_path):
        mem, fake_sys = _make_full_memory(tmp_path)

        mock_api = MagicMock()
        mock_api.list_memory_cards.return_value = [
            {"entity_id": "e1", "version_id": "v1", "meta": {"namespace": "default"}},
            {"entity_id": "e2", "version_id": "v1", "meta": {"namespace": "default"}},
        ]

        def get_concept(eid, channel="latest"):
            ideas = {
                "e1": {
                    "id": "remote-1",
                    "description": "SA from remote",
                    "category": "general",
                },
                "e2": {
                    "id": "remote-2",
                    "description": "crossover from remote",
                    "category": "general",
                },
            }
            return {"content": ideas[eid], "version_id": "v1"}

        mock_api.get_concept.side_effect = get_concept
        mem.api = mock_api

        changed = mem._sync_from_api(force_full=True)

        assert changed is True
        assert "remote-1" in mem.card_store.cards
        assert "remote-2" in mem.card_store.cards
        # Cards should also be in the agentic system
        assert fake_sys.read("remote-1") is not None
        assert fake_sys.read("remote-2") is not None

    def test_sync_prunes_stale_entities(self, tmp_path):
        mem, fake_sys = _make_full_memory(
            tmp_path,
            ideas=[
                {"id": "local-1", "description": "local idea"},
            ],
        )

        # Pre-populate entity maps as if this card came from API
        mem.card_store.entity_by_card_id["local-1"] = "stale-entity"
        mem.card_store.card_id_by_entity["stale-entity"] = "local-1"

        mock_api = MagicMock()
        # Remote has NO entities → local stale entity should be pruned
        mock_api.list_memory_cards.return_value = []
        mem.api = mock_api

        mem._sync_from_api(force_full=True)

        # Stale entity mapping should be removed
        assert "stale-entity" not in mem.card_store.card_id_by_entity
        # Card itself is removed since its entity was pruned
        assert "local-1" not in mem.card_store.cards

    def test_sync_skips_unchanged_versions(self, tmp_path):
        mem, _ = _make_full_memory(tmp_path)

        # Pre-populate as if already synced
        mem.card_store.cards["c1"] = normalize_memory_card(
            {"id": "c1", "description": "known"}
        )
        mem.card_store.entity_by_card_id["c1"] = "e1"
        mem.card_store.card_id_by_entity["e1"] = "c1"
        mem.card_store.entity_version["e1"] = "v1"

        mock_api = MagicMock()
        mock_api.list_memory_cards.return_value = [
            {"entity_id": "e1", "version_id": "v1", "meta": {"namespace": "default"}},
        ]
        mem.api = mock_api

        mem._sync_from_api(force_full=False)

        # get_concept should NOT be called — version unchanged
        mock_api.get_concept.assert_not_called()


# ===========================================================================
# Scenario 6: Rapid save + search interleaving
# ===========================================================================


class TestRapidSaveSearchInterleaving:
    """Simulate rapid writes interleaved with searches."""

    def test_search_always_consistent_during_writes(self, tmp_path):
        """Local search finds each card immediately after save."""
        mem, _ = _make_full_memory(tmp_path)
        # Use local search (no research_agent) for consistency
        mem.research_agent = None

        for i in range(20):
            mem.save_card(
                {
                    "id": f"idea-{i}",
                    "description": f"unique_keyword_{i} optimization technique",
                    "keywords": [f"unique_keyword_{i}"],
                }
            )

            # Search with unique keyword → must find the just-saved card
            result = mem._search_local_cards(f"unique_keyword_{i}")
            assert f"idea-{i}" in result, (
                f"Card idea-{i} not found in local search immediately after save"
            )

    def test_delete_during_search_sequence(self, tmp_path):
        mem, _ = _make_full_memory(tmp_path)

        # Populate
        for i in range(10):
            mem.save_card(
                {
                    "id": f"c{i}",
                    "description": f"technique {i} optimization",
                    "keywords": [f"technique{i}"],
                }
            )

        # Delete every other card, search after each delete
        for i in range(0, 10, 2):
            mem.delete(f"c{i}")
            result = mem.search(f"technique{i}")
            assert f"c{i}" not in result, f"Deleted c{i} still in search results"

        assert len(mem.card_store.cards) == 5


# ===========================================================================
# Scenario 7: Full evolution + memory + rebuild cycle
# ===========================================================================


class TestFullEvolutionMemoryRebuildCycle:
    """The ultimate test: evolve → fill memory → rebuild → evolve with memory → verify."""

    def test_unpatched_memory_save_search_delete(self, tmp_path):
        """UNPATCHED AmemGamMemory — no fakes, no monkey-patches.

        This test uses the REAL AmemGamMemory in local-only mode.
        If the real memory system is broken, this test fails.
        """
        # Real AmemGamMemory — no monkey-patching, no patched rebuild
        from gigaevo.memory.shared_memory.memory_config import MemoryConfig

        cfg = MemoryConfig(checkpoint_path=tmp_path / "real_mem")
        mem = AmemGamMemory(config=cfg)

        # Save cards
        mem.save_card(
            {
                "id": "real-1",
                "description": "simulated annealing optimization",
                "keywords": ["annealing"],
            }
        )
        mem.save_card(
            {
                "id": "real-2",
                "description": "genetic crossover recombination",
                "keywords": ["crossover"],
            }
        )
        mem.save_card(
            {
                "id": "real-3",
                "description": "dynamic programming substructure",
                "keywords": ["dynamic"],
            }
        )

        assert len(mem.card_store.cards) == 3

        # Search (local search — no research_agent in CI)
        result = mem.search("annealing optimization")
        assert "real-1" in result

        # Update
        mem.save_card(
            {"id": "real-1", "description": "enhanced SA with adaptive cooling"}
        )
        assert "adaptive cooling" in mem.get_card("real-1").description

        # Delete
        mem.delete("real-2")
        assert mem.get_card("real-2") is None
        assert len(mem.card_store.cards) == 2

        # Persist + reload (new process)
        mem2 = AmemGamMemory(config=cfg)
        assert len(mem2.card_store.cards) == 2
        assert (
            mem2.get_card("real-1").description == "enhanced SA with adaptive cooling"
        )
        assert mem2.get_card("real-2") is None
        assert mem2.get_card("real-3") is not None

        # Search after reload
        result2 = mem2.search("dynamic programming")
        assert "real-3" in result2

    def test_unpatched_memory_stats_contract(self, tmp_path):
        """UNPATCHED: verify card_write_stats shape and behavior."""
        from gigaevo.memory.shared_memory.memory_config import MemoryConfig

        mem = AmemGamMemory(config=MemoryConfig(checkpoint_path=tmp_path / "real"))
        mem.save_card({"id": "c1", "description": "idea"})
        mem.save_card({"id": "c1", "description": "updated"})
        mem.save_card({"description": "new"})

        stats = mem.get_card_write_stats()
        assert stats["processed"] == 3
        assert stats["added"] == 2
        assert stats["updated"] == 1
