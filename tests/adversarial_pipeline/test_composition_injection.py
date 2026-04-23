"""Tests for CompositionInjectionHook — code-level D ∘ G composition with permanent dedup.

The hook iterates ALL G programs in storage. For each G, it asks the
DGImprovementTracker for the best D and (if the (D, G) pair has not been
injected before) composes a new G program whose entrypoint chains D after G.
Pairs are permanently marked — never repeated.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pytest

from gigaevo.adversarial.composition_injection import (
    CompositionInjectionHook,
    _rename_entrypoint,
)
from gigaevo.adversarial.opponent_provider import OpponentProgram
from gigaevo.programs.program import Program


@pytest.fixture
def d_provider():
    return AsyncMock()


@pytest.fixture
def g_storage():
    mock = AsyncMock()
    mock.get_all.return_value = []
    return mock


@pytest.fixture
def dg_tracker():
    mock = AsyncMock()
    # Default: no D recorded for any G.
    mock.get_best_d_for_g.return_value = None
    mock.is_pair_injected.return_value = False
    return mock


@pytest.fixture
def hook(d_provider, g_storage, dg_tracker):
    return CompositionInjectionHook(
        d_provider=d_provider,
        g_storage=g_storage,
        dg_tracker=dg_tracker,
    )


# ===================================================================
# Test: _rename_entrypoint
# ===================================================================


class TestRenameEntrypoint:
    def test_renames_first_def(self):
        code = "def entrypoint():\n    return 1\n"
        result = _rename_entrypoint(code, "_g_entrypoint")
        assert "def _g_entrypoint(" in result
        assert "def entrypoint(" not in result

    def test_only_renames_top_level(self):
        # Inner `def entrypoint` (indented) should not be touched.
        code = (
            "def entrypoint():\n"
            "    def entrypoint():\n"
            "        return 1\n"
            "    return entrypoint\n"
        )
        result = _rename_entrypoint(code, "_g_entrypoint")
        # Top-level renamed once.
        assert result.count("def _g_entrypoint(") == 1
        # Inner one (indented) preserved.
        assert "    def entrypoint(" in result


# ===================================================================
# Test: _compose
# ===================================================================


class TestCompose:
    def test_compose_returns_executable_program(self):
        """The composed program defines entrypoint() that runs D(G())."""
        g_code = (
            "import numpy as np\n"
            "def entrypoint():\n"
            "    return np.array([[1.0, 2.0], [3.0, 4.0]])\n"
        )
        d_code = (
            "import numpy as np\n"
            "def entrypoint():\n"
            "    def improve(pts):\n"
            "        return pts * 2.0\n"
            "    return improve\n"
        )
        composed = CompositionInjectionHook._compose(g_code, d_code)

        assert "def _g_entrypoint(" in composed
        assert "def _d_entrypoint(" in composed
        assert "def entrypoint(" in composed

        ns: dict = {}
        exec(composed, ns)
        result = ns["entrypoint"]()
        np.testing.assert_array_equal(result, np.array([[2.0, 4.0], [6.0, 8.0]]))

    def test_compose_does_not_hardcode_data(self):
        """No `_G_POINTS` constant — composition is code-level."""
        g_code = "def entrypoint():\n    return [[1.0, 2.0]]\n"
        d_code = "def entrypoint():\n    return lambda pts: pts\n"
        composed = CompositionInjectionHook._compose(g_code, d_code)
        assert "_G_POINTS" not in composed


# ===================================================================
# Test: inject_all — empty cases
# ===================================================================


class TestEmptyCases:
    @pytest.mark.asyncio
    async def test_empty_archive_returns_empty_list(self, hook, g_storage):
        g_storage.get_all.return_value = []
        out = await hook.inject_all()
        assert out == []

    @pytest.mark.asyncio
    async def test_no_recorded_d_for_any_g_injects_nothing(
        self, hook, g_storage, dg_tracker, d_provider
    ):
        g_storage.get_all.return_value = [
            Program(code="def entrypoint(): pass", metadata={}),
            Program(code="def entrypoint(): pass", metadata={}),
        ]
        dg_tracker.get_best_d_for_g.return_value = None
        out = await hook.inject_all()
        assert out == []
        g_storage.add.assert_not_called()
        d_provider.get_programs_by_ids.assert_not_called()


# ===================================================================
# Test: inject_all — happy path
# ===================================================================


class TestInjectAllHappyPath:
    @pytest.mark.asyncio
    async def test_injects_one_per_g_with_recorded_d(
        self, hook, g_storage, d_provider, dg_tracker
    ):
        g1 = Program(
            code="import numpy as np\ndef entrypoint():\n    return np.zeros((3, 2))\n",
            metadata={},
        )
        g2 = Program(
            code="import numpy as np\ndef entrypoint():\n    return np.ones((3, 2))\n",
            metadata={},
        )
        g_storage.get_all.return_value = [g1, g2]

        # Each G has a different best-D recorded.
        async def best_for(g_id):
            return ("d-A", 0.123) if g_id == g1.id else ("d-B", 0.456)

        dg_tracker.get_best_d_for_g.side_effect = best_for
        dg_tracker.is_pair_injected.return_value = False

        d_a = OpponentProgram(
            program_id="d-A",
            code=(
                "import numpy as np\n"
                "def entrypoint():\n"
                "    return lambda pts: pts + 1.0\n"
            ),
            fitness=0.7,
        )
        d_b = OpponentProgram(
            program_id="d-B",
            code=(
                "import numpy as np\n"
                "def entrypoint():\n"
                "    return lambda pts: pts * 2.0\n"
            ),
            fitness=0.9,
        )

        async def fetch_d(ids):
            return [d_a if ids[0] == "d-A" else d_b]

        d_provider.get_programs_by_ids.side_effect = fetch_d

        out = await hook.inject_all()
        assert len(out) == 2
        assert g_storage.add.call_count == 2

        # Each composed program is well-formed and metadata is right.
        injected_progs = [c.args[0] for c in g_storage.add.call_args_list]
        d_sources = {p.metadata["d_source_id"] for p in injected_progs}
        g_sources = {p.metadata["g_source_id"] for p in injected_progs}
        assert d_sources == {"d-A", "d-B"}
        assert g_sources == {g1.id, g2.id}
        for p in injected_progs:
            assert p.metadata["mutation_type"] == "d_improvement"
            assert "tracked_delta" in p.metadata

        # Both pairs marked permanently.
        marked = {
            (c.args[0], c.args[1]) for c in dg_tracker.mark_pair_injected.call_args_list
        }
        assert marked == {("d-A", g1.id), ("d-B", g2.id)}


# ===================================================================
# Test: inject_all — lineage propagation (I-17 fix)
#
# Previously the hook constructed Program(code=..., metadata={...}) with no
# lineage, so every injected program landed as generation=1, iteration=0,
# is_root=True. That corrupted every `is_root` / generation slice of the
# archive. Injected programs must inherit lineage from their G parent: the
# composed code is a descendant of G, so generation = G.generation + 1 and
# parents = [g_id]. D is NOT added to parents — D lives in a separate Redis
# DB (G's graph walker cannot resolve it); the D reference stays in
# metadata.d_source_id.
# ===================================================================


class TestInjectedLineage:
    @pytest.mark.asyncio
    async def test_injected_program_inherits_generation_from_g_plus_one(
        self, hook, g_storage, d_provider, dg_tracker
    ):
        from gigaevo.programs.program import Lineage

        g = Program(
            code="def entrypoint():\n    return 0\n",
            lineage=Lineage(parents=["any-parent"], mutation="seed", generation=7),
        )
        g_storage.get_all.return_value = [g]
        dg_tracker.get_best_d_for_g.return_value = ("d-1", 0.1)
        dg_tracker.is_pair_injected.return_value = False
        d_provider.get_programs_by_ids.return_value = [
            OpponentProgram(
                program_id="d-1",
                code="def entrypoint():\n    return lambda x: x\n",
                fitness=0.5,
            )
        ]

        await hook.inject_all()

        injected = g_storage.add.call_args.args[0]
        assert injected.lineage.generation == 8, (
            f"generation must be G.generation + 1, got {injected.lineage.generation}"
        )

    @pytest.mark.asyncio
    async def test_injected_program_is_not_root_and_parents_is_g_only(
        self, hook, g_storage, d_provider, dg_tracker
    ):
        g = Program(code="def entrypoint():\n    return 0\n", metadata={})
        g_storage.get_all.return_value = [g]
        dg_tracker.get_best_d_for_g.return_value = ("d-1", 0.1)
        dg_tracker.is_pair_injected.return_value = False
        d_provider.get_programs_by_ids.return_value = [
            OpponentProgram(
                program_id="d-1",
                code="def entrypoint():\n    return lambda x: x\n",
                fitness=0.5,
            )
        ]

        await hook.inject_all()

        injected = g_storage.add.call_args.args[0]
        # D is NOT in parents (it lives in a different Redis DB); only G.
        assert injected.lineage.parents == [g.id]
        assert "d-1" not in injected.lineage.parents
        assert injected.is_root is False
        assert injected.lineage.mutation == "d_improvement"
        # D reference still accessible via metadata.
        assert injected.metadata["d_source_id"] == "d-1"
        assert injected.metadata["g_source_id"] == g.id

    # -----------------------------------------------------------------
    # I-18: Program.create_child silently defaulted iteration=0, so every
    # injected program landed at step=0 on the frontier/per-iter stats
    # regardless of when in the run it was produced. That made watchdog
    # `comparison` plots look like the max-fitness line "starts at gen 0"
    # even though the real breakthrough was at lineage.generation=8.
    # Injected programs must inherit iteration from their G parent.
    # -----------------------------------------------------------------
    @pytest.mark.asyncio
    async def test_injected_program_iteration_matches_g_parent(
        self, hook, g_storage, d_provider, dg_tracker
    ):
        g = Program(code="def entrypoint():\n    return 0\n", metadata={})
        g.iteration = 12
        g_storage.get_all.return_value = [g]
        dg_tracker.get_best_d_for_g.return_value = ("d-1", 0.1)
        dg_tracker.is_pair_injected.return_value = False
        d_provider.get_programs_by_ids.return_value = [
            OpponentProgram(
                program_id="d-1",
                code="def entrypoint():\n    return lambda x: x\n",
                fitness=0.5,
            )
        ]

        await hook.inject_all()

        injected = g_storage.add.call_args.args[0]
        assert injected.iteration == g.iteration, (
            "injected program must inherit G's iteration (frontier/per-iter "
            f"stats are indexed by program.iteration); got {injected.iteration}, "
            f"expected {g.iteration}"
        )


# ===================================================================
# Test: inject_all — dedup
# ===================================================================


class TestDedup:
    @pytest.mark.asyncio
    async def test_skip_already_injected_pair(
        self, hook, g_storage, d_provider, dg_tracker
    ):
        g = Program(code="def entrypoint(): pass", metadata={})
        g_storage.get_all.return_value = [g]
        dg_tracker.get_best_d_for_g.return_value = ("d-1", 0.5)
        dg_tracker.is_pair_injected.return_value = True  # already done

        out = await hook.inject_all()
        assert out == []
        g_storage.add.assert_not_called()
        d_provider.get_programs_by_ids.assert_not_called()
        dg_tracker.mark_pair_injected.assert_not_called()

    @pytest.mark.asyncio
    async def test_partial_dedup(self, hook, g_storage, d_provider, dg_tracker):
        """First G already injected, second G fresh — only second composes."""
        g1 = Program(code="def entrypoint(): pass", metadata={})
        g2 = Program(
            code="import numpy as np\ndef entrypoint():\n    return np.zeros((2, 2))\n",
            metadata={},
        )
        g_storage.get_all.return_value = [g1, g2]
        dg_tracker.get_best_d_for_g.return_value = ("d-1", 0.4)

        async def is_injected(d_id, g_id):
            return g_id == g1.id

        dg_tracker.is_pair_injected.side_effect = is_injected

        d_provider.get_programs_by_ids.return_value = [
            OpponentProgram(
                program_id="d-1",
                code=(
                    "import numpy as np\n"
                    "def entrypoint():\n"
                    "    return lambda pts: pts + 0.1\n"
                ),
                fitness=0.5,
            )
        ]

        out = await hook.inject_all()
        assert len(out) == 1
        g_storage.add.assert_called_once()
        dg_tracker.mark_pair_injected.assert_called_once_with("d-1", g2.id)


# ===================================================================
# Test: inject_all — D missing from archive
# ===================================================================


class TestDMissing:
    @pytest.mark.asyncio
    async def test_skip_when_d_no_longer_in_archive(
        self, hook, g_storage, d_provider, dg_tracker
    ):
        g = Program(code="def entrypoint(): pass", metadata={})
        g_storage.get_all.return_value = [g]
        dg_tracker.get_best_d_for_g.return_value = ("d-stale", 0.3)
        dg_tracker.is_pair_injected.return_value = False
        d_provider.get_programs_by_ids.return_value = []  # D evicted

        out = await hook.inject_all()
        assert out == []
        g_storage.add.assert_not_called()
        # We should NOT mark the pair as injected — D was simply missing.
        dg_tracker.mark_pair_injected.assert_not_called()


# ===================================================================
# Test: __call__ delegates to inject_all
# ===================================================================


class TestCallDelegates:
    @pytest.mark.asyncio
    async def test_call_invokes_inject_all(self, hook, g_storage):
        g_storage.get_all.return_value = []
        await hook()
        g_storage.get_all.assert_called_once()
