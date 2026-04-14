"""Tests for CompositionInjectionHook — code composition and delta gating."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from gigaevo.adversarial.composition_injection import CompositionInjectionHook
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
    return AsyncMock()


@pytest.fixture
def hook(d_provider, g_storage):
    return CompositionInjectionHook(
        d_provider=d_provider,
        g_storage=g_storage,
    )


@pytest.fixture
def hook_with_tracker(d_provider, g_storage, dg_tracker):
    return CompositionInjectionHook(
        d_provider=d_provider,
        g_storage=g_storage,
        dg_tracker=dg_tracker,
    )


# ===================================================================
# Test 1: _compose_g_program produces valid G-style Python code
# ===================================================================


class TestComposeGProgram:
    def test_compose_g_program_returns_python_string(self):
        """_compose_g_program takes d_code and g_points and returns a Python string."""
        d_code = "import numpy as np\n\ndef entrypoint():\n    def improve(pts):\n        return pts * 1.1\n    return improve\n"
        g_points = [[0.0, 0.0]] * 11
        result = CompositionInjectionHook._compose_g_program(d_code, g_points)
        assert isinstance(result, str)
        assert "def entrypoint():" in result
        assert "_d_entrypoint" in result

    def test_compose_g_program_renames_d_entrypoint(self):
        """D's entrypoint is renamed to _d_entrypoint in the composed code."""
        d_code = "def entrypoint():\n    def improve(pts):\n        return pts\n    return improve\n"
        g_points = [[0.0, 0.0]] * 11
        result = CompositionInjectionHook._compose_g_program(d_code, g_points)
        assert "def _d_entrypoint(" in result
        # The original entrypoint should not appear unmodified
        lines = result.split("\n")
        d_section_lines = [
            line
            for line in lines
            if "def entrypoint" in line and "_d_entrypoint" not in line
        ]
        # The wrapper's entrypoint should be there once
        assert len(d_section_lines) == 1

    def test_compose_g_program_contains_g_points(self):
        """The composed code embeds G's point configuration."""
        d_code = "def entrypoint():\n    return lambda pts: pts\n"
        g_points = [[1.0, 2.0], [3.0, 4.0]]
        result = CompositionInjectionHook._compose_g_program(d_code, g_points)
        assert "_G_POINTS" in result
        assert "1.0" in result
        assert "2.0" in result

    def test_compose_g_program_executable(self):
        """The composed program can be exec'd and defines entrypoint() returning ndarray."""
        d_code = (
            "import numpy as np\n"
            "\n"
            "def entrypoint():\n"
            "    def improve(pts):\n"
            "        return pts * 1.1\n"
            "    return improve\n"
        )
        g_points = [[float(i), float(i)] for i in range(11)]
        composed = CompositionInjectionHook._compose_g_program(d_code, g_points)

        namespace: dict = {}
        exec(composed, namespace)
        result = namespace["entrypoint"]()
        assert isinstance(result, np.ndarray)
        assert result.shape == (11, 2)


# ===================================================================
# Test 3-4: inject() delta gating
# ===================================================================


class TestInjectDeltaGating:
    @pytest.mark.asyncio
    async def test_inject_only_when_output_differs(self, hook, d_provider, g_storage):
        """inject() creates a Program only when D improves G (output differs)."""
        d_provider.get_top_k.return_value = [
            OpponentProgram(
                program_id="d-1",
                code=(
                    "import numpy as np\n"
                    "def entrypoint():\n"
                    "    def improve(pts):\n"
                    "        return pts * 1.5\n"
                    "    return improve\n"
                ),
                fitness=0.8,
            )
        ]

        g_points = np.array([[float(i), float(i)] for i in range(11)])
        g_prog = Program(
            code="import numpy as np\ndef entrypoint():\n    return np.array([[float(i), float(i)] for i in range(11)])\n",
            metadata={},
        )
        g_storage.get_all.return_value = [g_prog]

        # Mock run_exec_runner: first call returns G's points, second returns improved
        improved_points = g_points * 1.5
        with patch(
            "gigaevo.adversarial.composition_injection.run_exec_runner",
            new_callable=AsyncMock,
        ) as mock_runner:
            mock_runner.side_effect = [
                (g_points.tolist(), b"", ""),  # G execution
                (improved_points.tolist(), b"", ""),  # Composed execution
            ]
            result = await hook.inject()

        assert result is not None
        g_storage.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_inject_returns_none_when_no_improvement(
        self, hook, d_provider, g_storage
    ):
        """inject() returns None when composed output equals original G output."""
        d_provider.get_top_k.return_value = [
            OpponentProgram(
                program_id="d-1",
                code="def entrypoint():\n    return lambda pts: pts\n",
                fitness=0.5,
            )
        ]

        g_points = [[1.0, 2.0]] * 11
        g_prog = Program(
            code="def entrypoint():\n    return [[1.0, 2.0]] * 11\n",
            metadata={},
        )
        g_storage.get_all.return_value = [g_prog]

        with patch(
            "gigaevo.adversarial.composition_injection.run_exec_runner",
            new_callable=AsyncMock,
        ) as mock_runner:
            mock_runner.side_effect = [
                (g_points, b"", ""),  # G execution
                (g_points, b"", ""),  # Composed execution (same = no improvement)
            ]
            result = await hook.inject()

        assert result is None
        g_storage.add.assert_not_called()


# ===================================================================
# Test 5-6: inject() with empty archives
# ===================================================================


class TestInjectEmptyArchives:
    @pytest.mark.asyncio
    async def test_inject_returns_none_when_d_archive_empty(
        self, hook, d_provider, g_storage
    ):
        """inject() returns None when D's archive is empty."""
        d_provider.get_top_k.return_value = []
        result = await hook.inject()
        assert result is None
        g_storage.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_inject_returns_none_when_g_archive_empty(
        self, hook, d_provider, g_storage
    ):
        """inject() returns None when G archive is empty (no G programs to improve)."""
        d_provider.get_top_k.return_value = [
            OpponentProgram(
                program_id="d-1", code="def entrypoint(): pass", fitness=0.5
            )
        ]
        g_storage.get_all.return_value = []
        result = await hook.inject()
        assert result is None
        g_storage.add.assert_not_called()


# ===================================================================
# Test 7: Injected program metadata
# ===================================================================


class TestInjectedProgramMetadata:
    @pytest.mark.asyncio
    async def test_injected_program_has_correct_metadata(
        self, hook, d_provider, g_storage
    ):
        """Injected program contains mutation_type, d_source_id, g_source_id metadata."""
        d_provider.get_top_k.return_value = [
            OpponentProgram(
                program_id="d-best-42",
                code=(
                    "import numpy as np\n"
                    "def entrypoint():\n"
                    "    def improve(pts):\n"
                    "        return pts * 2.0\n"
                    "    return improve\n"
                ),
                fitness=0.9,
            )
        ]

        g_points = np.array([[1.0, 1.0]] * 11)
        g_prog = Program(code="def entrypoint():\n    pass\n", metadata={})
        g_storage.get_all.return_value = [g_prog]

        improved = g_points * 2.0
        with patch(
            "gigaevo.adversarial.composition_injection.run_exec_runner",
            new_callable=AsyncMock,
        ) as mock_runner:
            mock_runner.side_effect = [
                (g_points.tolist(), b"", ""),
                (improved.tolist(), b"", ""),
            ]
            result = await hook.inject()

        assert result is not None
        injected = g_storage.add.call_args[0][0]
        assert isinstance(injected, Program)
        assert injected.metadata["mutation_type"] == "d_improvement"
        assert injected.metadata["d_source_id"] == "d-best-42"
        assert injected.metadata["g_source_id"] == g_prog.id


# ===================================================================
# Test 8: _compose_g_program produces G-valid code
# ===================================================================


class TestComposeGProgramValidity:
    def test_composed_code_returns_float64_ndarray(self):
        """The composed code returns np.float64 ndarray."""
        d_code = (
            "import numpy as np\n"
            "\n"
            "def entrypoint():\n"
            "    def improve(pts):\n"
            "        return pts + 0.01\n"
            "    return improve\n"
        )
        g_points = [[float(i) * 0.1, float(i) * 0.1] for i in range(11)]
        composed = CompositionInjectionHook._compose_g_program(d_code, g_points)

        namespace: dict = {}
        exec(composed, namespace)
        result = namespace["entrypoint"]()

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert result.shape == (11, 2)


# ===================================================================
# Test: __call__ delegates to inject()
# ===================================================================


class TestCallDelegatesToInject:
    @pytest.mark.asyncio
    async def test_call_invokes_inject(self, hook, d_provider, g_storage):
        """__call__ delegates to inject()."""
        d_provider.get_top_k.return_value = []
        await hook()
        d_provider.get_top_k.assert_called_once()


# ===================================================================
# Test: dg_tracker recording
# ===================================================================


class TestDGTrackerRecording:
    @pytest.mark.asyncio
    async def test_tracker_called_on_successful_injection(
        self, hook_with_tracker, d_provider, g_storage, dg_tracker
    ):
        """dg_tracker.record_improvement is called with d_id, g_id, delta when injection succeeds."""
        d_provider.get_top_k.return_value = [
            OpponentProgram(
                program_id="d-1",
                code=(
                    "import numpy as np\n"
                    "def entrypoint():\n"
                    "    def improve(pts):\n"
                    "        return pts * 3.0\n"
                    "    return improve\n"
                ),
                fitness=0.7,
            )
        ]

        g_points = np.array([[1.0, 1.0]] * 11)
        g_prog = Program(code="def entrypoint():\n    pass\n", metadata={})
        g_storage.get_all.return_value = [g_prog]

        improved = g_points * 3.0
        with patch(
            "gigaevo.adversarial.composition_injection.run_exec_runner",
            new_callable=AsyncMock,
        ) as mock_runner:
            mock_runner.side_effect = [
                (g_points.tolist(), b"", ""),
                (improved.tolist(), b"", ""),
            ]
            await hook_with_tracker.inject()

        dg_tracker.record_improvement.assert_called_once()
        call_kwargs = dg_tracker.record_improvement.call_args
        assert call_kwargs[1]["d_id"] == "d-1"
        assert call_kwargs[1]["g_id"] == g_prog.id
        assert call_kwargs[1]["delta"] > 0  # positive delta when improvement occurred

    @pytest.mark.asyncio
    async def test_tracker_none_inject_succeeds_without_recording(
        self, hook, d_provider, g_storage
    ):
        """When dg_tracker is None, inject() succeeds without recording (no error)."""
        d_provider.get_top_k.return_value = [
            OpponentProgram(
                program_id="d-1",
                code=(
                    "import numpy as np\n"
                    "def entrypoint():\n"
                    "    def improve(pts):\n"
                    "        return pts * 2.0\n"
                    "    return improve\n"
                ),
                fitness=0.6,
            )
        ]

        g_points = np.array([[1.0, 1.0]] * 11)
        g_prog = Program(code="def entrypoint():\n    pass\n", metadata={})
        g_storage.get_all.return_value = [g_prog]

        improved = g_points * 2.0
        with patch(
            "gigaevo.adversarial.composition_injection.run_exec_runner",
            new_callable=AsyncMock,
        ) as mock_runner:
            mock_runner.side_effect = [
                (g_points.tolist(), b"", ""),
                (improved.tolist(), b"", ""),
            ]
            result = await hook.inject()

        assert result is not None  # injection succeeded
        g_storage.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_tracker_not_called_when_no_improvement(
        self, hook_with_tracker, d_provider, g_storage, dg_tracker
    ):
        """When inject() fails (no improvement), dg_tracker.record_improvement is NOT called."""
        d_provider.get_top_k.return_value = [
            OpponentProgram(
                program_id="d-1",
                code="def entrypoint():\n    return lambda pts: pts\n",
                fitness=0.5,
            )
        ]

        g_points = [[1.0, 2.0]] * 11
        g_prog = Program(code="def entrypoint():\n    pass\n", metadata={})
        g_storage.get_all.return_value = [g_prog]

        with patch(
            "gigaevo.adversarial.composition_injection.run_exec_runner",
            new_callable=AsyncMock,
        ) as mock_runner:
            mock_runner.side_effect = [
                (g_points, b"", ""),
                (g_points, b"", ""),  # same output = no improvement
            ]
            result = await hook_with_tracker.inject()

        assert result is None
        dg_tracker.record_improvement.assert_not_called()

    @pytest.mark.asyncio
    async def test_tracker_recording_exception_caught(
        self, hook_with_tracker, d_provider, g_storage, dg_tracker
    ):
        """When dg_tracker.record_improvement raises, injection still succeeds."""
        dg_tracker.record_improvement.side_effect = RuntimeError(
            "Redis connection lost"
        )
        d_provider.get_top_k.return_value = [
            OpponentProgram(
                program_id="d-1",
                code=(
                    "import numpy as np\n"
                    "def entrypoint():\n"
                    "    def improve(pts):\n"
                    "        return pts * 1.5\n"
                    "    return improve\n"
                ),
                fitness=0.7,
            )
        ]

        g_points = np.array([[1.0, 1.0]] * 11)
        g_prog = Program(code="def entrypoint():\n    pass\n", metadata={})
        g_storage.get_all.return_value = [g_prog]

        improved = g_points * 1.5
        with patch(
            "gigaevo.adversarial.composition_injection.run_exec_runner",
            new_callable=AsyncMock,
        ) as mock_runner:
            mock_runner.side_effect = [
                (g_points.tolist(), b"", ""),
                (improved.tolist(), b"", ""),
            ]
            result = await hook_with_tracker.inject()

        # Injection still succeeded despite tracker error
        assert result is not None
        g_storage.add.assert_called_once()
