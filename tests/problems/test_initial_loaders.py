"""Tests for gigaevo.problems.initial_loaders program loading utilities."""

from __future__ import annotations

from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from gigaevo.problems.initial_loaders import (
    DirectoryProgramLoader,
    RedisTopProgramsLoader,
)

# ===================================================================
# DirectoryProgramLoader
# ===================================================================


class TestDirectoryProgramLoader:
    @pytest.mark.asyncio
    async def test_loads_programs_from_directory(self):
        """Test loading Python files from initial_programs directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            problem_dir = Path(tmpdir)
            initial_dir = problem_dir / "initial_programs"
            initial_dir.mkdir()

            # Create test program files
            (initial_dir / "program1.py").write_text("def solve():\n    return 42")
            (initial_dir / "program2.py").write_text("def solve():\n    return 99")

            loader = DirectoryProgramLoader(problem_dir)
            storage = AsyncMock()
            storage.add = AsyncMock()

            programs = await loader.load(storage)

            assert len(programs) == 2
            assert storage.add.call_count == 2

            # Verify programs have correct metadata
            for prog in programs:
                assert prog.metadata["source"] == "initial_program"
                assert "strategy_name" in prog.metadata
                assert "file_path" in prog.metadata
                assert prog.metadata["iteration"] == 0

    @pytest.mark.asyncio
    async def test_loads_specific_strategy_names(self):
        """Test that strategy_name metadata uses file stem."""
        with tempfile.TemporaryDirectory() as tmpdir:
            problem_dir = Path(tmpdir)
            initial_dir = problem_dir / "initial_programs"
            initial_dir.mkdir()

            (initial_dir / "greedy.py").write_text("pass")
            (initial_dir / "random.py").write_text("pass")

            loader = DirectoryProgramLoader(problem_dir)
            storage = AsyncMock()
            storage.add = AsyncMock()

            programs = await loader.load(storage)

            strategy_names = {p.metadata["strategy_name"] for p in programs}
            assert strategy_names == {"greedy", "random"}

    @pytest.mark.asyncio
    async def test_returns_empty_if_no_initial_dir(self):
        """Test graceful handling when initial_programs dir doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            problem_dir = Path(tmpdir)
            # Don't create initial_programs directory

            loader = DirectoryProgramLoader(problem_dir)
            storage = AsyncMock()

            programs = await loader.load(storage)

            assert programs == []
            storage.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_invalid_python_files(self):
        """Test that exceptions during program loading are silently skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            problem_dir = Path(tmpdir)
            initial_dir = problem_dir / "initial_programs"
            initial_dir.mkdir()

            # Valid file
            (initial_dir / "good.py").write_text("def solve():\n    return 42")
            # Invalid file (will cause exception during storage.add)
            (initial_dir / "bad.py").write_text("syntax error {{{{")

            loader = DirectoryProgramLoader(problem_dir)
            storage = AsyncMock()

            # Make add() raise for bad.py
            async def side_effect(prog):
                if "bad" in prog.metadata.get("file_path", ""):
                    raise ValueError("bad program")

            storage.add = AsyncMock(side_effect=side_effect)

            programs = await loader.load(storage)

            # Should only load the good file
            assert len(programs) == 1

    @pytest.mark.asyncio
    async def test_program_code_matches_file_content(self):
        """Test that loaded program code matches the source file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            problem_dir = Path(tmpdir)
            initial_dir = problem_dir / "initial_programs"
            initial_dir.mkdir()

            code = """def solve():
    x = 42
    y = x + 1
    return y"""
            (initial_dir / "test.py").write_text(code)

            loader = DirectoryProgramLoader(problem_dir)
            storage = AsyncMock()
            storage.add = AsyncMock()

            programs = await loader.load(storage)

            assert len(programs) == 1
            assert programs[0].code == code

    @pytest.mark.asyncio
    async def test_stores_file_path_in_metadata(self):
        """Test that file_path metadata is absolute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            problem_dir = Path(tmpdir)
            initial_dir = problem_dir / "initial_programs"
            initial_dir.mkdir()

            (initial_dir / "prog.py").write_text("pass")

            loader = DirectoryProgramLoader(problem_dir)
            storage = AsyncMock()
            storage.add = AsyncMock()

            programs = await loader.load(storage)

            assert len(programs) == 1
            file_path = programs[0].metadata["file_path"]
            assert file_path == str(initial_dir / "prog.py")

    @pytest.mark.asyncio
    async def test_only_loads_py_files(self):
        """Test that only .py files are loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            problem_dir = Path(tmpdir)
            initial_dir = problem_dir / "initial_programs"
            initial_dir.mkdir()

            (initial_dir / "prog.py").write_text("pass")
            (initial_dir / "readme.txt").write_text("test")
            (initial_dir / "data.json").write_text("{}")

            loader = DirectoryProgramLoader(problem_dir)
            storage = AsyncMock()
            storage.add = AsyncMock()

            programs = await loader.load(storage)

            assert len(programs) == 1
            assert programs[0].metadata["strategy_name"] == "prog"


# ===================================================================
# RedisTopProgramsLoader
# ===================================================================


class TestRedisTopProgramsLoader:
    def test_init_stores_config(self):
        """Test that constructor stores all config parameters."""
        loader = RedisTopProgramsLoader(
            source_host="localhost",
            source_port=6379,
            source_db=1,
            key_prefix="exp:v1",
            metric_key="fitness",
            higher_is_better=True,
            top_n=25,
            max_connections=100,
            connection_pool_timeout=60.0,
            health_check_interval=120,
        )

        assert loader.source_host == "localhost"
        assert loader.source_port == 6379
        assert loader.source_db == 1
        assert loader.key_prefix == "exp:v1"
        assert loader.metric_key == "fitness"
        assert loader.higher_is_better is True
        assert loader.top_n == 25
        assert loader.max_connections == 100
        assert loader.connection_pool_timeout == 60.0
        assert loader.health_check_interval == 120

    @pytest.mark.asyncio
    async def test_returns_empty_if_no_programs_in_source(self):
        """Test graceful handling when source storage is empty."""
        loader = RedisTopProgramsLoader(
            source_host="localhost",
            source_port=6379,
            source_db=1,
            key_prefix="test",
            metric_key="fitness",
            higher_is_better=True,
        )

        # Mock source storage to return empty list
        with patch(
            "gigaevo.problems.initial_loaders.RedisProgramStorage"
        ) as MockStorage:
            source = AsyncMock()
            source.get_all = AsyncMock(return_value=[])
            source.close = AsyncMock()
            MockStorage.return_value = source

            dest = AsyncMock()
            result = await loader.load(dest)

            assert result == []
            source.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_filters_programs_without_metric(self):
        """Test that programs without the metric key are filtered out."""
        loader = RedisTopProgramsLoader(
            source_host="localhost",
            source_port=6379,
            source_db=1,
            key_prefix="test",
            metric_key="fitness",
            higher_is_better=True,
            top_n=10,
        )

        # Create programs: some with metric, some without
        prog_with = MagicMock()
        prog_with.id = uuid.uuid4()
        prog_with.code = "def solve(): pass"
        prog_with.metrics = {"fitness": 0.9, "other": 1.0}
        prog_with.lineage.children = []
        prog_with.lineage.parents = []
        prog_with.stage_results = {}
        prog_with.metadata = {}

        prog_without = MagicMock()
        prog_without.id = uuid.uuid4()
        prog_without.code = "def solve(): pass"
        prog_without.metrics = {"other": 1.0}  # No fitness metric
        prog_without.lineage.children = []
        prog_without.lineage.parents = []
        prog_without.stage_results = {}
        prog_without.metadata = {}

        with patch(
            "gigaevo.problems.initial_loaders.RedisProgramStorage"
        ) as MockStorage:
            source = AsyncMock()
            source.get_all = AsyncMock(return_value=[prog_with, prog_without])
            source.close = AsyncMock()
            MockStorage.return_value = source

            dest = AsyncMock()
            dest.add = AsyncMock()

            result = await loader.load(dest)

            # Only prog_with should be added
            assert len(result) == 1
            assert dest.add.call_count == 1

    @pytest.mark.asyncio
    async def test_sorts_by_fitness_higher_is_better(self):
        """Test that programs are sorted by fitness in descending order."""
        loader = RedisTopProgramsLoader(
            source_host="localhost",
            source_port=6379,
            source_db=1,
            key_prefix="test",
            metric_key="fitness",
            higher_is_better=True,
            top_n=2,
        )

        # Create programs with different fitness values
        programs = []
        for i, fitness in enumerate([0.5, 0.9, 0.7]):
            prog = MagicMock()
            prog.id = uuid.uuid4()
            prog.code = "def solve(): pass"
            prog.metrics = {"fitness": fitness}
            prog.lineage.children = []
            prog.lineage.parents = []
            prog.stage_results = {}
            prog.metadata = {}
            programs.append(prog)

        with patch(
            "gigaevo.problems.initial_loaders.RedisProgramStorage"
        ) as MockStorage:
            source = AsyncMock()
            source.get_all = AsyncMock(return_value=programs)
            source.close = AsyncMock()
            MockStorage.return_value = source

            dest = AsyncMock()
            dest.add = AsyncMock()

            result = await loader.load(dest)

            # Should select top 2: 0.9 and 0.7
            assert len(result) == 2
            # Check that the highest fitness was selected
            assert dest.add.call_count == 2

    @pytest.mark.asyncio
    async def test_sorts_by_fitness_lower_is_better(self):
        """Test that programs are sorted correctly when lower is better."""
        loader = RedisTopProgramsLoader(
            source_host="localhost",
            source_port=6379,
            source_db=1,
            key_prefix="test",
            metric_key="loss",
            higher_is_better=False,  # Lower is better
            top_n=2,
        )

        # Create programs with different loss values
        programs = []
        for i, loss in enumerate([1.5, 0.2, 0.8]):
            prog = MagicMock()
            prog.id = uuid.uuid4()
            prog.code = "def solve(): pass"
            prog.metrics = {"loss": loss}
            prog.lineage.children = []
            prog.lineage.parents = []
            prog.stage_results = {}
            prog.metadata = {}
            programs.append(prog)

        with patch(
            "gigaevo.problems.initial_loaders.RedisProgramStorage"
        ) as MockStorage:
            source = AsyncMock()
            source.get_all = AsyncMock(return_value=programs)
            source.close = AsyncMock()
            MockStorage.return_value = source

            dest = AsyncMock()
            dest.add = AsyncMock()

            result = await loader.load(dest)

            # Should select top 2 (lowest losses): 0.2 and 0.8
            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_respects_top_n_limit(self):
        """Test that only top_n programs are selected."""
        loader = RedisTopProgramsLoader(
            source_host="localhost",
            source_port=6379,
            source_db=1,
            key_prefix="test",
            metric_key="fitness",
            higher_is_better=True,
            top_n=3,
        )

        # Create 10 programs
        programs = []
        for i in range(10):
            prog = MagicMock()
            prog.id = uuid.uuid4()
            prog.code = "def solve(): pass"
            prog.metrics = {"fitness": float(i)}
            prog.lineage.children = []
            prog.lineage.parents = []
            prog.stage_results = {}
            prog.metadata = {}
            programs.append(prog)

        with patch(
            "gigaevo.problems.initial_loaders.RedisProgramStorage"
        ) as MockStorage:
            source = AsyncMock()
            source.get_all = AsyncMock(return_value=programs)
            source.close = AsyncMock()
            MockStorage.return_value = source

            dest = AsyncMock()
            dest.add = AsyncMock()

            result = await loader.load(dest)

            assert len(result) == 3

    @pytest.mark.asyncio
    async def test_sets_metadata_for_loaded_programs(self):
        """Test that loaded programs have correct metadata set."""
        loader = RedisTopProgramsLoader(
            source_host="redis.example.com",
            source_port=6379,
            source_db=2,
            key_prefix="test",
            metric_key="fitness",
            higher_is_better=True,
            top_n=1,
        )

        prog = MagicMock()
        original_id = uuid.uuid4()
        prog.id = original_id
        prog.code = "def solve(): pass"
        prog.metrics = {"fitness": 0.9}
        prog.lineage.children = []
        prog.lineage.parents = []
        prog.stage_results = {}
        prog.metadata = {"existing": "value"}

        with patch(
            "gigaevo.problems.initial_loaders.RedisProgramStorage"
        ) as MockStorage:
            source = AsyncMock()
            source.get_all = AsyncMock(return_value=[prog])
            source.close = AsyncMock()
            MockStorage.return_value = source

            dest = AsyncMock()
            dest.add = AsyncMock()

            result = await loader.load(dest)

            assert len(result) == 1
            loaded = result[0]
            assert loaded.metadata["source"] == "redis_selection"
            assert loaded.metadata["source_db"] == 2
            assert loaded.metadata["selection_rank"] == 1
            assert loaded.metadata["original_id"] == original_id
            assert loaded.metadata["iteration"] == 0

    @pytest.mark.asyncio
    async def test_closes_source_even_on_error(self):
        """Test that source storage is closed even if error occurs."""
        loader = RedisTopProgramsLoader(
            source_host="localhost",
            source_port=6379,
            source_db=1,
            key_prefix="test",
            metric_key="fitness",
            higher_is_better=True,
        )

        with patch(
            "gigaevo.problems.initial_loaders.RedisProgramStorage"
        ) as MockStorage:
            source = AsyncMock()
            source.get_all = AsyncMock(side_effect=RuntimeError("connection failed"))
            source.close = AsyncMock()
            MockStorage.return_value = source

            dest = AsyncMock()

            with pytest.raises(RuntimeError):
                await loader.load(dest)

            # Verify that close was still called
            source.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_close_error_gracefully(self):
        """Test that errors during close don't propagate."""
        loader = RedisTopProgramsLoader(
            source_host="localhost",
            source_port=6379,
            source_db=1,
            key_prefix="test",
            metric_key="fitness",
            higher_is_better=True,
        )

        with patch(
            "gigaevo.problems.initial_loaders.RedisProgramStorage"
        ) as MockStorage:
            source = AsyncMock()
            source.get_all = AsyncMock(return_value=[])
            source.close = AsyncMock(side_effect=RuntimeError("close failed"))
            MockStorage.return_value = source

            dest = AsyncMock()

            # Should not raise even though close() fails
            result = await loader.load(dest)
            assert result == []

    @pytest.mark.asyncio
    async def test_uses_sentinel_for_missing_metrics(self):
        """Test that sentinel values are used for programs without the metric."""
        loader = RedisTopProgramsLoader(
            source_host="localhost",
            source_port=6379,
            source_db=1,
            key_prefix="test",
            metric_key="fitness",
            higher_is_better=True,
            top_n=10,
        )

        # One program with metric, one without
        prog_with = MagicMock()
        prog_with.id = uuid.uuid4()
        prog_with.code = "pass"
        prog_with.metrics = {"fitness": 0.5}
        prog_with.lineage.children = []
        prog_with.lineage.parents = []
        prog_with.stage_results = {}
        prog_with.metadata = {}

        prog_without = MagicMock()
        prog_without.id = uuid.uuid4()
        prog_without.code = "pass"
        prog_without.metrics = {}
        prog_without.lineage.children = []
        prog_without.lineage.parents = []
        prog_without.stage_results = {}
        prog_without.metadata = {}

        with patch(
            "gigaevo.problems.initial_loaders.RedisProgramStorage"
        ) as MockStorage:
            source = AsyncMock()
            source.get_all = AsyncMock(return_value=[prog_with, prog_without])
            source.close = AsyncMock()
            MockStorage.return_value = source

            dest = AsyncMock()
            dest.add = AsyncMock()

            result = await loader.load(dest)

            # Filters out prog_without since it has no metric
            assert len(result) == 1
