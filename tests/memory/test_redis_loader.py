"""Tests for redis_loader — loading Program objects from a live Redis DB."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from gigaevo.database.redis import RedisProgramStorageConfig
from gigaevo.memory.ideas_tracker.redis_loader import load_programs_from_redis
from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS, Lineage, Program
from gigaevo.programs.program_state import ProgramState

_PATCH_TARGET = "gigaevo.memory.ideas_tracker.redis_loader.RedisProgramStorage"


def _make_program(
    *,
    program_id: str = "11111111-1111-1111-1111-111111111111",
    fitness: float = 0.75,
    is_valid: float = 1.0,
) -> Program:
    return Program(
        id=program_id,
        code="def solve(): return 42",
        state=ProgramState.DONE,
        lineage=Lineage(parents=["00000000-0000-0000-0000-000000000000"], generation=2),
        metrics={"fitness": fitness, "is_valid": is_valid},
    )


def _mock_storage(return_programs: list[Program]) -> tuple[MagicMock, MagicMock]:
    """Return (MockStorageClass, mock_instance) with get_all pre-configured."""
    mock_instance = MagicMock()
    mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
    mock_instance.__aexit__ = AsyncMock(return_value=None)
    mock_instance.get_all = AsyncMock(return_value=return_programs)
    return MagicMock(return_value=mock_instance), mock_instance


class TestLoadProgramsFromRedis:
    def test_returns_programs_from_redis(self) -> None:
        """load_programs_from_redis returns all programs produced by the storage."""
        programs_in_db = [
            _make_program(
                program_id="aaaaaaaa-0000-0000-0000-000000000000", fitness=0.8
            ),
            _make_program(
                program_id="bbbbbbbb-0000-0000-0000-000000000000", fitness=0.5
            ),
        ]
        MockStorage, _ = _mock_storage(programs_in_db)
        with patch(_PATCH_TARGET, MockStorage):
            result = load_programs_from_redis(
                host="localhost", port=6379, db=0, prefix="chains/test/run"
            )

        assert len(result) == 2
        ids = {p.id for p in result}
        assert "aaaaaaaa-0000-0000-0000-000000000000" in ids
        assert "bbbbbbbb-0000-0000-0000-000000000000" in ids

    def test_builds_correct_redis_url(self) -> None:
        """load_programs_from_redis constructs the right redis_url and key_prefix."""
        MockStorage, _ = _mock_storage([])
        with patch(_PATCH_TARGET, MockStorage):
            load_programs_from_redis(
                host="10.0.0.1", port=6380, db=3, prefix="chains/hover/static"
            )

        config_arg: RedisProgramStorageConfig = MockStorage.call_args[0][0]
        assert config_arg.redis_url == "redis://10.0.0.1:6380/3"
        assert config_arg.key_prefix == "chains/hover/static"

    def test_read_only_mode(self) -> None:
        """Storage is created in read_only mode to avoid acquiring the distributed lock."""
        MockStorage, _ = _mock_storage([])
        with patch(_PATCH_TARGET, MockStorage):
            load_programs_from_redis()

        config_arg: RedisProgramStorageConfig = MockStorage.call_args[0][0]
        assert config_arg.read_only is True

    def test_get_all_called_with_exclude_stage_results(self) -> None:
        """stage_results are excluded for performance."""
        MockStorage, mock_instance = _mock_storage([])
        with patch(_PATCH_TARGET, MockStorage):
            load_programs_from_redis()

        mock_instance.get_all.assert_awaited_once_with(exclude=EXCLUDE_STAGE_RESULTS)

    def test_empty_db_returns_empty_list(self) -> None:
        """Empty Redis DB → empty list (no crash)."""
        MockStorage, _ = _mock_storage([])
        with patch(_PATCH_TARGET, MockStorage):
            result = load_programs_from_redis()

        assert result == []

    def test_defaults(self) -> None:
        """Default args produce localhost:6379/0 with empty prefix."""
        MockStorage, _ = _mock_storage([])
        with patch(_PATCH_TARGET, MockStorage):
            load_programs_from_redis()

        config_arg: RedisProgramStorageConfig = MockStorage.call_args[0][0]
        assert config_arg.redis_url == "redis://localhost:6379/0"
        assert config_arg.key_prefix == ""
