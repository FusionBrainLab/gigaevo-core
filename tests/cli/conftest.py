"""Shared fixtures for CLI tests."""

from __future__ import annotations

import asyncio
from pathlib import Path
import uuid

import pytest

from gigaevo.database.disk_program_storage import (
    DiskProgramStorage,
    DiskProgramStorageConfig,
)
from gigaevo.programs.program import Program


@pytest.fixture
def seed_disk_run(tmp_path: Path):
    """Factory: create an on-disk program storage and seed it with programs.

    Returns (root_dir, prefix) after writing one program per fitness value.
    """

    def _seed(
        prefix: str = "toy", fitnesses: tuple[float, ...] = (0.5, 0.8, 0.65)
    ) -> tuple[Path, str]:
        root = tmp_path / "storage"
        storage = DiskProgramStorage(
            DiskProgramStorageConfig(root_dir=str(root), key_prefix=prefix)
        )

        async def _fill() -> None:
            for i, fitness in enumerate(fitnesses):
                await storage.add(
                    Program(
                        id=str(uuid.UUID(int=i)),
                        code=f"def solve(): return {i}",
                        iteration=i,
                        metrics={"fitness": fitness},
                    )
                )
            await storage.close()

        asyncio.run(_fill())
        return root, prefix

    return _seed
