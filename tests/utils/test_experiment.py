"""check_storage_resume behavior against a fake ProgramStorage."""

from __future__ import annotations

import pytest

from gigaevo.utils.experiment import check_storage_resume


class _FakeStorage:
    def __init__(self, has_data: bool) -> None:
        self._has_data = has_data

    async def has_data(self) -> bool:
        return self._has_data


async def test_raises_when_data_exists_and_no_resume():
    with pytest.raises(RuntimeError, match="not empty"):
        await check_storage_resume(
            _FakeStorage(True),
            resume=False,
            location="Redis DB 0 at localhost:6379",
            flush_hint="gigaevo flush --db 0 --confirm",
        )


async def test_resumes_when_data_exists_and_resume_set():
    assert (
        await check_storage_resume(
            _FakeStorage(True),
            resume=True,
            location="db",
            flush_hint="hint",
        )
        is True
    )


async def test_fresh_start_when_empty():
    assert (
        await check_storage_resume(
            _FakeStorage(False),
            resume=False,
            location="db",
            flush_hint="hint",
        )
        is False
    )
    assert (
        await check_storage_resume(
            _FakeStorage(False),
            resume=True,
            location="db",
            flush_hint="hint",
        )
        is False
    )
