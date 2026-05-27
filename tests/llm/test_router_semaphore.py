"""Tests for MultiModelRouter LLM concurrency cap (semaphore)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from gigaevo.llm.models import MultiModelRouter, _StructuredOutputRouter
from tests.conftest import NullWriter


def _mock_model(name: str) -> MagicMock:
    m = MagicMock()
    m.model_name = name
    m.with_structured_output = MagicMock(return_value=MagicMock())
    return m


def _make_router(max_concurrent: int | None = None) -> MultiModelRouter:
    return MultiModelRouter(
        [_mock_model("m")],
        [1.0],
        writer=NullWriter(),
        name="sema-test",
        max_concurrent=max_concurrent,
    )


class TestSemaphoreFactory:
    def test_none_returns_none(self):
        router = _make_router(max_concurrent=None)
        assert router.get_semaphore() is None

    async def test_lazy_init_returns_same_instance(self):
        router = _make_router(max_concurrent=1)
        sema_a = router.get_semaphore()
        sema_b = router.get_semaphore()
        assert sema_a is sema_b
        assert isinstance(sema_a, asyncio.Semaphore)

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="max_concurrent"):
            _make_router(max_concurrent=0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="max_concurrent"):
            _make_router(max_concurrent=-1)


class TestStructuredOutputSharesSemaphore:
    async def test_with_structured_output_yields_same_semaphore(self):
        router = _make_router(max_concurrent=2)
        wrapped = router.with_structured_output(MagicMock())
        assert isinstance(wrapped, _StructuredOutputRouter)
        # factory must yield the SAME semaphore instance as the parent router
        assert wrapped._semaphore_factory() is router.get_semaphore()

    def test_with_structured_output_none_when_uncapped(self):
        router = _make_router(max_concurrent=None)
        wrapped = router.with_structured_output(MagicMock())
        assert wrapped._semaphore_factory() is None


class TestAinvokeSerializes:
    async def test_max_concurrent_one_serializes_two_calls(self):
        """With cap=1, two concurrent ainvoke calls must NOT overlap."""

        in_flight = 0
        peak = 0
        lock = asyncio.Lock()

        async def fake_ainvoke(*args, **kwargs):
            nonlocal in_flight, peak
            async with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            try:
                await asyncio.sleep(0.05)
                return MagicMock(content="ok")
            finally:
                async with lock:
                    in_flight -= 1

        model = _mock_model("m")
        model.ainvoke = fake_ainvoke
        router = MultiModelRouter(
            [model],
            [1.0],
            writer=NullWriter(),
            name="sema-serial",
            max_concurrent=1,
        )

        await asyncio.gather(router.ainvoke("a"), router.ainvoke("b"))
        assert peak == 1, f"expected serial execution, observed peak in-flight={peak}"

    async def test_no_cap_allows_overlap(self):
        """Sanity: with cap=None the same scenario allows overlap."""

        in_flight = 0
        peak = 0
        lock = asyncio.Lock()

        async def fake_ainvoke(*args, **kwargs):
            nonlocal in_flight, peak
            async with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            try:
                await asyncio.sleep(0.05)
                return MagicMock(content="ok")
            finally:
                async with lock:
                    in_flight -= 1

        model = _mock_model("m")
        model.ainvoke = fake_ainvoke
        router = MultiModelRouter(
            [model],
            [1.0],
            writer=NullWriter(),
            name="sema-uncapped",
            max_concurrent=None,
        )

        await asyncio.gather(router.ainvoke("a"), router.ainvoke("b"))
        assert peak == 2, f"expected overlap without cap, observed peak={peak}"

    async def test_structured_router_shares_cap_with_parent(self):
        """Structured-output router and parent router serialize together."""

        in_flight = 0
        peak = 0
        lock = asyncio.Lock()

        async def fake_call(*args, **kwargs):
            nonlocal in_flight, peak
            async with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            try:
                await asyncio.sleep(0.05)
                return MagicMock(content="ok")
            finally:
                async with lock:
                    in_flight -= 1

        # parent model — used directly by parent.ainvoke
        parent_model = _mock_model("m")
        parent_model.ainvoke = fake_call

        # structured wrapper returns a model whose ainvoke yields {"raw":..., "parsed":...}
        async def fake_structured_ainvoke(*args, **kwargs):
            nonlocal in_flight, peak
            async with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            try:
                await asyncio.sleep(0.05)
                return {"raw": MagicMock(content="x"), "parsed": {"ok": True}}
            finally:
                async with lock:
                    in_flight -= 1

        structured_wrapper = MagicMock()
        structured_wrapper.ainvoke = fake_structured_ainvoke
        parent_model.with_structured_output = MagicMock(return_value=structured_wrapper)

        router = MultiModelRouter(
            [parent_model],
            [1.0],
            writer=NullWriter(),
            name="sema-shared",
            max_concurrent=1,
        )
        structured = router.with_structured_output(MagicMock())

        await asyncio.gather(router.ainvoke("a"), structured.ainvoke("b"))
        assert peak == 1, (
            f"expected parent+structured to share semaphore (peak=1), got peak={peak}"
        )
