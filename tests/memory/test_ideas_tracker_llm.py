"""Prompt-step LLM calls route through the memory MultiModelRouter."""

from __future__ import annotations

from pydantic import BaseModel
import pytest

from gigaevo.memory.ideas_tracker.llm import (
    call_step,
    call_step_async,
    call_step_structured,
    call_step_structured_async,
    render_messages,
)
from tests.fakes.llm_router import FakeMemoryRouter

STEP = "cluster_fast_refine"


class _Partition(BaseModel):
    included: list[int] = []
    rejected: list[int] = []


class TestRenderMessages:
    def test_string_content_fills_insert_placeholder(self):
        messages = dict(render_messages(STEP, "IDEA-XYZ"))
        assert "IDEA-XYZ" in messages["user"]
        assert "<INSERT>" not in messages["user"]
        assert messages["system"]

    def test_dict_content_replaces_each_placeholder(self):
        messages = dict(
            render_messages("cluster_desc_synth", {"<INSERT_REP>": "rep-1"})
        )
        assert "rep-1" in messages["user"]
        assert "<INSERT_REP>" not in messages["user"]

    def test_unknown_step_raises(self):
        with pytest.raises(FileNotFoundError):
            render_messages("no_such_step")


class TestPlainCalls:
    def test_call_step_returns_router_text(self):
        llm = FakeMemoryRouter(text="the answer")
        assert call_step(llm, STEP, "x") == "the answer"
        assert llm.calls[0][0] == "invoke"

    def test_call_step_returns_empty_on_router_failure(self):
        llm = FakeMemoryRouter()
        llm.invoke = lambda messages: (_ for _ in ()).throw(RuntimeError("down"))
        assert call_step(llm, STEP, "x") == ""

    @pytest.mark.asyncio
    async def test_call_step_async_returns_router_text(self):
        llm = FakeMemoryRouter(text="async answer")
        assert await call_step_async(llm, STEP, "x") == "async answer"
        assert llm.calls[0][0] == "ainvoke"

    @pytest.mark.asyncio
    async def test_call_step_async_returns_empty_on_router_failure(self):
        llm = FakeMemoryRouter()

        async def boom(messages):
            raise RuntimeError("down")

        llm.ainvoke = boom
        assert await call_step_async(llm, STEP, "x") == ""


class TestStructuredCalls:
    def test_returns_parsed_schema_instance(self):
        llm = FakeMemoryRouter(
            respond=lambda schema, messages: schema(included=[1], rejected=[2])
        )
        parsed = call_step_structured(llm, STEP, _Partition, "x")
        assert parsed == _Partition(included=[1], rejected=[2])

    def test_defers_structured_method_to_router(self):
        captured = {}

        class _Capture(FakeMemoryRouter):
            def with_structured_output(self, schema, **kwargs):
                captured.update(kwargs)
                return super().with_structured_output(schema, **kwargs)

        call_step_structured(_Capture(), STEP, _Partition, "x")
        assert "method" not in captured

    @pytest.mark.asyncio
    async def test_async_defers_structured_method_to_router(self):
        captured = {}

        class _Capture(FakeMemoryRouter):
            def with_structured_output(self, schema, **kwargs):
                captured.update(kwargs)
                return super().with_structured_output(schema, **kwargs)

        await call_step_structured_async(_Capture(), STEP, _Partition, "x")
        assert "method" not in captured

    def test_router_failure_propagates(self):
        llm = FakeMemoryRouter(
            respond=lambda schema, messages: (_ for _ in ()).throw(
                ValueError("parse failed")
            )
        )
        with pytest.raises(ValueError, match="parse failed"):
            call_step_structured(llm, STEP, _Partition, "x")

    @pytest.mark.asyncio
    async def test_async_returns_parsed_schema_instance(self):
        llm = FakeMemoryRouter(
            respond=lambda schema, messages: schema(included=[3]), allow_sync=False
        )
        parsed = await call_step_structured_async(llm, STEP, _Partition, "x")
        assert parsed == _Partition(included=[3])
