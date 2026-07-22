"""``structured_output_method="auto"`` negotiates a working method per endpoint.

Different OpenAI-compatible servers accept different structured-output transports:
the in-cluster Qwen vLLM proxy serves ``json_schema`` (``response_format``) but
rejects ``function_calling`` (forced named ``tool_choice``); some models only
serve ``function_calling``. ``auto`` probes ``json_schema → function_calling →
json_mode`` per model, keeps the first that transports + parses, and caches it so
later calls skip the dead methods. Pinned methods and ``None`` (langchain default)
bypass negotiation entirely so the mutation path is unchanged.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage
from pydantic import BaseModel
import pytest

from gigaevo.llm.models import MultiModelRouter
from tests.conftest import NullWriter


class _Parsed(BaseModel):
    ok: bool = True


class _Bound:
    def __init__(self, model: _FakeModel, method: str | None) -> None:
        self._model = model
        self._method = method

    def invoke(self, _input, _config=None, **_kwargs):
        return self._model._respond(self._method)

    async def ainvoke(self, _input, _config=None, **_kwargs):
        return self._model._respond(self._method)


class _FakeModel:
    """ChatOpenAI stand-in that only serves a fixed set of structured methods.

    No ``base_url`` attribute, so ``MultiModelRouter._verify_models`` skips the
    network probe for it.
    """

    def __init__(self, name: str, supported, parsed: _Parsed) -> None:
        self.model_name = name
        self._supported = set(supported)
        self._parsed = parsed
        self.method_attempts: list[str | None] = []
        self.schemas_seen: list[object] = []

    def with_structured_output(self, schema, include_raw=False, method=None, **_kw):
        self.method_attempts.append(method)
        self.schemas_seen.append(schema)
        return _Bound(self, method)

    def _respond(self, method: str | None):
        if method not in self._supported:
            raise RuntimeError(f"method {method!r} unsupported by {self.model_name}")
        return {"raw": AIMessage(content="{}"), "parsed": self._parsed}


def _router(model: _FakeModel, method: str | None) -> MultiModelRouter:
    return MultiModelRouter(
        [model],
        [1.0],
        writer=NullWriter(),
        name="test",
        structured_output_method=method,
    )


_MSG = [("user", "hi")]


def test_auto_uses_first_supported_method():
    model = _FakeModel("m", {"json_schema"}, _Parsed())
    out = _router(model, "auto").with_structured_output(_Parsed).invoke(_MSG)
    assert isinstance(out, _Parsed)
    assert model.method_attempts == ["json_schema"]


def test_auto_falls_back_to_function_calling():
    model = _FakeModel("m", {"function_calling"}, _Parsed())
    out = _router(model, "auto").with_structured_output(_Parsed).invoke(_MSG)
    assert isinstance(out, _Parsed)
    assert model.method_attempts == ["json_schema", "function_calling"]


def test_auto_caches_resolved_method_across_calls():
    model = _FakeModel("m", {"function_calling"}, _Parsed())
    router = _router(model, "auto")
    router.with_structured_output(_Parsed).invoke(_MSG)
    router.with_structured_output(_Parsed).invoke(_MSG)
    assert model.method_attempts == [
        "json_schema",
        "function_calling",
        "function_calling",
    ]


def test_auto_raises_when_no_method_works():
    model = _FakeModel("m", set(), _Parsed())
    with pytest.raises(Exception):
        _router(model, "auto").with_structured_output(_Parsed).invoke(_MSG)


def test_pinned_method_skips_negotiation():
    model = _FakeModel("m", {"function_calling"}, _Parsed())
    out = (
        _router(model, "function_calling").with_structured_output(_Parsed).invoke(_MSG)
    )
    assert isinstance(out, _Parsed)
    assert model.method_attempts == ["function_calling"]


def test_function_calling_preserves_wrapped_json_schema_as_parameters():
    model = _FakeModel("m", {"function_calling"}, _Parsed())
    json_schema = {
        "title": "dag_tab_feature_graph_diff",
        "type": "object",
        "properties": {"base_parent": {"type": "string"}},
        "required": ["base_parent"],
    }
    wrapped = {"name": "dag_tab_feature_graph_diff", "schema": json_schema}

    _router(model, "function_calling").with_structured_output(wrapped).invoke(_MSG)

    assert model.schemas_seen == [
        {"name": "dag_tab_feature_graph_diff", "parameters": json_schema}
    ]


def test_json_schema_preserves_named_schema_wrapper():
    model = _FakeModel("m", {"json_schema"}, _Parsed())
    wrapped = {
        "name": "dag_tab_feature_graph_diff",
        "schema": {"title": "dag_tab_feature_graph_diff", "type": "object"},
    }

    _router(model, "json_schema").with_structured_output(wrapped).invoke(_MSG)

    assert model.schemas_seen == [wrapped]


def test_none_forwards_langchain_default():
    model = _FakeModel("m", {None}, _Parsed())
    out = _router(model, None).with_structured_output(_Parsed).invoke(_MSG)
    assert isinstance(out, _Parsed)
    assert model.method_attempts == [None]


def test_explicit_method_kwarg_overrides_auto():
    model = _FakeModel("m", {"function_calling"}, _Parsed())
    out = (
        _router(model, "auto")
        .with_structured_output(_Parsed, method="function_calling")
        .invoke(_MSG)
    )
    assert isinstance(out, _Parsed)
    assert model.method_attempts == ["function_calling"]


async def test_auto_async_falls_back_to_function_calling():
    model = _FakeModel("m", {"function_calling"}, _Parsed())
    out = await _router(model, "auto").with_structured_output(_Parsed).ainvoke(_MSG)
    assert isinstance(out, _Parsed)
    assert model.method_attempts == ["json_schema", "function_calling"]
