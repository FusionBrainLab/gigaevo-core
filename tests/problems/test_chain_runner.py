"""Tests for problems.chains.chain_runner reference resolution and stepwise dispatch."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load_module(module_name: str, relative_path: str) -> ModuleType:
    """Load a problems/ submodule by file path.

    tests/problems/__init__.py shadows the top-level ``problems`` namespace
    package, so direct ``from problems.chains.* import …`` does not work here.
    We therefore wire each chain_runner dependency explicitly.
    """
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = _ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_types = _load_module("chains_types_under_test", "problems/chains/types.py")
# chain_runner imports ``from problems.chains.types import …`` at top level,
# so register the loaded types module under that name before loading the runner.
sys.modules.setdefault("problems", ModuleType("problems"))
sys.modules.setdefault("problems.chains", ModuleType("problems.chains"))
sys.modules["problems.chains.types"] = _types

_chain_runner = _load_module(
    "chains_chain_runner_under_test", "problems/chains/chain_runner.py"
)

_resolve_reference = _chain_runner._resolve_reference
_run_chain_on_dataset_stepwise = _chain_runner._run_chain_on_dataset_stepwise
ChainSpec = _types.ChainSpec
PromptBuilder = _types.PromptBuilder
ToolConfig = _types.ToolConfig
ToolStep = _types.ToolStep


# ---------------------------------------------------------------------------
# _resolve_reference unit coverage
# ---------------------------------------------------------------------------


def test_resolve_reference_outer_context() -> None:
    assert _resolve_reference("$outer_context", "ctx", []) == "ctx"


def test_resolve_reference_history_last_empty() -> None:
    assert _resolve_reference("$history[-1]", "", []) == ""


def test_resolve_reference_history_last_nonempty() -> None:
    assert _resolve_reference("$history[-1]", "", ["a", "b"]) == "b"


def test_resolve_reference_history_indexed() -> None:
    assert _resolve_reference("$history[0]", "", ["a", "b"]) == "a"
    assert _resolve_reference("$history[1]", "", ["a", "b"]) == "b"
    assert _resolve_reference("$history[5]", "", ["a", "b"]) == ""


def test_resolve_reference_sample_field_present() -> None:
    sample = {"question": "Q?", "answer": "A"}
    assert _resolve_reference("$sample.question", "", [], sample) == "Q?"


def test_resolve_reference_sample_field_missing() -> None:
    sample = {"question": "Q?"}
    assert _resolve_reference("$sample.missing", "", [], sample) == ""


def test_resolve_reference_sample_none_returns_empty() -> None:
    """When no sample is threaded through, $sample.X resolves to empty string."""
    assert _resolve_reference("$sample.question", "", [], None) == ""
    assert _resolve_reference("$sample.question", "", []) == ""


def test_resolve_reference_sample_dotted_path() -> None:
    sample = {"meta": {"id": "42"}}
    assert _resolve_reference("$sample.meta.id", "", [], sample) == "42"


def test_resolve_reference_sample_non_string_coerced() -> None:
    sample = {"n": 7}
    assert _resolve_reference("$sample.n", "", [], sample) == "7"


def test_resolve_reference_unknown_syntax_raises() -> None:
    with pytest.raises(ValueError, match="Unknown reference syntax"):
        _resolve_reference("$bogus", "", [])


# ---------------------------------------------------------------------------
# Stepwise execution threads sample through to tool resolution
# ---------------------------------------------------------------------------


def _make_tool_chain() -> "ChainSpec":
    """One-step chain whose tool reads $sample.field and echoes it back."""
    return ChainSpec(
        system_prompt="",
        steps=[
            ToolStep(
                number=1,
                title="echo_field",
                step_type="tool",
                step_config=ToolConfig(
                    tool_name="echo",
                    input_mapping={"value": "$sample.field"},
                ),
                dependencies=[],
            ),
        ],
        prompt_builder=PromptBuilder(),
    )


def test_stepwise_threads_sample_into_tool_resolution() -> None:
    """The stepwise runner must pass each sample to $sample.* reference resolution.

    Regression: without this, $sample.X silently resolves to "" for every sample
    and the tool receives empty input.
    """
    chain = _make_tool_chain()
    dataset = [{"field": "alpha"}, {"field": "beta"}]
    received_kwargs: list[dict[str, str]] = []

    def echo_tool(value: str) -> str:
        received_kwargs.append({"value": value})
        return value

    results = asyncio.run(
        _run_chain_on_dataset_stepwise(
            chain=chain,
            client=None,  # No LLM steps in this chain.
            dataset=dataset,
            outer_context_builder=lambda _s: "",
            tool_registry={"echo": echo_tool},
        )
    )

    assert [kw["value"] for kw in received_kwargs] == ["alpha", "beta"]
    assert [r.final_output for r in results] == ["alpha", "beta"]
