"""Tests for the CARL-integrated chain infrastructure.

Coverage
--------
- ``carl_bridge``:  ``GigaEvoClientAdapter``, ``GigaEvoPromptTemplate``
- ``types``:        parse-layer models, ``to_carl_step()`` converters
- ``chain_validation``: all validation paths (static / full_chain / DAG / frozen)
- ``chain_runner``: step-batched execution — LLM steps, tool steps (per-sample
  and batch), step_max_tokens, _strip_thinking, dependency-filtered history
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from mmar_carl import LLMStepDescription, ToolStepDescription
from mmar_carl.models import Language
from mmar_carl.models.config import ToolStepConfig
import pytest

from problems.chains.carl_bridge import GigaEvoClientAdapter, GigaEvoPromptTemplate
from problems.chains.chain_runner import (
    _resolve_dependencies,
    _resolve_reference,
    _run_chain_on_dataset_stepwise,
    _strip_thinking,
    run_chain_on_dataset_stepwise,
)
from problems.chains.chain_validation import validate_chain_spec
from problems.chains.types import (
    ChainSpec,
    LLMStep,
    PromptBuilder,
    RawChainSpec,
    ToolConfig,
    ToolStep,
)

# ===========================================================================
# Fixtures
# ===========================================================================


def _make_llm_step_dict(**overrides) -> dict:
    base = {
        "number": 1,
        "title": "Analyse",
        "step_type": "llm",
        "aim": "Analyse the data",
        "stage_action": "Perform analysis",
    }
    return {**base, **overrides}


def _make_tool_step_dict(number: int = 2, tool: str = "retriever") -> dict:
    return {
        "number": number,
        "title": "Retrieve",
        "step_type": "tool",
        "dependencies": [number - 1],
        "step_config": {
            "tool_name": tool,
            "input_mapping": {"query": "$history[-1]"},
        },
    }


def _minimal_chain_dict(extra_steps: list[dict] | None = None) -> dict:
    steps = [_make_llm_step_dict()]
    if extra_steps:
        steps.extend(extra_steps)
    return {"system_prompt": "sys", "steps": steps}


# ---------------------------------------------------------------------------
# Minimal ChainSpec builder for runner tests
# ---------------------------------------------------------------------------


def _build_chain_spec(
    llm_steps: list[dict] | None = None,
    tool_steps: list[dict] | None = None,
    system_prompt: str = "sys",
) -> ChainSpec:
    """Build a ChainSpec directly from CARL step descriptions."""
    steps: list[LLMStepDescription | ToolStepDescription] = []
    for s in llm_steps or []:
        steps.append(
            LLMStepDescription(
                number=s["number"],
                title=s.get("title", "Step"),
                dependencies=s.get("dependencies", []),
                aim=s.get("aim", "Do it"),
                stage_action=s.get("stage_action", "Action"),
                reasoning_questions=s.get("reasoning_questions", ""),
                example_reasoning=s.get("example_reasoning", ""),
            )
        )
    for s in tool_steps or []:
        steps.append(
            ToolStepDescription(
                number=s["number"],
                title=s.get("title", "Tool"),
                dependencies=s.get("dependencies", []),
                config=ToolStepConfig(
                    tool_name=s["tool_name"],
                    input_mapping=s.get("input_mapping", {}),
                ),
            )
        )
    steps.sort(key=lambda x: x.number)
    return ChainSpec(system_prompt=system_prompt, steps=steps)


# ===========================================================================
# carl_bridge — GigaEvoClientAdapter
# ===========================================================================


class TestGigaEvoClientAdapter:
    @pytest.mark.asyncio
    async def test_get_response_delegates_to_client(self):
        mock_client = AsyncMock(return_value="answer")
        adapter = GigaEvoClientAdapter(mock_client)
        result = await adapter.get_response("prompt")
        mock_client.assert_awaited_once_with("prompt")
        assert result == "answer"

    @pytest.mark.asyncio
    async def test_get_response_with_retries_delegates(self):
        mock_client = AsyncMock(return_value="answer")
        adapter = GigaEvoClientAdapter(mock_client)
        result = await adapter.get_response_with_retries("prompt", retries=5)
        mock_client.assert_awaited_once_with("prompt")
        assert result == "answer"

    @pytest.mark.asyncio
    async def test_max_tokens_forwarded(self):
        mock_client = AsyncMock(return_value="answer")
        adapter = GigaEvoClientAdapter(mock_client, max_tokens=512)
        await adapter.get_response("p")
        mock_client.assert_awaited_once_with("p", max_tokens=512)

    @pytest.mark.asyncio
    async def test_no_max_tokens_no_kwarg(self):
        mock_client = AsyncMock(return_value="x")
        adapter = GigaEvoClientAdapter(mock_client)
        await adapter.get_response("p")
        # Called with just the prompt — no max_tokens kwarg
        mock_client.assert_awaited_once_with("p")


# ===========================================================================
# carl_bridge — GigaEvoPromptTemplate
# ===========================================================================


class TestGigaEvoPromptTemplate:
    def setup_method(self):
        self.tpl = GigaEvoPromptTemplate()

    def _make_llm_desc(self, **kw) -> LLMStepDescription:
        defaults = dict(
            number=1,
            title="T",
            aim="aim text",
            stage_action="action text",
            reasoning_questions="q?",
            example_reasoning="ex",
        )
        return LLMStepDescription(**{**defaults, **kw})

    def test_format_history_entry(self):
        entry = self.tpl.format_history_entry(2, "My Step", "raw output")
        assert entry == "Step 2. My Step\nResult: raw output\n"

    def test_format_step_prompt_no_context_queries_in_output(self):
        step = self._make_llm_desc()
        prompt = self.tpl.format_step_prompt(step, "ctx", Language.ENGLISH)
        # No "No specific context queries" or CARL's RAG text should appear
        assert "context queries" not in prompt.lower()
        assert "Objective: aim text" in prompt
        assert "Task: action text" in prompt
        assert "Questions: q?" in prompt
        assert "Example reasoning: ex" in prompt

    def test_format_chain_prompt_no_prescriptive_text(self):
        step = self._make_llm_desc()
        step_prompt = self.tpl.format_step_prompt(step, "ctx", Language.ENGLISH)
        full = self.tpl.format_chain_prompt(
            outer_context="DATA",
            current_task=step_prompt,
            history="",
            language=Language.ENGLISH,
            system_prompt="",
        )
        assert full.startswith("Data:\nDATA\n\n")
        # CARL's "Respond concisely" suffix should not appear
        assert "Respond concisely" not in full

    def test_format_chain_prompt_with_history(self):
        step = self._make_llm_desc()
        step_prompt = self.tpl.format_step_prompt(step, "ctx", Language.ENGLISH)
        full = self.tpl.format_chain_prompt(
            outer_context="DATA",
            current_task=step_prompt,
            history="Step 1. Prev\nResult: prev_out\n",
            language=Language.ENGLISH,
            system_prompt="",
        )
        assert "Previous steps:" in full
        assert "Based on the results of previous steps" in full

    def test_format_chain_prompt_with_system_prompt(self):
        step = self._make_llm_desc()
        step_prompt = self.tpl.format_step_prompt(step, "ctx", Language.ENGLISH)
        full = self.tpl.format_chain_prompt(
            outer_context="D",
            current_task=step_prompt,
            history="",
            language=Language.ENGLISH,
            system_prompt="Be careful.",
        )
        assert full.startswith("System Instructions:\nBe careful.\n\n")

    def test_matches_prompt_builder_format_no_history(self):
        """GigaEvoPromptTemplate must produce the same output as PromptBuilder."""
        step_dict = _make_llm_step_dict(
            number=1,
            title="Analysis",
            aim="Analyse it",
            stage_action="Do the analysis",
            reasoning_questions="Why?",
            example_reasoning="Because…",
        )
        parse_step = LLMStep(**step_dict)
        builder = PromptBuilder()
        builder_out = builder.build_prompt(
            step=parse_step,
            visible_history=[],
            outer_context="some data",
            system_prompt="",
        )

        carl_step = parse_step.to_carl_step()
        tpl = GigaEvoPromptTemplate()
        tpl_step_prompt = tpl.format_step_prompt(
            carl_step, "some data", Language.ENGLISH
        )
        tpl_out = tpl.format_chain_prompt(
            outer_context="some data",
            current_task=tpl_step_prompt,
            history="",
            language=Language.ENGLISH,
            system_prompt="",
        )
        assert builder_out == tpl_out

    def test_matches_prompt_builder_format_with_history(self):
        step_dict = _make_llm_step_dict(
            number=2,
            title="Step 2",
            aim="aim",
            stage_action="action",
            reasoning_questions="",
            example_reasoning="",
        )
        parse_step = LLMStep(**step_dict)
        history_entry = "Step 1. First\nResult: first_output\n"
        builder = PromptBuilder()
        builder_out = builder.build_prompt(
            step=parse_step,
            visible_history=[history_entry],
            outer_context="data",
            system_prompt="SYS",
        )

        carl_step = parse_step.to_carl_step()
        tpl = GigaEvoPromptTemplate()
        tpl_step_prompt = tpl.format_step_prompt(carl_step, "data", Language.ENGLISH)
        tpl_out = tpl.format_chain_prompt(
            outer_context="data",
            current_task=tpl_step_prompt,
            history=history_entry,
            language=Language.ENGLISH,
            system_prompt="SYS",
        )
        assert builder_out == tpl_out


# ===========================================================================
# types — parse-layer models
# ===========================================================================


class TestParseLayerTypes:
    def test_llm_step_parses_valid(self):
        step = LLMStep(**_make_llm_step_dict())
        assert step.step_type == "llm"
        assert step.frozen is False
        assert step.dependencies == []

    def test_llm_step_rejects_extra_fields(self):
        with pytest.raises(Exception):
            LLMStep(**_make_llm_step_dict(), unknown_field="x")

    def test_llm_step_coerces_list_to_str(self):
        step = LLMStep(**_make_llm_step_dict(aim=["a", "b"]))
        assert step.aim == "a b"

    def test_llm_step_requires_non_empty_aim(self):
        with pytest.raises(Exception):
            LLMStep(**_make_llm_step_dict(aim=""))

    def test_tool_step_parses_valid(self):
        raw = _make_tool_step_dict()
        step = ToolStep(**raw)
        assert step.step_type == "tool"
        assert step.step_config.tool_name == "retriever"
        assert step.step_config.input_mapping == {"query": "$history[-1]"}

    def test_tool_config_rejects_non_dollar_ref(self):
        with pytest.raises(Exception):
            ToolConfig(tool_name="t", input_mapping={"q": "not_a_ref"})

    def test_raw_chain_spec_discriminated_union(self):
        raw = _minimal_chain_dict(extra_steps=[_make_tool_step_dict()])
        parsed = RawChainSpec.model_validate(raw)
        assert isinstance(parsed.steps[0], LLMStep)
        assert isinstance(parsed.steps[1], ToolStep)


# ===========================================================================
# types — to_carl_step() converters
# ===========================================================================


class TestToCarlStep:
    def test_llm_step_to_carl(self):
        parse_step = LLMStep(**_make_llm_step_dict(frozen=True, dependencies=[]))
        carl_step = parse_step.to_carl_step()
        assert isinstance(carl_step, LLMStepDescription)
        assert carl_step.number == parse_step.number
        assert carl_step.title == parse_step.title
        assert carl_step.aim == parse_step.aim
        assert carl_step.stage_action == parse_step.stage_action
        assert carl_step.reasoning_questions == parse_step.reasoning_questions
        assert carl_step.example_reasoning == parse_step.example_reasoning
        # frozen is a parse-layer concept — must not appear on CARL step
        assert not hasattr(carl_step, "frozen")

    def test_tool_step_to_carl(self):
        parse_step = ToolStep(**_make_tool_step_dict())
        carl_step = parse_step.to_carl_step()
        assert isinstance(carl_step, ToolStepDescription)
        assert carl_step.number == parse_step.number
        assert carl_step.config.tool_name == "retriever"
        assert carl_step.config.input_mapping == {"query": "$history[-1]"}

    def test_tool_step_to_carl_preserves_dependencies(self):
        parse_step_with_deps = ToolStep(
            number=3,
            title="T",
            step_type="tool",
            dependencies=[1, 2],
            step_config=ToolConfig(tool_name="t", input_mapping={"q": "$history[-1]"}),
        )
        carl_step = parse_step_with_deps.to_carl_step()
        assert carl_step.dependencies == [1, 2]


# ===========================================================================
# chain_validation
# ===========================================================================


class TestValidateChainSpec:
    # -- happy paths ----------------------------------------------------------

    def test_static_mode_single_llm_step(self):
        topology = {
            "num_steps": 1,
            "steps": [{"number": 1, "step_type": "llm", "dependencies": []}],
        }
        spec = validate_chain_spec(
            _minimal_chain_dict(),
            mode="static",
            topology=topology,
        )
        assert isinstance(spec, ChainSpec)
        assert len(spec.steps) == 1
        assert isinstance(spec.steps[0], LLMStepDescription)

    def test_static_mode_llm_and_tool_steps(self):
        raw = _minimal_chain_dict(extra_steps=[_make_tool_step_dict()])
        topology = {
            "num_steps": 2,
            "steps": [
                {"number": 1, "step_type": "llm", "dependencies": []},
                {"number": 2, "step_type": "tool", "dependencies": [1]},
            ],
        }
        spec = validate_chain_spec(raw, mode="static", topology=topology)
        assert len(spec.steps) == 2
        assert isinstance(spec.steps[0], LLMStepDescription)
        assert isinstance(spec.steps[1], ToolStepDescription)

    def test_full_chain_mode(self):
        raw = _minimal_chain_dict()
        fc_config = {
            "max_steps": 5,
            "allowed_step_types": ["llm", "tool"],
            "available_tools": [],
            "require_final_llm": True,
        }
        spec = validate_chain_spec(raw, mode="full_chain", full_chain_config=fc_config)
        assert len(spec.steps) == 1

    def test_steps_sorted_by_number(self):
        """Steps must come out sorted even if the LLM emits them unordered."""
        raw = {
            "system_prompt": "",
            "steps": [
                _make_llm_step_dict(number=3, aim="a", stage_action="s"),
                _make_llm_step_dict(number=1, aim="a", stage_action="s"),
                _make_llm_step_dict(
                    number=2, aim="a", stage_action="s", dependencies=[1]
                ),
            ],
        }
        topology = {
            "num_steps": 3,
            "steps": [
                {"number": 1, "step_type": "llm", "dependencies": []},
                {"number": 2, "step_type": "llm", "dependencies": [1]},
                {"number": 3, "step_type": "llm", "dependencies": []},
            ],
        }
        spec = validate_chain_spec(raw, mode="static", topology=topology)
        assert [s.number for s in spec.steps] == [1, 2, 3]

    def test_system_prompt_preserved(self):
        raw = {"system_prompt": "Be concise.", "steps": [_make_llm_step_dict()]}
        topology = {
            "num_steps": 1,
            "steps": [{"number": 1, "step_type": "llm", "dependencies": []}],
        }
        spec = validate_chain_spec(raw, mode="static", topology=topology)
        assert spec.system_prompt == "Be concise."

    def test_carl_steps_in_chain_spec(self):
        """validate_chain_spec must produce CARL types in ChainSpec.steps."""
        raw = _minimal_chain_dict(extra_steps=[_make_tool_step_dict()])
        topology = {
            "num_steps": 2,
            "steps": [
                {"number": 1, "step_type": "llm", "dependencies": []},
                {"number": 2, "step_type": "tool", "dependencies": [1]},
            ],
        }
        spec = validate_chain_spec(raw, mode="static", topology=topology)
        for step in spec.steps:
            assert isinstance(step, (LLMStepDescription, ToolStepDescription))

    # -- failure paths --------------------------------------------------------

    def test_duplicate_step_numbers(self):
        raw = {
            "system_prompt": "",
            "steps": [
                _make_llm_step_dict(number=1, aim="a", stage_action="s"),
                _make_llm_step_dict(number=1, aim="b", stage_action="s"),
            ],
        }
        topology = {
            "num_steps": 2,
            "steps": [
                {"number": 1, "step_type": "llm", "dependencies": []},
            ],
        }
        with pytest.raises(ValueError, match="Duplicate"):
            validate_chain_spec(raw, mode="static", topology=topology)

    def test_dag_forward_dependency(self):
        raw = {
            "system_prompt": "",
            "steps": [
                _make_llm_step_dict(
                    number=1, aim="a", stage_action="s", dependencies=[2]
                ),
                _make_llm_step_dict(number=2, aim="b", stage_action="s"),
            ],
        }
        with pytest.raises(ValueError, match="depends on later step"):
            validate_chain_spec(
                raw,
                mode="full_chain",
                full_chain_config={
                    "max_steps": 10,
                    "allowed_step_types": ["llm"],
                    "available_tools": [],
                    "require_final_llm": True,
                },
            )

    def test_dag_nonexistent_dependency(self):
        raw = {
            "system_prompt": "",
            "steps": [
                _make_llm_step_dict(
                    number=1, aim="a", stage_action="s", dependencies=[99]
                ),
            ],
        }
        with pytest.raises(ValueError, match="non-existent"):
            validate_chain_spec(
                raw,
                mode="full_chain",
                full_chain_config={
                    "max_steps": 10,
                    "allowed_step_types": ["llm"],
                    "available_tools": [],
                    "require_final_llm": True,
                },
            )

    def test_static_topology_mismatch(self):
        raw = _minimal_chain_dict()
        topology = {
            "num_steps": 2,  # expecting 2 but spec has 1
            "steps": [
                {"number": 1, "step_type": "llm", "dependencies": []},
                {"number": 2, "step_type": "llm", "dependencies": [1]},
            ],
        }
        with pytest.raises(ValueError, match="Expected 2 steps"):
            validate_chain_spec(raw, mode="static", topology=topology)

    def test_static_step_type_mismatch(self):
        raw = {
            "system_prompt": "",
            "steps": [_make_llm_step_dict()],
        }
        topology = {
            "num_steps": 1,
            "steps": [{"number": 1, "step_type": "tool", "dependencies": []}],
        }
        with pytest.raises(ValueError, match="type mismatch"):
            validate_chain_spec(raw, mode="static", topology=topology)

    def test_full_chain_too_many_steps(self):
        raw = {
            "system_prompt": "",
            "steps": [
                _make_llm_step_dict(number=i, aim="a", stage_action="s")
                for i in range(1, 4)
            ],
        }
        with pytest.raises(ValueError, match="Too many steps"):
            validate_chain_spec(
                raw,
                mode="full_chain",
                full_chain_config={
                    "max_steps": 2,
                    "allowed_step_types": ["llm"],
                    "available_tools": [],
                    "require_final_llm": True,
                },
            )

    def test_frozen_step_equality(self):
        """Frozen step must match the baseline exactly."""
        baseline = {
            "system_prompt": "",
            "steps": [_make_llm_step_dict(frozen=True)],
        }
        # Same content → should pass
        spec = validate_chain_spec(
            baseline,
            mode="static",
            topology={
                "num_steps": 1,
                "steps": [
                    {
                        "number": 1,
                        "step_type": "llm",
                        "dependencies": [],
                        "frozen": True,
                    }
                ],
            },
            frozen_baseline=baseline,
        )
        assert len(spec.steps) == 1

    def test_frozen_step_modified_raises(self):
        baseline = {
            "system_prompt": "",
            "steps": [_make_llm_step_dict(frozen=True, aim="original aim")],
        }
        modified = {
            "system_prompt": "",
            "steps": [_make_llm_step_dict(frozen=True, aim="different aim")],
        }
        with pytest.raises(ValueError, match="frozen but differs"):
            validate_chain_spec(
                modified,
                mode="static",
                topology={
                    "num_steps": 1,
                    "steps": [
                        {
                            "number": 1,
                            "step_type": "llm",
                            "dependencies": [],
                            "frozen": True,
                        }
                    ],
                },
                frozen_baseline=baseline,
            )

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown validation mode"):
            validate_chain_spec(_minimal_chain_dict(), mode="invalid")


# ===========================================================================
# chain_runner — utility functions
# ===========================================================================


class TestStripThinking:
    def test_strips_well_formed_block(self):
        assert _strip_thinking("hello <think>internal</think> world") == "hello  world"

    def test_strips_truncated_block(self):
        assert _strip_thinking("answer <think>unfinished") == "answer"

    def test_no_thinking_unchanged(self):
        assert _strip_thinking("plain text") == "plain text"

    def test_multiline_thinking(self):
        result = _strip_thinking("<think>\nline1\nline2\n</think>answer")
        assert result == "answer"

    def test_strips_and_strips(self):
        """Both well-formed and truncated blocks can coexist."""
        text = "<think>A</think>mid<think>B"
        result = _strip_thinking(text)
        assert "<think>" not in result


class TestResolveReference:
    def test_outer_context(self):
        assert _resolve_reference("$outer_context", "CTX", []) == "CTX"

    def test_history_last(self):
        assert _resolve_reference("$history[-1]", "ctx", ["a", "b"]) == "b"

    def test_history_last_empty(self):
        assert _resolve_reference("$history[-1]", "ctx", []) == ""

    def test_history_by_index(self):
        assert _resolve_reference("$history[0]", "ctx", ["first", "second"]) == "first"
        assert _resolve_reference("$history[1]", "ctx", ["first", "second"]) == "second"

    def test_history_index_out_of_range(self):
        assert _resolve_reference("$history[5]", "ctx", ["only"]) == ""

    def test_unknown_ref_raises(self):
        with pytest.raises(ValueError, match="Unknown reference"):
            _resolve_reference("$unknown_ref", "ctx", [])


class TestResolveDependencies:
    def test_empty_deps_returns_all(self):
        vis_hist, vis_outs = _resolve_dependencies(
            [], ["h1", "h2", "h3"], ["o1", "o2", "o3"]
        )
        assert vis_hist == ["h1", "h2", "h3"]
        assert vis_outs == {1: "o1", 2: "o2", 3: "o3"}

    def test_specific_deps_filters(self):
        vis_hist, vis_outs = _resolve_dependencies(
            [1, 3], ["h1", "h2", "h3"], ["o1", "o2", "o3"]
        )
        assert vis_hist == ["h1", "h3"]
        assert vis_outs == {1: "o1", 3: "o3"}

    def test_dep_beyond_completed_ignored(self):
        vis_hist, vis_outs = _resolve_dependencies([1, 5], ["h1"], ["o1"])
        assert vis_hist == ["h1"]
        assert 5 not in vis_outs


# ===========================================================================
# chain_runner — step-batched execution
# ===========================================================================


class _FakeClient:
    """Minimal fake gigaevo LLM client that returns a canned response."""

    def __init__(self, response: str = "llm_output", *, record_calls: bool = False):
        self._response = response
        self.calls: list[tuple[str, dict]] = []
        self._record = record_calls

    def copy(self) -> _FakeClient:
        return _FakeClient(self._response, record_calls=self._record)

    async def __call__(self, prompt: str, **kwargs) -> str:
        if self._record:
            self.calls.append((prompt, kwargs))
        return self._response


@pytest.mark.asyncio
class TestStepBatchedRunner:
    async def test_single_llm_step_single_sample(self):
        chain = _build_chain_spec(
            llm_steps=[{"number": 1, "title": "T", "aim": "a", "stage_action": "s"}]
        )
        client = _FakeClient("my_answer")
        results = await _run_chain_on_dataset_stepwise(
            chain=chain,
            client=client,
            dataset=[{"text": "sample1"}],
            outer_context_builder=lambda s: s["text"],
        )
        assert len(results) == 1
        assert results[0].final_output == "my_answer"
        assert results[0].step_outputs == ["my_answer"]
        assert len(results[0].history) == 1
        assert "Step 1." in results[0].history[0]
        assert "my_answer" in results[0].history[0]

    async def test_multiple_samples(self):
        chain = _build_chain_spec(
            llm_steps=[{"number": 1, "title": "T", "aim": "a", "stage_action": "s"}]
        )
        counter = {"n": 0}

        class CountingClient:
            def copy(self):
                return self

            async def __call__(self, prompt, **kw):
                counter["n"] += 1
                return f"out_{counter['n']}"

        results = await _run_chain_on_dataset_stepwise(
            chain=chain,
            client=CountingClient(),
            dataset=[{"x": i} for i in range(5)],
            outer_context_builder=lambda s: str(s["x"]),
        )
        assert len(results) == 5
        for r in results:
            assert r.final_output.startswith("out_")

    async def test_thinking_stripped_from_final_output(self):
        chain = _build_chain_spec(
            llm_steps=[{"number": 1, "aim": "a", "stage_action": "s"}]
        )
        client = _FakeClient("<think>internal reasoning</think>clean answer")
        results = await _run_chain_on_dataset_stepwise(
            chain=chain,
            client=client,
            dataset=[{"x": 1}],
            outer_context_builder=lambda s: "ctx",
        )
        assert results[0].final_output == "clean answer"
        assert "<think>" not in results[0].history[0]

    async def test_tool_step_per_sample_registry(self):
        chain = _build_chain_spec(
            llm_steps=[{"number": 1, "aim": "a", "stage_action": "s"}],
            tool_steps=[
                {
                    "number": 2,
                    "title": "Retrieve",
                    "dependencies": [1],
                    "tool_name": "search",
                    "input_mapping": {"query": "$history[-1]"},
                }
            ],
        )
        client = _FakeClient("step1_output")

        def fake_search(query: str) -> str:
            return f"result_for_{query}"

        results = await _run_chain_on_dataset_stepwise(
            chain=chain,
            client=client,
            dataset=[{"x": 1}],
            outer_context_builder=lambda s: "ctx",
            tool_registry={"search": fake_search},
        )
        assert results[0].step_outputs[0] == "step1_output"
        assert results[0].step_outputs[1] == "result_for_step1_output"
        assert results[0].final_output == "result_for_step1_output"

    async def test_tool_step_batch_registry_takes_precedence(self):
        chain = _build_chain_spec(
            llm_steps=[{"number": 1, "aim": "a", "stage_action": "s"}],
            tool_steps=[
                {
                    "number": 2,
                    "title": "BatchSearch",
                    "dependencies": [1],
                    "tool_name": "bm25",
                    "input_mapping": {"query": "$history[-1]"},
                }
            ],
        )
        client = _FakeClient("llm_out")
        batch_calls: list[list[dict]] = []

        def batch_bm25(items: list[dict]) -> list[str]:
            batch_calls.append(items)
            return [f"batch_result_{kw['query']}" for kw in items]

        results = await _run_chain_on_dataset_stepwise(
            chain=chain,
            client=client,
            dataset=[{"x": 1}, {"x": 2}],
            outer_context_builder=lambda s: "ctx",
            batch_tool_registry={"bm25": batch_bm25},
        )
        # Batch function called once for both samples
        assert len(batch_calls) == 1
        assert len(batch_calls[0]) == 2
        assert results[0].step_outputs[1] == "batch_result_llm_out"

    async def test_tool_step_dollar_outer_context(self):
        chain = _build_chain_spec(
            tool_steps=[
                {
                    "number": 1,
                    "title": "Search",
                    "tool_name": "search",
                    "input_mapping": {"query": "$outer_context"},
                }
            ]
        )
        called_with = []

        def fake_search(query: str) -> str:
            called_with.append(query)
            return "found"

        results = await _run_chain_on_dataset_stepwise(
            chain=chain,
            client=_FakeClient(),
            dataset=[{"ctx": "my context"}],
            outer_context_builder=lambda s: s["ctx"],
            tool_registry={"search": fake_search},
        )
        assert called_with == ["my context"]
        assert results[0].final_output == "found"

    async def test_step_max_tokens_forwarded(self):
        """max_tokens override must be forwarded to the LLM client call."""
        chain = _build_chain_spec(
            llm_steps=[{"number": 1, "aim": "a", "stage_action": "s"}]
        )
        received_kwargs: list[dict] = []

        class KwClient:
            def copy(self):
                return self

            async def __call__(self, prompt, **kw):
                received_kwargs.append(kw)
                return "ok"

        await _run_chain_on_dataset_stepwise(
            chain=chain,
            client=KwClient(),
            dataset=[{"x": 1}],
            outer_context_builder=lambda s: "ctx",
            step_max_tokens={1: 256},
        )
        assert received_kwargs[0].get("max_tokens") == 256

    async def test_step_max_tokens_only_for_specified_step(self):
        """Only the step in step_max_tokens gets the override, others don't."""
        chain = _build_chain_spec(
            llm_steps=[
                {"number": 1, "aim": "a", "stage_action": "s"},
                {"number": 2, "aim": "b", "stage_action": "t", "dependencies": []},
            ]
        )
        received: list[dict] = []

        class KwClient:
            def copy(self):
                return self

            async def __call__(self, prompt, **kw):
                received.append(kw)
                return "ok"

        await _run_chain_on_dataset_stepwise(
            chain=chain,
            client=KwClient(),
            dataset=[{"x": 1}],
            outer_context_builder=lambda s: "ctx",
            step_max_tokens={2: 512},
        )
        # Step 1 call should have no max_tokens
        assert "max_tokens" not in received[0]
        # Step 2 call should have max_tokens=512
        assert received[1].get("max_tokens") == 512

    async def test_dependency_filtered_history_in_prompt(self):
        """Step 2 with dependency=[1] should see step 1 history in its prompt."""
        chain = _build_chain_spec(
            llm_steps=[
                {"number": 1, "aim": "a", "stage_action": "s"},
                {"number": 2, "aim": "b", "stage_action": "t", "dependencies": [1]},
            ]
        )
        prompts_seen: list[str] = []

        class RecordingClient:
            def copy(self):
                return self

            async def __call__(self, prompt, **kw):
                prompts_seen.append(prompt)
                return "ok"

        await _run_chain_on_dataset_stepwise(
            chain=chain,
            client=RecordingClient(),
            dataset=[{"x": 1}],
            outer_context_builder=lambda s: "ctx",
        )
        # Step 2's prompt should contain "Previous steps:" with step 1's result
        step2_prompt = prompts_seen[1]
        assert "Previous steps:" in step2_prompt
        assert "Step 1." in step2_prompt

    async def test_missing_tool_registry_raises(self):
        chain = _build_chain_spec(
            tool_steps=[
                {
                    "number": 1,
                    "tool_name": "missing",
                    "input_mapping": {"q": "$outer_context"},
                }
            ]
        )
        with pytest.raises(ValueError, match="Tool step|not found in any registry"):
            await _run_chain_on_dataset_stepwise(
                chain=chain,
                client=_FakeClient(),
                dataset=[{"x": 1}],
                outer_context_builder=lambda s: "ctx",
            )

    async def test_empty_dataset_returns_empty(self):
        chain = _build_chain_spec(
            llm_steps=[{"number": 1, "aim": "a", "stage_action": "s"}]
        )
        results = await _run_chain_on_dataset_stepwise(
            chain=chain,
            client=_FakeClient(),
            dataset=[],
            outer_context_builder=lambda s: "ctx",
        )
        assert results == []

    async def test_carl_context_history_matches_chain_result(self):
        """ChainResult.history must match what ReasoningContext accumulated."""
        chain = _build_chain_spec(
            llm_steps=[
                {"number": 1, "aim": "a", "stage_action": "s"},
                {"number": 2, "aim": "b", "stage_action": "t", "dependencies": []},
            ]
        )
        outputs = ["first", "second"]
        idx = {"n": 0}

        class SeqClient:
            def copy(self):
                return self

            async def __call__(self, prompt, **kw):
                out = outputs[idx["n"] % len(outputs)]
                idx["n"] += 1
                return out

        results = await _run_chain_on_dataset_stepwise(
            chain=chain,
            client=SeqClient(),
            dataset=[{"x": 1}],
            outer_context_builder=lambda s: "ctx",
        )
        assert len(results[0].history) == 2
        assert "first" in results[0].history[0]
        assert "second" in results[0].history[1]


# ===========================================================================
# Sync wrappers
# ===========================================================================


class TestSyncWrappers:
    def test_run_chain_on_dataset_stepwise_sync(self):
        chain = _build_chain_spec(
            llm_steps=[{"number": 1, "aim": "a", "stage_action": "s"}]
        )
        client = _FakeClient("sync_out")
        results = run_chain_on_dataset_stepwise(
            chain=chain,
            client=client,
            dataset=[{"x": 1}],
            outer_context_builder=lambda s: "ctx",
        )
        assert len(results) == 1
        assert results[0].final_output == "sync_out"


# ===========================================================================
# Integration — validate_chain_spec → run_chain_on_dataset_stepwise
# ===========================================================================


class TestIntegration:
    @pytest.mark.asyncio
    async def test_validate_then_run(self):
        """Full pipeline: parse → validate → CARL steps → step-batched runner."""
        raw = {
            "system_prompt": "Be concise.",
            "steps": [
                _make_llm_step_dict(
                    number=1,
                    title="Analysis",
                    aim="Analyse",
                    stage_action="Do it",
                    stage_action2=None,
                ),
            ],
        }
        # Remove stray key
        raw["steps"][0].pop("stage_action2", None)
        topology = {
            "num_steps": 1,
            "steps": [{"number": 1, "step_type": "llm", "dependencies": []}],
        }
        chain = validate_chain_spec(raw, mode="static", topology=topology)
        # Steps are CARL types
        assert isinstance(chain.steps[0], LLMStepDescription)

        client = _FakeClient("integration_answer")
        results = await _run_chain_on_dataset_stepwise(
            chain=chain,
            client=client,
            dataset=[{"q": "What is 2+2?"}],
            outer_context_builder=lambda s: s["q"],
        )
        assert results[0].final_output == "integration_answer"
        assert results[0].step_outputs == ["integration_answer"]

    @pytest.mark.asyncio
    async def test_tool_step_resolves_llm_output(self):
        """Tool step must receive the raw LLM output, not the formatted entry."""
        raw = {
            "system_prompt": "",
            "steps": [
                _make_llm_step_dict(number=1, aim="a", stage_action="s"),
                _make_tool_step_dict(number=2, tool="search"),
            ],
        }
        topology = {
            "num_steps": 2,
            "steps": [
                {"number": 1, "step_type": "llm", "dependencies": []},
                {"number": 2, "step_type": "tool", "dependencies": [1]},
            ],
        }
        chain = validate_chain_spec(raw, mode="static", topology=topology)
        client = _FakeClient("raw_answer")
        tool_queries: list[str] = []

        def search(query: str) -> str:
            tool_queries.append(query)
            return "docs"

        results = await _run_chain_on_dataset_stepwise(
            chain=chain,
            client=client,
            dataset=[{"x": 1}],
            outer_context_builder=lambda s: "ctx",
            tool_registry={"search": search},
        )
        # Tool must have received the raw LLM output, not "Step 1. ...\nResult: raw_answer\n"
        assert tool_queries == ["raw_answer"]
        assert results[0].step_outputs == ["raw_answer", "docs"]
