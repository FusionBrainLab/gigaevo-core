"""Tests for MutationAgent: code extraction, diff, prompt building, parsing, LLM calls."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.evolution.mutation.constants import MUTATION_CONTEXT_METADATA_KEY
from gigaevo.llm.agents.mutation import (
    MutationAgent,
    MutationPromptFields,
    MutationState,
    MutationStructuredOutput,
)
from gigaevo.programs.program import Program

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    mutation_mode: str = "rewrite",
    system_prompt: str = "You are a mutation agent.",
    user_prompt_template: str = "Mutate {count} parent programs:\n{parent_blocks}",
) -> MutationAgent:
    """Create a MutationAgent with a fully mocked LLM."""
    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=MagicMock())
    return MutationAgent(
        llm=mock_llm,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        mutation_mode=mutation_mode,
    )


def _make_program(
    code: str = "def solve(): return 42",
    metadata: dict[str, Any] | None = None,
) -> Program:
    """Create a minimal Program for tests."""
    p = Program(code=code)
    if metadata:
        p.metadata = metadata
    return p


def _make_structured_output(**kwargs) -> MutationStructuredOutput:
    defaults = {
        "archetype": "Precision Optimization",
        "justification": "Improved via targeted mutation.",
        "insights_used": ["insight_a"],
        "code": "def solve(): return 99",
    }
    defaults.update(kwargs)
    return MutationStructuredOutput(**defaults)


def _make_state(
    parents: list[Program] | None = None,
    mutation_mode: str = "rewrite",
    **overrides: Any,
) -> MutationState:
    """Build a MutationState dict with sensible defaults."""
    state: MutationState = {
        "input": parents or [_make_program()],
        "mutation_mode": mutation_mode,
        "messages": [],
        "llm_response": None,
        "final_code": "",
        "mutation_label": "",
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


# ---------------------------------------------------------------------------
# TestExtractCodeBlock
# ---------------------------------------------------------------------------


class TestExtractCodeBlock:
    """Tests for MutationAgent._extract_code_block."""

    def setup_method(self):
        self.agent = _make_agent()

    def test_fenced_python(self):
        """Standard fenced code block with python language tag."""
        text = "Some text\n```python\ndef solve(): return 1\n```\nMore text"
        result = self.agent._extract_code_block(text)
        assert result == "def solve(): return 1"

    def test_no_fence(self):
        """Plain text without fences is returned stripped."""
        text = "  def solve(): return 1  "
        result = self.agent._extract_code_block(text)
        assert result == "def solve(): return 1"

    def test_indented_fence_ignored(self):
        """Fences not at start-of-line are ignored (regex requires ^)."""
        text = "  ```python\ndef solve(): return 1\n  ```"
        result = self.agent._extract_code_block(text)
        # Indented fences don't match, so entire text is returned stripped
        assert result == text.strip()

    def test_backticks_in_code(self):
        """Backticks inside the code (e.g. docstrings) don't close the block."""
        inner = 'def solve():\n    """Uses `x` and `y`."""\n    return 1'
        text = f"```python\n{inner}\n```"
        result = self.agent._extract_code_block(text)
        assert result == inner

    def test_missing_close(self):
        """If closing fence is absent, entire text is returned stripped."""
        text = "```python\ndef solve(): return 1"
        result = self.agent._extract_code_block(text)
        assert result == text.strip()

    def test_multiple_blocks_takes_first(self):
        """Only the first (outermost) fenced block is extracted."""
        text = "```python\nfirst_block\n```\n\n```python\nsecond_block\n```"
        result = self.agent._extract_code_block(text)
        assert result == "first_block"


# ---------------------------------------------------------------------------
# TestApplyDiffAndExtract
# ---------------------------------------------------------------------------


class TestApplyDiffAndExtract:
    """Tests for MutationAgent._apply_diff_and_extract."""

    def setup_method(self):
        self.agent = _make_agent()

    def test_valid_diff(self):
        """A correct unified diff is applied to original code."""
        original = "line1\nline2\nline3\n"
        diff = (
            "--- a/file\n+++ b/file\n@@ -1,3 +1,3 @@\n line1\n-line2\n+lineX\n line3\n"
        )
        fenced = f"```diff\n{diff}```"
        result = self.agent._apply_diff_and_extract(original, fenced)
        assert "lineX" in result
        assert "line2" not in result

    def test_empty_diff_raises(self):
        """An empty diff raises ValueError."""
        original = "line1\nline2\n"
        with pytest.raises(ValueError, match="Empty diff"):
            self.agent._apply_diff_and_extract(original, "```\n   \n```")

    def test_invalid_diff_raises(self):
        """A malformed diff raises ValueError about patch failure."""
        original = "line1\nline2\n"
        bad_diff = "```diff\nthis is not a diff\n```"
        with pytest.raises(ValueError, match="Failed to apply patch"):
            self.agent._apply_diff_and_extract(original, bad_diff)


# ---------------------------------------------------------------------------
# TestBuildPrompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    """Tests for MutationAgent.build_prompt."""

    def test_system_and_human_messages(self):
        """build_prompt produces SystemMessage + HumanMessage."""
        from langchain_core.messages import HumanMessage, SystemMessage

        agent = _make_agent(system_prompt="SYS")
        parent = _make_program(metadata={MUTATION_CONTEXT_METADATA_KEY: "context info"})
        state = _make_state(parents=[parent])

        result = agent.build_prompt(state)

        msgs = result["messages"]
        assert len(msgs) == 2
        assert isinstance(msgs[0], SystemMessage)
        assert isinstance(msgs[1], HumanMessage)
        assert msgs[0].content == "SYS"

    def test_parent_blocks_content(self):
        """Parent code and mutation context appear in user prompt."""
        agent = _make_agent()
        parent = _make_program(
            code="def solve(): return 1",
            metadata={MUTATION_CONTEXT_METADATA_KEY: "metrics: score=0.9"},
        )
        state = _make_state(parents=[parent])

        result = agent.build_prompt(state)

        user_content = result["messages"][1].content
        assert "def solve(): return 1" in user_content
        assert "metrics: score=0.9" in user_content
        assert "=== Parent 1 ===" in user_content

    def test_count_substitution(self):
        """Template {count} is replaced with number of parents."""
        agent = _make_agent()
        parents = [
            _make_program(metadata={MUTATION_CONTEXT_METADATA_KEY: f"ctx{i}"})
            for i in range(3)
        ]
        state = _make_state(parents=parents)

        result = agent.build_prompt(state)

        user_content = result["messages"][1].content
        assert "Mutate 3 parent programs:" in user_content

    def test_template_substitution_custom(self):
        """Custom template with {count} and {parent_blocks} placeholders."""
        agent = _make_agent(
            user_prompt_template="Process {count} programs.\n{parent_blocks}\nDone."
        )
        parent = _make_program(metadata={MUTATION_CONTEXT_METADATA_KEY: "some context"})
        state = _make_state(parents=[parent])

        result = agent.build_prompt(state)

        user_content = result["messages"][1].content
        assert user_content.startswith("Process 1 programs.")
        assert user_content.endswith("Done.")


# ---------------------------------------------------------------------------
# TestParseResponse
# ---------------------------------------------------------------------------


class TestParseResponse:
    """Tests for MutationAgent.parse_response."""

    def test_rewrite_mode(self):
        """In rewrite mode, code is extracted from structured output directly."""
        agent = _make_agent(mutation_mode="rewrite")
        output = _make_structured_output(code="def solve(): return 99")
        state = _make_state(
            mutation_mode="rewrite",
            structured_output=output,
        )

        result = agent.parse_response(state)

        assert result["parsed_output"]["code"] == "def solve(): return 99"
        assert result["parsed_output"]["archetype"] == "Precision Optimization"
        assert (
            result["parsed_output"]["justification"]
            == "Improved via targeted mutation."
        )
        assert result["parsed_output"]["insights_used"] == ["insight_a"]

    def test_diff_mode(self):
        """In diff mode, the code field is treated as a diff applied to parent."""
        agent = _make_agent(mutation_mode="diff")
        original = "line1\nline2\nline3\n"
        diff_str = (
            "--- a/file\n+++ b/file\n@@ -1,3 +1,3 @@\n line1\n-line2\n+lineX\n line3\n"
        )
        parent = _make_program(code=original)
        output = _make_structured_output(code=diff_str)
        state = _make_state(
            parents=[parent],
            mutation_mode="diff",
            structured_output=output,
        )

        result = agent.parse_response(state)

        assert "lineX" in result["parsed_output"]["code"]
        assert "line2" not in result["parsed_output"]["code"]

    def test_no_output(self):
        """When structured_output is None, parsed_output has empty code + error."""
        agent = _make_agent()
        state = _make_state()
        # No structured_output set

        result = agent.parse_response(state)

        assert result["parsed_output"]["code"] == ""
        assert "error" in result["parsed_output"]

    def test_diff_multi_parents_raises(self):
        """Diff mode with >1 parent raises ValueError (stored in error)."""
        agent = _make_agent(mutation_mode="diff")
        parents = [_make_program(), _make_program()]
        output = _make_structured_output(code="some diff text")
        state = _make_state(
            parents=parents,
            mutation_mode="diff",
            structured_output=output,
        )

        result = agent.parse_response(state)

        assert result["parsed_output"]["code"] == ""
        assert "exactly 1 parent" in result["parsed_output"]["error"]


# ---------------------------------------------------------------------------
# TestAcallLlm
# ---------------------------------------------------------------------------


class TestAcallLlm:
    """Tests for MutationAgent.acall_llm."""

    @pytest.mark.asyncio
    async def test_success(self):
        """Successful LLM call populates llm_response and structured_output."""
        agent = _make_agent()
        expected = _make_structured_output()
        agent.structured_llm = AsyncMock(return_value=expected)
        agent.structured_llm.ainvoke = AsyncMock(return_value=expected)

        state = _make_state()
        from langchain_core.messages import HumanMessage

        state["messages"] = [HumanMessage(content="test")]

        result = await agent.acall_llm(state)

        assert result["llm_response"] is expected
        assert result["structured_output"] is expected

    @pytest.mark.asyncio
    async def test_success_forwards_messages_to_llm(self):
        """acall_llm passes the exact messages list to structured_llm.ainvoke."""
        agent = _make_agent()
        expected = _make_structured_output()
        agent.structured_llm = MagicMock()
        agent.structured_llm.ainvoke = AsyncMock(return_value=expected)

        from langchain_core.messages import HumanMessage, SystemMessage

        msgs = [SystemMessage(content="sys"), HumanMessage(content="user")]
        state = _make_state()
        state["messages"] = msgs

        await agent.acall_llm(state)

        agent.structured_llm.ainvoke.assert_awaited_once_with(msgs)

    @pytest.mark.asyncio
    async def test_exception_sets_error(self):
        """When the LLM raises, state gets an error field and llm_response is None."""
        agent = _make_agent()
        agent.structured_llm = MagicMock()
        agent.structured_llm.ainvoke = AsyncMock(
            side_effect=RuntimeError("LLM exploded")
        )

        state = _make_state()
        from langchain_core.messages import HumanMessage

        state["messages"] = [HumanMessage(content="test")]

        result = await agent.acall_llm(state)

        assert result["llm_response"] is None
        assert "LLM exploded" in result["error"]


# ---------------------------------------------------------------------------
# TestArun
# ---------------------------------------------------------------------------


class TestArun:
    """Tests for MutationAgent.arun end-to-end."""

    @pytest.mark.asyncio
    async def test_end_to_end_mocked_graph(self):
        """arun invokes the graph and returns parsed_output from final state."""
        agent = _make_agent()

        expected_parsed = {
            "code": "def solve(): return 99",
            "structured_output": {"archetype": "test"},
            "archetype": "test",
            "justification": "test justification",
            "insights_used": [],
        }

        # Mock the compiled graph's ainvoke to return a state with parsed_output
        agent.graph = AsyncMock()
        agent.graph.ainvoke = AsyncMock(return_value={"parsed_output": expected_parsed})

        parent = _make_program(metadata={MUTATION_CONTEXT_METADATA_KEY: "ctx"})
        result = await agent.arun(input=[parent], mutation_mode="rewrite")

        assert result == expected_parsed
        agent.graph.ainvoke.assert_called_once()

        # Verify the initial state structure passed to graph
        call_args = agent.graph.ainvoke.call_args[0][0]
        assert call_args["input"] == [parent]
        assert call_args["mutation_mode"] == "rewrite"
        assert call_args["messages"] == []
        assert call_args["final_code"] == ""

    @pytest.mark.asyncio
    async def test_arun_returns_empty_dict_when_no_parsed_output(self):
        """When graph returns state without parsed_output, arun returns {}."""
        agent = _make_agent()
        agent.graph = AsyncMock()
        agent.graph.ainvoke = AsyncMock(return_value={"error": "something"})

        parent = _make_program(metadata={MUTATION_CONTEXT_METADATA_KEY: "ctx"})
        result = await agent.arun(input=[parent], mutation_mode="rewrite")
        assert result == {"prompt_id": None}


# ---------------------------------------------------------------------------
# TestMutationStructuredOutput
# ---------------------------------------------------------------------------


class TestMutationStructuredOutput:
    """Tests for the MutationStructuredOutput Pydantic model."""

    def test_defaults(self):
        """List fields default to empty lists."""
        out = MutationStructuredOutput(
            archetype="test",
            justification="just",
            code="print(1)",
        )
        assert out.insights_used == []
        assert out.changes == []

    def test_model_dump(self):
        """model_dump returns all fields."""
        out = _make_structured_output()
        d = out.model_dump()
        assert set(d.keys()) == {
            "archetype",
            "justification",
            "insights_used",
            "changes",
            "code",
        }
        assert d["changes"] == []


# ---------------------------------------------------------------------------
# TestMutationPromptFields
# ---------------------------------------------------------------------------


class TestMutationPromptFields:
    """Tests for MutationPromptFields validation."""

    def test_valid(self):
        fields = MutationPromptFields(count=2, parent_blocks="block1\nblock2")
        assert fields.count == 2
        assert "block1" in fields.parent_blocks


# ---------------------------------------------------------------------------
# TestBuildPromptEdgeCases
# ---------------------------------------------------------------------------


class TestBuildPromptEdgeCases:
    """Edge cases for build_prompt."""

    def test_missing_mutation_context_key_produces_none_text(self):
        """When MUTATION_CONTEXT_METADATA_KEY is absent, formatted_context is None.

        This means "None" appears literally in the prompt — test the actual output.
        """
        agent = _make_agent()
        parent = _make_program(code="def solve(): return 1", metadata={})
        state = _make_state(parents=[parent])

        result = agent.build_prompt(state)
        user_content = result["messages"][1].content
        # metadata.get(key) returns None → formatted as "None"
        assert "None" in user_content

    def test_multiple_parents_count(self):
        """build_prompt with 3 parents produces count=3."""
        agent = _make_agent()
        parents = [
            _make_program(metadata={MUTATION_CONTEXT_METADATA_KEY: f"ctx{i}"})
            for i in range(3)
        ]
        state = _make_state(parents=parents)

        result = agent.build_prompt(state)
        user_content = result["messages"][1].content
        assert "Mutate 3 parent programs:" in user_content
        assert "=== Parent 1 ===" in user_content
        assert "=== Parent 2 ===" in user_content
        assert "=== Parent 3 ===" in user_content


# ---------------------------------------------------------------------------
# TestParseResponseEdgeCases
# ---------------------------------------------------------------------------


class TestParseResponseEdgeCases:
    """Edge cases for parse_response."""

    def test_rewrite_with_fenced_code(self):
        """In rewrite mode, code surrounded by fences is extracted properly."""
        agent = _make_agent(mutation_mode="rewrite")
        output = _make_structured_output(code="```python\ndef solve(): return 99\n```")
        state = _make_state(mutation_mode="rewrite", structured_output=output)

        result = agent.parse_response(state)
        assert result["parsed_output"]["code"] == "def solve(): return 99"

    def test_error_in_diff_application_stored_in_parsed_output(self):
        """When diff application fails, error is captured in parsed_output."""
        agent = _make_agent(mutation_mode="diff")
        parent = _make_program(code="original code\n")
        output = _make_structured_output(code="this is not a valid diff")
        state = _make_state(
            parents=[parent], mutation_mode="diff", structured_output=output
        )

        result = agent.parse_response(state)
        assert result["parsed_output"]["code"] == ""
        assert "error" in result["parsed_output"]
