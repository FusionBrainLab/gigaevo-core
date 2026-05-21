"""Tests for MutationAgent: code extraction, diff, prompt building, parsing, LLM calls."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.evolution.mutation.constants import (
    MUTATION_CONTEXT_METADATA_KEY,
    MUTATION_MEMORY_METADATA_KEY,
)
from gigaevo.llm.agents.mutation import (
    MutationAgent,
    MutationPromptFields,
    MutationState,
    MutationStructuredOutput,
)
from gigaevo.programs.program import Program
from gigaevo.prompts.fetcher import FetchedPrompt, PromptFetcher

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

    @pytest.mark.asyncio
    async def test_success_emits_llm_call_ok(self):
        """Successful structured-LLM call emits exactly one LLM_CALL with ok=True."""
        import json
        import re

        from loguru import logger

        captured: list[str] = []
        sink_id = logger.add(
            lambda m: captured.append(str(m)), level="DEBUG", format="{message}"
        )
        try:
            agent = _make_agent()
            expected = _make_structured_output()
            agent.structured_llm = MagicMock()
            agent.structured_llm.ainvoke = AsyncMock(return_value=expected)

            from langchain_core.messages import HumanMessage

            state = _make_state()
            state["messages"] = [HumanMessage(content="test")]

            await agent.acall_llm(state)
        finally:
            logger.remove(sink_id)

        lines = [m for m in captured if "[LLM_CALL]" in m]
        assert len(lines) == 1, f"expected 1 LLM_CALL, got {lines}"
        body = json.loads(re.search(r"\{.*\}\s*$", lines[0]).group(0))
        assert body["event"] == "LLM_CALL"
        assert body["stage"] == "MutationAgent"
        assert body["ok"] is True
        assert body["latency_ms"] >= 0.0
        assert body["error_type"] is None

    @pytest.mark.asyncio
    async def test_exception_emits_llm_call_with_error_type(self):
        """Failed structured-LLM call emits LLM_CALL with ok=False and error_type set."""
        import json
        import re

        from loguru import logger

        captured: list[str] = []
        sink_id = logger.add(
            lambda m: captured.append(str(m)), level="DEBUG", format="{message}"
        )
        try:
            agent = _make_agent()
            agent.structured_llm = MagicMock()
            agent.structured_llm.ainvoke = AsyncMock(
                side_effect=RuntimeError("LLM exploded")
            )

            from langchain_core.messages import HumanMessage

            state = _make_state()
            state["messages"] = [HumanMessage(content="test")]

            await agent.acall_llm(state)
        finally:
            logger.remove(sink_id)

        lines = [m for m in captured if "[LLM_CALL]" in m]
        assert len(lines) == 1, f"expected 1 LLM_CALL, got {lines}"
        body = json.loads(re.search(r"\{.*\}\s*$", lines[0]).group(0))
        assert body["event"] == "LLM_CALL"
        assert body["stage"] == "MutationAgent"
        assert body["ok"] is False
        assert body["error_type"] == "RuntimeError"


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
            archetype="Precision Optimization",
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

    def test_missing_mutation_context_key_produces_empty_context(self):
        """When MUTATION_CONTEXT_METADATA_KEY is absent, formatted_context is empty string."""
        agent = _make_agent()
        parent = _make_program(code="def solve(): return 1", metadata={})
        state = _make_state(parents=[parent])

        result = agent.build_prompt(state)
        user_content = result["messages"][1].content
        # metadata.get(key) returns None → or "" → no "None" literal in prompt
        assert "None" not in user_content

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


# ---------------------------------------------------------------------------
# TestFixJsonEscapedCode
# ---------------------------------------------------------------------------


class TestFixJsonEscapedCode:
    """Tests for MutationAgent._fix_json_escaped_code (static method)."""

    def test_no_escape_sequences_returns_early_unchanged(self):
        """Code with no JSON escape sequences skips all parsing and is returned as-is."""
        code = "def solve():\n    return 42"
        assert MutationAgent._fix_json_escaped_code(code) == code

    def test_already_valid_python_with_literal_backslash_n_unchanged(self):
        """Valid Python that contains a literal \\n inside a string is returned unchanged."""
        # The code has a real newline for structure AND a two-char \\n inside a string literal.
        # ast.parse succeeds → no transformation applied.
        code = 'def solve():\n    msg = "hello\\nworld"\n    return msg'
        assert MutationAgent._fix_json_escaped_code(code) == code

    def test_json_escaped_newlines_are_fixed(self):
        """Two-char \\n sequences (JSON escaping) are converted to real newlines."""
        # LLM produced \\n (two chars) instead of actual newlines → invalid Python.
        # After replace("\\n", "\n") → valid Python → return cleaned.
        broken = "def solve():\\n    return 42"
        result = MutationAgent._fix_json_escaped_code(broken)
        assert result == "def solve():\n    return 42"

    def test_json_escaped_quotes_are_fixed(self):
        """Two-char \\" sequences (JSON escaping) are converted to real quote chars."""
        broken = 'def solve():\\n    return \\"hello\\"'
        result = MutationAgent._fix_json_escaped_code(broken)
        assert result == 'def solve():\n    return "hello"'

    def test_unfixable_code_returned_unchanged(self):
        """Code that remains invalid even after unescaping is returned as-is."""
        broken = "\\n!!! not valid python !!!\\n"
        assert MutationAgent._fix_json_escaped_code(broken) == broken


# ---------------------------------------------------------------------------
# TestBuildMemoryBlock
# ---------------------------------------------------------------------------


class TestBuildMemoryBlock:
    """Tests for MutationAgent._build_memory_block."""

    def setup_method(self):
        self.agent = _make_agent()

    def test_no_memory_key_returns_empty_string(self):
        """Parents with no memory metadata key produce an empty string."""
        parents = [_make_program(metadata={}), _make_program(metadata={})]
        assert self.agent._build_memory_block(parents) == ""

    def test_first_parent_with_memory_key_wins(self):
        """The first parent that has a non-empty memory key is used; later parents ignored."""
        parents = [
            _make_program(metadata={MUTATION_MEMORY_METADATA_KEY: "Use caching."}),
            _make_program(
                metadata={MUTATION_MEMORY_METADATA_KEY: "Should be ignored."}
            ),
        ]
        result = self.agent._build_memory_block(parents)
        assert result == "## Memory Instructions\nUse caching."
        assert "ignored" not in result

    def test_whitespace_only_memory_value_treated_as_absent(self):
        """A memory value that is all whitespace is skipped (treated as no memory)."""
        parents = [_make_program(metadata={MUTATION_MEMORY_METADATA_KEY: "   "})]
        assert self.agent._build_memory_block(parents) == ""


# ---------------------------------------------------------------------------
# TestBuildUserPromptWithMemory
# ---------------------------------------------------------------------------


class TestBuildUserPromptWithMemory:
    """Tests for build_user_prompt — memory block integration."""

    def test_memory_block_appended_when_present(self):
        """When a parent has memory instructions, they appear in the user prompt."""
        agent = _make_agent()
        parent = _make_program(
            code="def solve(): return 1",
            metadata={
                MUTATION_CONTEXT_METADATA_KEY: "score=0.9",
                MUTATION_MEMORY_METADATA_KEY: "Prefer vectorised ops.",
            },
        )
        result = agent.build_user_prompt([parent])
        assert "## Memory Instructions" in result
        assert "Prefer vectorised ops." in result

    def test_no_memory_block_when_absent(self):
        """When no parent has memory instructions, the memory section is absent."""
        agent = _make_agent()
        parent = _make_program(
            code="def solve(): return 1",
            metadata={MUTATION_CONTEXT_METADATA_KEY: "score=0.9"},
        )
        result = agent.build_user_prompt([parent])
        assert "## Memory Instructions" not in result


# ---------------------------------------------------------------------------
# TestDynamicPromptFetcher
# ---------------------------------------------------------------------------


class TestDynamicPromptFetcher:
    """Tests for the dynamic prompt_fetcher path inside build_prompt."""

    def _make_dynamic_fetcher(
        self,
        system_text: str = "Dynamic: {task_description} {metrics_description}",
        prompt_id: str = "abc123def456",
        user_prompt_id: str | None = None,
    ) -> MagicMock:
        """Return a mock PromptFetcher with is_dynamic=True."""
        fetcher = MagicMock(spec=PromptFetcher)
        fetcher.is_dynamic = True

        def _fetch(agent_name: str, prompt_type: str) -> FetchedPrompt:
            if prompt_type == "system":
                return FetchedPrompt(text=system_text, prompt_id=prompt_id)
            return FetchedPrompt(
                text="user template {count} {parent_blocks}", prompt_id=user_prompt_id
            )

        fetcher.fetch.side_effect = _fetch
        return fetcher

    def test_dynamic_fetcher_refreshes_system_prompt_and_stamps_prompt_id(self):
        """Dynamic fetcher: system prompt is refreshed and prompt_id stamped in state."""
        fetcher = self._make_dynamic_fetcher(
            system_text="Dynamic: {task_description} {metrics_description}",
            prompt_id="abc123def456",
        )
        agent = _make_agent(system_prompt="original static prompt")
        agent._prompt_fetcher = fetcher
        agent._task_description = "solve problems"
        agent._metrics_formatter = MagicMock()
        agent._metrics_formatter.format_metrics_description.return_value = (
            "fitness: 0-1"
        )

        state = _make_state(parents=[_make_program()])
        result = agent.build_prompt(state)

        assert result["prompt_id"] == "abc123def456"
        assert "Dynamic: solve problems fitness: 0-1" in result["system_prompt"]
        assert "original static prompt" not in result["system_prompt"]

    def test_non_dynamic_fetcher_leaves_prompt_unchanged_and_sets_prompt_id_none(self):
        """Non-dynamic (fixed) fetcher: system prompt unchanged, prompt_id=None."""
        fetcher = MagicMock(spec=PromptFetcher)
        fetcher.is_dynamic = False

        agent = _make_agent(system_prompt="static prompt")
        agent._prompt_fetcher = fetcher

        state = _make_state(parents=[_make_program()])
        result = agent.build_prompt(state)

        assert result["prompt_id"] is None
        assert result["system_prompt"] == "static prompt"
        fetcher.fetch.assert_not_called()


# ---------------------------------------------------------------------------
# TestJsonTemplateGuard
# ---------------------------------------------------------------------------


class TestJsonTemplateGuard:
    """Tests for the JSON-template guard in parse_response."""

    def test_json_template_echoed_as_code_is_rejected(self):
        """When LLM returns a JSON object instead of Python, parse_response captures the error."""
        agent = _make_agent(mutation_mode="rewrite")
        output = _make_structured_output(code='{"archetype": "x", "code": "..."}')
        state = _make_state(mutation_mode="rewrite", structured_output=output)

        result = agent.parse_response(state)

        assert result["parsed_output"]["code"] == ""
        assert "JSON template" in result["parsed_output"]["error"]

    def test_valid_python_starting_with_brace_is_not_rejected(self):
        """A dict literal assigned to a variable is valid Python and must not be rejected."""
        agent = _make_agent(mutation_mode="rewrite")
        output = _make_structured_output(
            code='CONFIG = {"key": 1}\n\ndef solve(x):\n    return CONFIG["key"] + x'
        )
        state = _make_state(mutation_mode="rewrite", structured_output=output)

        result = agent.parse_response(state)

        assert result["parsed_output"]["code"] != ""
        assert "error" not in result["parsed_output"]
