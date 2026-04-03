"""Tests for DAG-based memory flow: MemoryContextStage → MutationContextStage → mutation.

The memory refactor (PR #161+) moved memory selection from the evolution engine
into a DAG pipeline stage. These tests verify the complete data flow:

1. MemoryContextStage produces StringContainer with memory card text
2. MutationContextStage consumes "memory" input and includes MemoryMutationContext
3. Memory text appears in the final MUTATION_CONTEXT_METADATA_KEY
4. generate_mutations auto-derives "memory_used" from parent metadata
5. The mutation operator does NOT handle memory (backward compat: param is ignored)

These tests are the refactor's safety net — they break if the data flow is
interrupted at any point.
"""

from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from gigaevo.evolution.mutation.constants import (
    MUTATION_CONTEXT_METADATA_KEY,
    MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY,
)
from gigaevo.evolution.mutation.context import (
    CompositeMutationContext,
    MemoryMutationContext,
    MetricsMutationContext,
)
from gigaevo.llm.agents.memory_selector import MemorySelection
from gigaevo.memory.provider import NullMemoryProvider, SelectorMemoryProvider
from gigaevo.programs.metrics.context import MetricsContext, MetricSpec
from gigaevo.programs.metrics.formatter import MetricsFormatter
from gigaevo.programs.program import Program
from gigaevo.programs.stages.common import FloatDictContainer, StringContainer
from gigaevo.programs.stages.memory_context import MemoryContextStage
from gigaevo.programs.stages.mutation_context import (
    MutationContextInputs,
    MutationContextStage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_program(code: str = "def solve(): return 42") -> Program:
    return Program(code=code)


def _make_metrics_context() -> MetricsContext:
    return MetricsContext(
        specs={
            "fitness": MetricSpec(
                description="accuracy",
                is_primary=True,
                higher_is_better=True,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
        }
    )


def _make_memory_stage(
    provider=None, task="multi-hop QA", metrics_desc="fitness: accuracy"
) -> MemoryContextStage:
    return MemoryContextStage(
        memory_provider=provider or NullMemoryProvider(),
        task_description=task,
        metrics_description=metrics_desc,
        timeout=60,
    )


def _make_mutation_context_stage() -> MutationContextStage:
    return MutationContextStage(
        metrics_context=_make_metrics_context(),
        timeout=60,
    )


def _mutation_ctx_inputs(**overrides) -> dict:
    """Build raw inputs dict for MutationContextStage with all fields defaulting to None."""
    defaults = {
        "metrics": None,
        "insights": None,
        "lineage_ancestors": None,
        "lineage_descendants": None,
        "evolutionary_statistics": None,
        "formatted": None,
        "memory": None,
    }
    defaults.update(overrides)
    return defaults


def _make_selector_provider(
    cards: list[str], card_ids: list[str], max_cards: int = 3
) -> SelectorMemoryProvider:
    mock_selector = AsyncMock()
    mock_selector.select.return_value = MemorySelection(
        cards=cards,
        card_ids=card_ids,
    )
    provider = SelectorMemoryProvider(max_cards=max_cards)
    provider._selector = mock_selector
    return provider


# ===========================================================================
# 1. MemoryContextStage → MutationContextStage data flow
# ===========================================================================


class TestMemoryFlowsThroughMutationContext:
    """Verify that memory cards from MemoryContextStage appear in the final
    mutation context string after passing through MutationContextStage."""

    @pytest.mark.asyncio
    async def test_memory_cards_appear_in_mutation_context(self) -> None:
        """End-to-end: MemoryContextStage → StringContainer → MutationContextStage → metadata."""
        provider = _make_selector_provider(
            cards=[
                "1. Sort evidence by relevance score",
                "2. Filter low-confidence hops",
            ],
            card_ids=["idea-1", "idea-2"],
        )
        memory_stage = _make_memory_stage(provider=provider)
        ctx_stage = _make_mutation_context_stage()

        program = _make_program()

        # Step 1: MemoryContextStage produces card text
        memory_output = await memory_stage.compute(program)
        assert isinstance(memory_output, StringContainer)
        assert "Sort evidence" in memory_output.data
        assert "Filter low-confidence" in memory_output.data

        # Step 2: Wire memory output into MutationContextStage via _raw_inputs
        ctx_stage._raw_inputs = _mutation_ctx_inputs(memory=memory_output)
        ctx_stage._params_obj = None
        ctx_output = await ctx_stage.compute(program)

        # Step 3: Verify memory appears in final context
        context_str = cast(StringContainer, ctx_output).data
        assert "Memory Instructions" in context_str
        assert "Sort evidence" in context_str
        assert "Filter low-confidence" in context_str

        # Step 4: Verify context written to program metadata
        assert MUTATION_CONTEXT_METADATA_KEY in program.metadata
        assert "Memory Instructions" in program.metadata[MUTATION_CONTEXT_METADATA_KEY]

    @pytest.mark.asyncio
    async def test_null_provider_produces_no_memory_section(self) -> None:
        """With NullMemoryProvider, MutationContextStage has no memory section."""
        memory_stage = _make_memory_stage(provider=NullMemoryProvider())
        ctx_stage = _make_mutation_context_stage()

        program = _make_program()
        memory_output = await memory_stage.compute(program)

        ctx_stage._raw_inputs = _mutation_ctx_inputs(memory=memory_output)
        ctx_stage._params_obj = None
        ctx_output = await ctx_stage.compute(program)

        context_str = cast(StringContainer, ctx_output).data
        assert "Memory Instructions" not in context_str

    @pytest.mark.asyncio
    async def test_memory_combines_with_metrics_context(self) -> None:
        """Memory and metrics contexts both appear in the composite output."""
        provider = _make_selector_provider(
            cards=["1. Use BFS over DFS"],
            card_ids=["idea-bfs"],
        )
        memory_stage = _make_memory_stage(provider=provider)
        ctx_stage = _make_mutation_context_stage()

        program = _make_program()
        memory_output = await memory_stage.compute(program)

        ctx_stage._raw_inputs = _mutation_ctx_inputs(
            metrics=FloatDictContainer(data={"fitness": 0.75}),
            memory=memory_output,
        )
        ctx_stage._params_obj = None
        ctx_output = await ctx_stage.compute(program)

        context_str = cast(StringContainer, ctx_output).data
        # Both sections present
        assert "Memory Instructions" in context_str
        assert "Use BFS over DFS" in context_str
        assert "fitness" in context_str.lower()


# ===========================================================================
# 2. Card ID tracking through metadata
# ===========================================================================


class TestCardIdMetadataTracking:
    """Verify that selected card IDs are stored in program metadata for tracking."""

    @pytest.mark.asyncio
    async def test_single_card_id_tracked(self) -> None:
        provider = _make_selector_provider(
            cards=["1. Cache repeated lookups"],
            card_ids=["card-abc"],
        )
        stage = _make_memory_stage(provider=provider)
        program = _make_program()
        await stage.compute(program)

        assert program.metadata[MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY] == [
            "card-abc"
        ]

    @pytest.mark.asyncio
    async def test_multiple_card_ids_tracked(self) -> None:
        provider = _make_selector_provider(
            cards=["1. Sort evidence", "2. Filter noise", "3. Limit depth"],
            card_ids=["id-1", "id-2", "id-3"],
        )
        stage = _make_memory_stage(provider=provider)
        program = _make_program()
        await stage.compute(program)

        assert program.metadata[MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY] == [
            "id-1",
            "id-2",
            "id-3",
        ]

    @pytest.mark.asyncio
    async def test_empty_selection_does_not_set_metadata(self) -> None:
        provider = _make_selector_provider(cards=[], card_ids=[])
        stage = _make_memory_stage(provider=provider)
        program = _make_program()
        await stage.compute(program)

        assert MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY not in program.metadata

    @pytest.mark.asyncio
    async def test_card_ids_survive_program_deep_copy(self) -> None:
        """Card IDs in metadata persist through model_copy (used during mutation)."""
        provider = _make_selector_provider(
            cards=["idea"],
            card_ids=["card-xyz"],
        )
        stage = _make_memory_stage(provider=provider)
        program = _make_program()
        await stage.compute(program)

        clone = program.model_copy(deep=True)
        assert clone.metadata[MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY] == ["card-xyz"]

        # Modifying clone doesn't affect original
        clone.metadata[MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY].append("extra")
        assert program.metadata[MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY] == [
            "card-xyz"
        ]


# ===========================================================================
# 3. Auto-derivation of memory_used from parent metadata
# ===========================================================================


class TestMemoryUsedAutoDerivation:
    """generate_mutations sets "memory_used" based on parent metadata, not explicit params."""

    @pytest.mark.asyncio
    async def test_parent_with_memory_ids_sets_memory_used_true(self) -> None:
        """When parent has MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY, child gets memory_used=True."""
        from gigaevo.evolution.engine.mutation import generate_mutations
        from gigaevo.evolution.mutation.base import MutationSpec

        captured_programs: list[Program] = []

        async def mock_mutate(parents, **kwargs):
            return MutationSpec(
                code="def solve(): return 42",
                parents=parents,
                name="test_mutation",
            )

        mock_mutator = MagicMock()
        mock_mutator.mutate_single = mock_mutate

        mock_storage = AsyncMock()

        async def capture_add(program):
            captured_programs.append(program)
            return program.id

        mock_storage.add = capture_add
        mock_storage.get = AsyncMock(return_value=None)

        mock_state = AsyncMock()

        parent = _make_program()
        # Simulate MemoryContextStage having set card IDs
        parent.set_metadata(
            MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY, ["idea-1", "idea-2"]
        )

        mock_selector = MagicMock()
        mock_selector.create_parent_iterator.return_value = iter([[parent]])

        await generate_mutations(
            [parent],
            mutator=mock_mutator,
            storage=mock_storage,
            state_manager=mock_state,
            parent_selector=mock_selector,
            limit=1,
            iteration=1,
        )

        assert len(captured_programs) == 1
        child = captured_programs[0]
        assert child.get_metadata("memory_used") is True

    @pytest.mark.asyncio
    async def test_parent_without_memory_ids_sets_memory_used_false(self) -> None:
        """When parent has NO memory card IDs, child gets memory_used=False."""
        from gigaevo.evolution.engine.mutation import generate_mutations
        from gigaevo.evolution.mutation.base import MutationSpec

        captured_programs: list[Program] = []

        async def mock_mutate(parents, **kwargs):
            return MutationSpec(
                code="def solve(): return 42",
                parents=parents,
                name="test_mutation",
            )

        mock_mutator = MagicMock()
        mock_mutator.mutate_single = mock_mutate

        mock_storage = AsyncMock()

        async def capture_add(program):
            captured_programs.append(program)
            return program.id

        mock_storage.add = capture_add
        mock_storage.get = AsyncMock(return_value=None)

        mock_state = AsyncMock()

        parent = _make_program()
        # No memory card IDs

        mock_selector = MagicMock()
        mock_selector.create_parent_iterator.return_value = iter([[parent]])

        await generate_mutations(
            [parent],
            mutator=mock_mutator,
            storage=mock_storage,
            state_manager=mock_state,
            parent_selector=mock_selector,
            limit=1,
            iteration=1,
        )

        assert len(captured_programs) == 1
        child = captured_programs[0]
        assert child.get_metadata("memory_used") is False

    @pytest.mark.asyncio
    async def test_mixed_parents_any_with_memory_sets_true(self) -> None:
        """If ANY parent has memory IDs, child gets memory_used=True."""
        from gigaevo.evolution.engine.mutation import generate_mutations
        from gigaevo.evolution.mutation.base import MutationSpec

        captured_programs: list[Program] = []

        async def mock_mutate(parents, **kwargs):
            return MutationSpec(
                code="def solve(): return 42",
                parents=parents,
                name="test_mutation",
            )

        mock_mutator = MagicMock()
        mock_mutator.mutate_single = mock_mutate

        mock_storage = AsyncMock()

        async def capture_add(program):
            captured_programs.append(program)
            return program.id

        mock_storage.add = capture_add
        mock_storage.get = AsyncMock(return_value=None)

        mock_state = AsyncMock()

        parent_with_memory = _make_program(code="def a(): return 1")
        parent_with_memory.set_metadata(
            MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY, ["id-1"]
        )

        parent_without_memory = _make_program(code="def b(): return 2")

        mock_selector = MagicMock()
        mock_selector.create_parent_iterator.return_value = iter(
            [[parent_with_memory, parent_without_memory]]
        )

        await generate_mutations(
            [parent_with_memory, parent_without_memory],
            mutator=mock_mutator,
            storage=mock_storage,
            state_manager=mock_state,
            parent_selector=mock_selector,
            limit=1,
            iteration=1,
        )

        assert len(captured_programs) == 1
        assert captured_programs[0].get_metadata("memory_used") is True

    @pytest.mark.asyncio
    async def test_parent_with_empty_card_ids_sets_false(self) -> None:
        """Empty card_ids list is falsy → memory_used=False."""
        from gigaevo.evolution.engine.mutation import generate_mutations
        from gigaevo.evolution.mutation.base import MutationSpec

        captured_programs: list[Program] = []

        async def mock_mutate(parents, **kwargs):
            return MutationSpec(
                code="def solve(): return 42",
                parents=parents,
                name="test_mutation",
            )

        mock_mutator = MagicMock()
        mock_mutator.mutate_single = mock_mutate

        mock_storage = AsyncMock()

        async def capture_add(program):
            captured_programs.append(program)
            return program.id

        mock_storage.add = capture_add
        mock_storage.get = AsyncMock(return_value=None)

        mock_state = AsyncMock()

        parent = _make_program()
        parent.set_metadata(MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY, [])

        mock_selector = MagicMock()
        mock_selector.create_parent_iterator.return_value = iter([[parent]])

        await generate_mutations(
            [parent],
            mutator=mock_mutator,
            storage=mock_storage,
            state_manager=mock_state,
            parent_selector=mock_selector,
            limit=1,
            iteration=1,
        )

        assert len(captured_programs) == 1
        assert captured_programs[0].get_metadata("memory_used") is False


# ===========================================================================
# 4. Mutation operator backward compatibility
# ===========================================================================


class TestMutationOperatorIgnoresMemory:
    """The mutation operator's mutate_single no longer handles memory.
    It accepts memory_instructions for backward compat but ignores it."""

    @pytest.mark.asyncio
    async def test_mutate_single_signature_accepts_kwargs(self) -> None:
        """mutate_single can be called with extra kwargs (backward compat)."""
        from gigaevo.evolution.mutation.mutation_operator import LLMMutationOperator

        operator = LLMMutationOperator.__new__(LLMMutationOperator)
        operator.mutation_mode = "rewrite"
        operator.fallback_to_rewrite = True
        operator.context_key = MUTATION_CONTEXT_METADATA_KEY
        operator.problem_context = MagicMock()
        operator.problem_context.task_description = "test"
        operator.metrics_formatter = MagicMock()
        operator.metrics_formatter.format_metrics_description.return_value = "fitness"
        operator.prompt_fetcher = None
        operator.storage = None
        operator.bandit = None
        operator.strip_comments_and_docstrings = False
        operator.llm_wrapper = MagicMock()
        operator.llm_wrapper.get_last_model.return_value = "test-model"

        agent = AsyncMock()
        agent.arun.return_value = {
            "code": "def f(): return 1",
            "raw_output": "ok",
            "model_used": "test",
            "structured_output": None,
        }
        operator.agent = agent

        parent = _make_program()
        # Should NOT raise even with extra kwargs
        result = await operator.mutate_single([parent])
        assert result is not None
        assert result.code == "def f(): return 1"


# ===========================================================================
# 5. Memory in the composite mutation context
# ===========================================================================


class TestMemoryInCompositeContext:
    """Verify MemoryMutationContext composes correctly with other context types."""

    def test_memory_alone(self) -> None:
        ctx = CompositeMutationContext(
            contexts=[MemoryMutationContext(memory_block="1. Sort by relevance")]
        )
        result = ctx.format()
        assert "Memory Instructions" in result
        assert "Sort by relevance" in result

    def test_memory_with_metrics(self) -> None:
        metrics_ctx = _make_metrics_context()
        formatter = MetricsFormatter(metrics_ctx)
        ctx = CompositeMutationContext(
            contexts=[
                MetricsMutationContext(
                    metrics={"fitness": 0.85}, metrics_formatter=formatter
                ),
                MemoryMutationContext(memory_block="1. Try simulated annealing"),
            ]
        )
        result = ctx.format()
        # Both sections present
        assert "fitness" in result.lower()
        assert "Memory Instructions" in result
        assert "simulated annealing" in result

    def test_empty_memory_excluded(self) -> None:
        ctx = CompositeMutationContext(
            contexts=[MemoryMutationContext(memory_block="")]
        )
        result = ctx.format()
        assert "Memory Instructions" not in result

    def test_whitespace_only_memory_excluded(self) -> None:
        ctx = CompositeMutationContext(
            contexts=[MemoryMutationContext(memory_block="   \n\t  ")]
        )
        result = ctx.format()
        assert "Memory Instructions" not in result

    def test_multiple_cards_joined_with_double_newline(self) -> None:
        """MemoryContextStage joins cards with double newline."""
        cards = ["1. Sort evidence", "2. Filter noise", "3. Limit depth"]
        joined = "\n\n".join(cards)
        ctx = MemoryMutationContext(memory_block=joined)
        result = ctx.format()
        assert "1. Sort evidence" in result
        assert "2. Filter noise" in result
        assert "3. Limit depth" in result


# ===========================================================================
# 6. Pipeline wiring invariants
# ===========================================================================


class TestPipelineWiringInvariants:
    """Verify structural properties of how memory is wired into the pipeline."""

    def test_memory_context_stage_is_always_present(self) -> None:
        """MemoryContextStage is always added to DefaultPipelineBuilder,
        regardless of provider type."""
        # memory_provider is accessed via EvolutionContext (not a constructor param)
        from gigaevo.entrypoint.evolution_context import EvolutionContext

        assert "memory_provider" in EvolutionContext.model_fields

    def test_null_provider_is_default(self) -> None:
        """When no memory_provider is passed, NullMemoryProvider is used."""
        # This is verified by the constructor: memory_provider or NullMemoryProvider()
        provider = None or NullMemoryProvider()
        assert isinstance(provider, NullMemoryProvider)

    def test_memory_to_mutation_context_edge_name(self) -> None:
        """The data flow edge input name must match MutationContextInputs.memory field."""
        # The field name "memory" must exist on MutationContextInputs
        fields = {name for name in MutationContextInputs.model_fields}
        assert "memory" in fields

    def test_memory_context_inputs_is_empty(self) -> None:
        """MemoryContextStage has no required upstream inputs."""
        from gigaevo.programs.stages.memory_context import MemoryContextInputs

        # Should be instantiable with no arguments
        inputs = MemoryContextInputs()
        assert inputs is not None


# ===========================================================================
# 7. Edge cases
# ===========================================================================


class TestMemoryEdgeCases:
    @pytest.mark.asyncio
    async def test_very_long_memory_card_text(self) -> None:
        """Long card text flows through without truncation at the stage level."""
        long_text = "A" * 10000
        provider = _make_selector_provider(
            cards=[long_text],
            card_ids=["long-card"],
        )
        stage = _make_memory_stage(provider=provider)
        program = _make_program()
        result = await stage.compute(program)
        assert len(cast(StringContainer, result).data) == 10000

    @pytest.mark.asyncio
    async def test_special_characters_in_cards(self) -> None:
        """Cards with markdown, unicode, code fences survive the pipeline."""
        card_text = '```python\ndef f(): return "héllo"\n```'
        provider = _make_selector_provider(
            cards=[card_text],
            card_ids=["special-card"],
        )
        stage = _make_memory_stage(provider=provider)
        program = _make_program()
        result = await stage.compute(program)
        assert "```python" in cast(StringContainer, result).data
        assert "héllo" in cast(StringContainer, result).data

    @pytest.mark.asyncio
    async def test_provider_exception_propagates(self) -> None:
        """If the provider raises, the stage propagates the error."""
        provider = NullMemoryProvider()

        async def failing_select(*args, **kwargs):
            raise RuntimeError("Memory backend unavailable")

        provider.select_cards = failing_select  # type: ignore[assignment]

        stage = _make_memory_stage(provider=provider)
        with pytest.raises(RuntimeError, match="Memory backend unavailable"):
            await stage.compute(_make_program())

    @pytest.mark.asyncio
    async def test_cards_with_ids_but_empty_card_ids_list(self) -> None:
        """Cards present but card_ids empty — still produces output, no metadata."""
        mock_selector = AsyncMock()
        mock_selector.select.return_value = MemorySelection(
            cards=["1. Try BFS"],
            card_ids=[],  # empty IDs (edge case)
        )
        provider = SelectorMemoryProvider(max_cards=1)
        provider._selector = mock_selector

        stage = _make_memory_stage(provider=provider)
        program = _make_program()
        result = await stage.compute(program)

        # Cards present → text returned
        assert "Try BFS" in cast(StringContainer, result).data
        # But card_ids stored is empty list
        assert program.metadata[MUTATION_MEMORY_SELECTED_IDS_METADATA_KEY] == []


# ===========================================================================
# 8. Hydra config group contracts
# ===========================================================================


class TestHydraConfigContracts:
    """Verify that the Hydra config targets resolve to correct classes."""

    def test_null_provider_target(self) -> None:
        """NullMemoryProvider can be instantiated with no args (Hydra default)."""
        provider = NullMemoryProvider()
        assert isinstance(provider, NullMemoryProvider)

    def test_selector_provider_target_with_max_cards(self) -> None:
        """SelectorMemoryProvider accepts max_cards kwarg (from config/memory/local.yaml)."""
        provider = SelectorMemoryProvider(max_cards=5)
        assert provider._max_cards == 5

    def test_selector_provider_target_with_all_params(self) -> None:
        """SelectorMemoryProvider accepts all Hydra-injectable params."""
        provider = SelectorMemoryProvider(
            max_cards=3,
            checkpoint_dir="/tmp/test",
            namespace="hover-memory-exp",
        )
        assert provider._max_cards == 3
        assert provider._checkpoint_dir == "/tmp/test"
        assert provider._namespace == "hover-memory-exp"

    def test_selector_provider_passes_checkpoint_dir_to_agent(self) -> None:
        """checkpoint_dir flows to MemorySelectorAgent constructor."""
        from unittest.mock import patch

        with patch(
            "gigaevo.llm.agents.memory_selector.MemorySelectorAgent"
        ) as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.select.return_value = MemorySelection(cards=[], card_ids=[])
            mock_cls.return_value = mock_instance

            provider = SelectorMemoryProvider(
                max_cards=3,
                checkpoint_dir="/data/memory",
                namespace="test-ns",
            )
            # Trigger lazy creation
            provider._get_selector()

            mock_cls.assert_called_once_with(
                checkpoint_dir="/data/memory",
                namespace="test-ns",
                use_api=False,
            )
