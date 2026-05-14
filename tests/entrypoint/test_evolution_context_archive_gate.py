"""EvolutionContext exposes an optional ``archive_gate_provider`` field.

The field is consumed by ``DefaultPipelineBuilder`` when building the
``ArchivePotentialGateStage``. None disables gating (stage fails open).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from gigaevo.database.program_storage import ProgramStorage
from gigaevo.entrypoint.evolution_context import EvolutionContext
from gigaevo.llm.models import MultiModelRouter
from gigaevo.problems.context import ProblemContext
from gigaevo.programs.stages.archive_gate import (
    AllIslandsGateProvider,
    ArchiveGateProvider,
)


def _minimal_ctx_kwargs() -> dict:
    return dict(
        problem_ctx=MagicMock(spec=ProblemContext),
        llm_wrapper=MagicMock(spec=MultiModelRouter),
        storage=MagicMock(spec=ProgramStorage),
    )


def test_archive_gate_provider_defaults_to_none():
    ctx = EvolutionContext(**_minimal_ctx_kwargs())
    assert ctx.archive_gate_provider is None


def test_archive_gate_provider_can_be_set_to_provider_instance():
    provider = AllIslandsGateProvider(islands=[])
    ctx = EvolutionContext(**_minimal_ctx_kwargs(), archive_gate_provider=provider)
    assert isinstance(ctx.archive_gate_provider, ArchiveGateProvider)
    assert ctx.archive_gate_provider is provider
