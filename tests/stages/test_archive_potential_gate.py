"""Unit tests for ArchivePotentialGateStage.

See ``docs/superpowers/specs/2026-05-14-archive-potential-gate-design.md`` for
the design and the full case matrix.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

from pydantic import ValidationError
import pytest

from gigaevo.programs.core_types import ProgramStageResult, StageState
from gigaevo.programs.program import Program
from gigaevo.programs.stages.archive_gate import (
    AllIslandsGateProvider,
    ArchiveGateProvider,
    ArchiveGateTarget,
    ArchivePotentialGateInput,
    ArchivePotentialGateOutput,
    ArchivePotentialGateStage,
)

# --------------------------------------------------------------------------- #
# Stub helpers                                                                #
# --------------------------------------------------------------------------- #


class _StubBehaviorSpace:
    def __init__(
        self,
        cell_fn: Any = None,
        behavior_keys: frozenset[str] = frozenset({"fitness"}),
    ) -> None:
        self._cell_fn = cell_fn or (lambda m: ("c",))
        self.behavior_keys = behavior_keys

    def get_cell(self, metrics: dict[str, float]) -> tuple[Any, ...]:
        return self._cell_fn(metrics)


class _StubArchiveStorage:
    def __init__(
        self,
        elites: dict[tuple[Any, ...], Any] | None = None,
        raises: Exception | None = None,
    ) -> None:
        self._elites = elites or {}
        self._raises = raises

    async def get_elite(self, cell: tuple[Any, ...]) -> Any:
        if self._raises is not None:
            raise self._raises
        return self._elites.get(cell)


def _always_true(_p: Program, _c: Program) -> bool:
    return True


def _always_false(_p: Program, _c: Program) -> bool:
    return False


class _StubIslandConfig:
    """Mirrors ``IslandConfig`` shape: behavior_space + archive_selector
    live here on the real island via ``island.config.<attr>``."""

    def __init__(
        self,
        behavior_space: Any,
        archive_selector: Any,
    ) -> None:
        self.behavior_space = behavior_space
        self.archive_selector = archive_selector


class _StubIsland:
    """Mirrors ``MapElitesIsland`` shape: ``archive_storage`` is a direct
    attribute, everything else is nested under ``.config``. Keeping this
    in sync with the real class is what prevents the regression where
    ``AllIslandsGateProvider`` reached for flat attributes that don't
    exist on the production island.
    """

    def __init__(
        self,
        behavior_space: Any,
        archive_storage: Any,
        archive_selector: Any,
    ) -> None:
        self.archive_storage = archive_storage
        self.config = _StubIslandConfig(
            behavior_space=behavior_space,
            archive_selector=archive_selector,
        )


def _make_program(metrics: dict[str, float] | None = None) -> Any:
    p = MagicMock(spec=Program)
    p.metrics = metrics if metrics is not None else {"fitness": 0.5}
    return p


# --------------------------------------------------------------------------- #
# ArchiveGateTarget                                                            #
# --------------------------------------------------------------------------- #


def test_archive_gate_target_is_frozen_pydantic():
    tgt = ArchiveGateTarget(
        behavior_space=_StubBehaviorSpace(),
        archive_storage=_StubArchiveStorage(),
        archive_selector=_always_true,
        behavior_keys=frozenset({"fitness"}),
    )
    with pytest.raises(ValidationError):
        tgt.behavior_keys = frozenset({"other"})  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# AllIslandsGateProvider                                                       #
# --------------------------------------------------------------------------- #


def test_all_islands_gate_provider_returns_one_target_per_island():
    bs = _StubBehaviorSpace(behavior_keys=frozenset({"f"}))
    i1 = _StubIsland(bs, _StubArchiveStorage(), _always_true)
    i2 = _StubIsland(bs, _StubArchiveStorage(), _always_false)
    provider = AllIslandsGateProvider(islands=[i1, i2])
    targets = list(provider.targets_for(program=None))  # type: ignore[arg-type]
    assert len(targets) == 2
    assert all(isinstance(t, ArchiveGateTarget) for t in targets)
    assert all(t.behavior_keys == frozenset({"f"}) for t in targets)


def test_all_islands_gate_provider_empty_islands_returns_empty():
    provider = AllIslandsGateProvider(islands=[])
    assert list(provider.targets_for(program=None)) == []  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# I/O models                                                                   #
# --------------------------------------------------------------------------- #


def test_io_models_construct():
    inp = ArchivePotentialGateInput()
    assert isinstance(inp, ArchivePotentialGateInput)
    out = ArchivePotentialGateOutput(decision="run", reason="x")
    assert out.decision == "run"
    assert out.reason == "x"


def test_output_decision_field_is_literal_run():
    with pytest.raises(ValidationError):
        ArchivePotentialGateOutput(decision="other", reason="x")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Stage.compute() — fail-open paths                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_compute_provider_none_returns_run_fail_open_no_targets():
    stage = ArchivePotentialGateStage(provider=None, timeout=5.0)
    out = await stage.compute(_make_program())
    assert isinstance(out, ArchivePotentialGateOutput)
    assert out.decision == "run"
    assert out.reason == "fail_open_no_targets"


@pytest.mark.asyncio
async def test_compute_empty_targets_returns_run_fail_open_no_targets():
    class _EmptyProvider(ArchiveGateProvider):
        def targets_for(self, program: Program):
            return []

    stage = ArchivePotentialGateStage(provider=_EmptyProvider(), timeout=5.0)
    out = await stage.compute(_make_program())
    assert out.decision == "run"
    assert out.reason == "fail_open_no_targets"


@pytest.mark.asyncio
async def test_compute_missing_behavior_keys_returns_fail_open_missing_keys():
    tgt = ArchiveGateTarget(
        behavior_space=_StubBehaviorSpace(),
        archive_storage=_StubArchiveStorage(),
        archive_selector=_always_true,
        behavior_keys=frozenset({"missing_key"}),
    )

    class _P(ArchiveGateProvider):
        def targets_for(self, program: Program):
            return [tgt]

    stage = ArchivePotentialGateStage(provider=_P(), timeout=5.0)
    out = await stage.compute(_make_program({"fitness": 0.5}))
    assert out.decision == "run"
    assert out.reason == "fail_open_missing_keys"


@pytest.mark.asyncio
async def test_compute_get_cell_raises_returns_fail_open_target_error():
    bs = _StubBehaviorSpace(cell_fn=MagicMock(side_effect=RuntimeError("boom")))
    tgt = ArchiveGateTarget(
        behavior_space=bs,
        archive_storage=_StubArchiveStorage(),
        archive_selector=_always_true,
        behavior_keys=frozenset({"fitness"}),
    )

    class _P(ArchiveGateProvider):
        def targets_for(self, program: Program):
            return [tgt]

    stage = ArchivePotentialGateStage(provider=_P(), timeout=5.0)
    out = await stage.compute(_make_program())
    assert out.decision == "run"
    assert out.reason == "fail_open_target_error"


@pytest.mark.asyncio
async def test_compute_get_elite_raises_returns_fail_open_target_error():
    tgt = ArchiveGateTarget(
        behavior_space=_StubBehaviorSpace(),
        archive_storage=_StubArchiveStorage(raises=RuntimeError("redis down")),
        archive_selector=_always_true,
        behavior_keys=frozenset({"fitness"}),
    )

    class _P(ArchiveGateProvider):
        def targets_for(self, program: Program):
            return [tgt]

    stage = ArchivePotentialGateStage(provider=_P(), timeout=5.0)
    out = await stage.compute(_make_program())
    assert out.decision == "run"
    assert out.reason == "fail_open_target_error"


# --------------------------------------------------------------------------- #
# Stage.compute() — accept / reject paths                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_compute_empty_cell_returns_accepted_by_some_island():
    tgt = ArchiveGateTarget(
        behavior_space=_StubBehaviorSpace(),
        archive_storage=_StubArchiveStorage(elites={}),
        archive_selector=_always_false,
        behavior_keys=frozenset({"fitness"}),
    )

    class _P(ArchiveGateProvider):
        def targets_for(self, program: Program):
            return [tgt]

    stage = ArchivePotentialGateStage(provider=_P(), timeout=5.0)
    out = await stage.compute(_make_program())
    assert out.decision == "run"
    assert out.reason == "accepted_by_some_island"


@pytest.mark.asyncio
async def test_compute_elite_exists_predicate_true_returns_run():
    elite = _make_program({"fitness": 0.1})
    tgt = ArchiveGateTarget(
        behavior_space=_StubBehaviorSpace(),
        archive_storage=_StubArchiveStorage(elites={("c",): elite}),
        archive_selector=_always_true,
        behavior_keys=frozenset({"fitness"}),
    )

    class _P(ArchiveGateProvider):
        def targets_for(self, program: Program):
            return [tgt]

    stage = ArchivePotentialGateStage(provider=_P(), timeout=5.0)
    out = await stage.compute(_make_program({"fitness": 0.9}))
    assert out.decision == "run"
    assert out.reason == "accepted_by_some_island"


@pytest.mark.asyncio
async def test_compute_elite_exists_predicate_false_returns_skipped():
    elite = _make_program({"fitness": 0.9})
    tgt = ArchiveGateTarget(
        behavior_space=_StubBehaviorSpace(),
        archive_storage=_StubArchiveStorage(elites={("c",): elite}),
        archive_selector=_always_false,
        behavior_keys=frozenset({"fitness"}),
    )

    class _P(ArchiveGateProvider):
        def targets_for(self, program: Program):
            return [tgt]

    stage = ArchivePotentialGateStage(provider=_P(), timeout=5.0)
    result = await stage.compute(_make_program({"fitness": 0.1}))
    assert isinstance(result, ProgramStageResult)
    assert result.status == StageState.SKIPPED
    assert result.error is not None
    assert result.error.message == "dominated_in_all_islands"
    assert result.error.type == "ArchiveGateSkip"


@pytest.mark.asyncio
async def test_compute_first_target_rejects_second_accepts_short_circuits():
    elite = _make_program({"fitness": 0.9})
    rejecting = ArchiveGateTarget(
        behavior_space=_StubBehaviorSpace(cell_fn=lambda m: ("a",)),
        archive_storage=_StubArchiveStorage(elites={("a",): elite}),
        archive_selector=_always_false,
        behavior_keys=frozenset({"fitness"}),
    )
    second_target_calls: list[Any] = []

    def _spy_cell(m: dict[str, float]) -> tuple[Any, ...]:
        second_target_calls.append(m)
        return ("b",)

    accepting = ArchiveGateTarget(
        behavior_space=_StubBehaviorSpace(cell_fn=_spy_cell),
        archive_storage=_StubArchiveStorage(elites={}),
        archive_selector=_always_true,
        behavior_keys=frozenset({"fitness"}),
    )

    class _P(ArchiveGateProvider):
        def targets_for(self, program: Program):
            return [rejecting, accepting]

    stage = ArchivePotentialGateStage(provider=_P(), timeout=5.0)
    out = await stage.compute(_make_program({"fitness": 0.5}))
    assert out.decision == "run"
    assert len(second_target_calls) == 1  # second target was probed


@pytest.mark.asyncio
async def test_compute_all_targets_reject_returns_skipped():
    elite = _make_program({"fitness": 0.9})
    t1 = ArchiveGateTarget(
        behavior_space=_StubBehaviorSpace(cell_fn=lambda m: ("a",)),
        archive_storage=_StubArchiveStorage(elites={("a",): elite}),
        archive_selector=_always_false,
        behavior_keys=frozenset({"fitness"}),
    )
    t2 = ArchiveGateTarget(
        behavior_space=_StubBehaviorSpace(cell_fn=lambda m: ("b",)),
        archive_storage=_StubArchiveStorage(elites={("b",): elite}),
        archive_selector=_always_false,
        behavior_keys=frozenset({"fitness"}),
    )

    class _P(ArchiveGateProvider):
        def targets_for(self, program: Program):
            return [t1, t2]

    stage = ArchivePotentialGateStage(provider=_P(), timeout=5.0)
    result = await stage.compute(_make_program({"fitness": 0.1}))
    assert isinstance(result, ProgramStageResult)
    assert result.status == StageState.SKIPPED
    assert result.error.message == "dominated_in_all_islands"


# Ensure asyncio is imported (used by pytest-asyncio implicitly via decorator).
_ = asyncio  # silence unused-import linters
