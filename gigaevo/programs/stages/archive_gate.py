"""Archive-potential gate for InsightsStage.

Skips the (heavy) ``InsightsStage`` LLM call when the candidate program
cannot displace the current elite in any island's archive. Reuses the same
``archive_selector`` predicate the archive uses, so the gate cannot diverge
from the real insertion decision.

See ``docs/superpowers/specs/2026-05-14-archive-potential-gate-design.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict

from gigaevo.programs.core_types import ProgramStageResult, StageIO
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.stage_registry import StageRegistry


class ArchiveGateTarget(BaseModel):
    """A single ``(behavior_space, archive_storage, archive_selector)`` triple.

    The gate stage probes each target in turn: a target ``accepts`` the
    candidate iff its cell is empty OR ``archive_selector(candidate, current)``
    returns True. The gate skips downstream only when *every* target rejects.
    """

    behavior_space: Any
    archive_storage: Any
    archive_selector: Callable[[Program, Program], bool]
    behavior_keys: frozenset[str]

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


class ArchiveGateProvider(ABC):
    """Returns gate targets for a candidate program.

    An empty list means fail-open (the gate will not skip).
    """

    @abstractmethod
    def targets_for(self, program: Program) -> Sequence[ArchiveGateTarget]: ...


class AllIslandsGateProvider(ArchiveGateProvider):
    """Returns one target per configured island.

    At gate time the program is not yet routed to an island (``home_island``
    metadata is set *after* archive insertion), so we check every island and
    only skip if *all* of them reject. This is fail-safe: a false-negative
    (skipping a program that would have been accepted somewhere) cannot
    happen, while false-positives (running insights for a program that ends
    up rejected) are bounded.
    """

    def __init__(self, islands: Sequence[Any]) -> None:
        self._islands = list(islands)

    def targets_for(self, program: Program) -> Sequence[ArchiveGateTarget]:
        # Production ``MapElitesIsland`` keeps behavior_space + archive_selector
        # under ``island.config``; only ``archive_storage`` is a flat attribute
        # (see gigaevo/evolution/strategies/island.py). Reaching for flat
        # ``island.behavior_space`` crashed every gate evaluation.
        targets: list[ArchiveGateTarget] = []
        for island in self._islands:
            cfg = island.config
            targets.append(
                ArchiveGateTarget(
                    behavior_space=cfg.behavior_space,
                    archive_storage=island.archive_storage,
                    archive_selector=cfg.archive_selector,
                    behavior_keys=frozenset(cfg.behavior_space.behavior_keys),
                )
            )
        return targets


class ArchivePotentialGateInput(StageIO):
    """No declared inputs; ``Stage.execute()`` injects the program directly."""


class ArchivePotentialGateOutput(StageIO):
    """Output for the "run insights" branch.

    The SKIPPED branch is signaled via ``ProgramStageResult.skipped(...)``
    returned from ``compute()`` — not via this model.
    """

    decision: Literal["run"]
    reason: str


@StageRegistry.register(description="Gate insights on archive insertion potential")
class ArchivePotentialGateStage(Stage):
    """Skip downstream LLM stages when no island would accept this program.

    Returns ``ProgramStageResult.skipped(...)`` when every target rejects;
    the existing automata cascade then auto-skips any
    ``on_success(ArchivePotentialGateStage)`` dependent (notably
    ``InsightsStage``). ``always_after`` dependents (``LineageStage``,
    ``MutationContextStage``) still run because their lineage records are
    consumed by sibling/descendant programs even when the candidate itself
    is rejected.

    Fail-open in every ambiguous case: missing provider, empty targets,
    missing behavior keys, or any target-side exception → returns
    ``decision="run"``.
    """

    InputsModel: type[StageIO] = ArchivePotentialGateInput
    OutputModel: type[StageIO] = ArchivePotentialGateOutput

    def __init__(
        self,
        *,
        provider: ArchiveGateProvider | None,
        timeout: float,
    ) -> None:
        super().__init__(timeout=timeout)
        self._provider = provider

    async def compute(
        self, program: Program
    ) -> ArchivePotentialGateOutput | ProgramStageResult:
        targets = self._provider.targets_for(program) if self._provider else ()
        if not targets:
            return ArchivePotentialGateOutput(
                decision="run", reason="fail_open_no_targets"
            )

        for tgt in targets:
            if tgt.behavior_keys - set(program.metrics):
                return ArchivePotentialGateOutput(
                    decision="run", reason="fail_open_missing_keys"
                )
            try:
                cell = tgt.behavior_space.get_cell(program.metrics)
                current = await tgt.archive_storage.get_elite(cell)
            except Exception as e:
                logger.warning("[ArchiveGate] target error: {}", e)
                return ArchivePotentialGateOutput(
                    decision="run", reason="fail_open_target_error"
                )
            if current is None or tgt.archive_selector(program, current):
                return ArchivePotentialGateOutput(
                    decision="run", reason="accepted_by_some_island"
                )

        return ProgramStageResult.skipped(
            stage=self.stage_name,
            message="dominated_in_all_islands",
            error_type="ArchiveGateSkip",
        )
