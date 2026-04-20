"""Adversarial co-evolution pipeline stages.

Two-stage DAG pattern for automatic archive re-evaluation:

  FetchOpponentIdsStage (NO_CACHE)
    Always re-samples opponent IDs from the live archive.
    Output: Box[Any](data=[id1, id2, ...])

  FetchOpponentResultsStage (InputHashCache / DEFAULT_CACHE)
    Receives opponent IDs as input; reruns only when IDs change.
    The DAG cache key is the ID list hash, so any archive update that
    changes the sampled opponent set automatically triggers re-evaluation
    without engine-level hooks.
    Output: Box[Any](data=[result1, result2, ...])

Pipeline wiring (AdversarialPipelineBuilder):
  FetchOpponentIdsStage → FetchOpponentResultsStage (opponent_ids)
  FetchOpponentResultsStage → CallValidatorFunction (context)

FetchOpponentResultsStage is a thin coordinator over an
`OpponentResultProvider` strategy (exec vs cached); see
`opponent_result_provider.py` for the two concrete paths.
"""

from __future__ import annotations

from typing import Any, cast

from loguru import logger

from gigaevo.adversarial.opponent_provider import OpponentArchiveProvider
from gigaevo.adversarial.opponent_result_provider import (
    ExecOpponentResultProvider,
    OpponentResultProvider,
)
from gigaevo.programs.core_types import StageIO, VoidInput
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import DEFAULT_CACHE, NO_CACHE
from gigaevo.programs.stages.common import Box


class FetchOpponentIdsStage(Stage):
    """Sample opponent IDs from the archive — always fresh (NO_CACHE).

    Runs every time a program's DAG is evaluated (including archive refreshes).
    Its output feeds FetchOpponentResultsStage as the cache key: when the
    sampled IDs change, FetchOpponentResultsStage automatically reruns.
    """

    InputsModel = VoidInput
    OutputModel = Box[Any]
    cache_handler = NO_CACHE

    def __init__(
        self,
        opponent_provider: OpponentArchiveProvider,
        n_opponents: int = 5,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._provider = opponent_provider
        self._n = n_opponents

    async def compute(self, program: Program) -> Box[Any]:
        # get_top_k() is deterministic top-K-by-fitness (Hall of Fame).
        # Replaces stochastic get_opponents() so cache invalidation is governed
        # by the archive ranking, not per-call sampling noise.
        opponents = await self._provider.get_top_k(self._n)
        ids = [o.program_id for o in opponents]
        logger.info(
            "[FetchOpponentIds] get_top_k({}) -> {} ids: {}",
            self._n,
            len(ids),
            ids[:3],
        )
        return Box[Any](data=ids)


class OpponentIdsInput(StageIO):
    """Input model for FetchOpponentResultsStage."""

    opponent_ids: Box[Any]


class FetchOpponentResultsStage(Stage):
    """Produce one evaluation payload per opponent id.

    Delegates the actual work to an ``OpponentResultProvider`` strategy
    (exec or cached). The stage itself is role-agnostic: it just wires
    opponent ids in, provider output out, and handles the cold-start
    fallback path when the archive is empty.

    Cache: DEFAULT_CACHE (InputHashCache) when ``archive_reeval=True`` —
    reruns only when the opponent ID list changes. NO_CACHE when
    ``archive_reeval=False`` (adversarial-v2 baseline: always re-evaluate).

    Fallback: when opponent_ids is empty or every slot comes back None
    (cold start + archive empty), falls back to ``fallback_codes`` via
    ExecOpponentResultProvider.produce_from_codes(). Fallback is always
    run in-subprocess — we have codes, not stored outputs.
    """

    InputsModel = OpponentIdsInput
    OutputModel = Box[Any]

    def __init__(
        self,
        *,
        result_provider: OpponentResultProvider,
        fallback_codes: list[str] | None = None,
        archive_reeval: bool = True,
        fallback_exec_provider: ExecOpponentResultProvider | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._result_provider = result_provider
        self._fallback_codes = list(fallback_codes or [])
        self._archive_reeval = archive_reeval
        # Fallback codes always execute (we only have source code, not
        # cached outputs). If the main provider is already an Exec provider
        # we reuse it; otherwise the caller must supply an Exec provider
        # for fallback. Only required when fallback_codes is non-empty.
        self._fallback_exec = fallback_exec_provider or (
            result_provider
            if isinstance(result_provider, ExecOpponentResultProvider)
            else None
        )
        if self._fallback_codes and self._fallback_exec is None:
            raise ValueError(
                "FetchOpponentResultsStage: fallback_codes provided but no "
                "ExecOpponentResultProvider available for fallback. Pass "
                "fallback_exec_provider= explicitly when result_provider is "
                "not an ExecOpponentResultProvider."
            )

    def get_cache_handler(self):  # type: ignore[override]
        """InputHashCache when archive_reeval=True (re-eval only when IDs change).
        NO_CACHE when archive_reeval=False (always re-evaluate)."""
        return DEFAULT_CACHE if self._archive_reeval else NO_CACHE

    async def compute(self, program: Program) -> Box[Any]:
        params = cast(OpponentIdsInput, self.params)
        ids: list[str] = list(params.opponent_ids.data)

        results = await self._result_provider.produce(ids)

        all_missing = not ids or all(r is None for r in results)
        if all_missing and self._fallback_codes:
            logger.info(
                "[FetchOpponentResults] using {} fallback opponents (archive cold)",
                len(self._fallback_codes),
            )
            assert self._fallback_exec is not None  # guarded in __init__
            results = await self._fallback_exec.produce_from_codes(self._fallback_codes)

        logger.debug(
            "[FetchOpponentResults] produced {}/{} slots populated",
            sum(1 for r in results if r is not None),
            len(results),
        )
        return Box[Any](data=results)
