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
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

from loguru import logger

from gigaevo.adversarial.opponent_provider import (
    OpponentArchiveProvider,
)
from gigaevo.programs.core_types import StageIO, VoidInput
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import DEFAULT_CACHE, NO_CACHE
from gigaevo.programs.stages.common import Box
from gigaevo.programs.stages.python_executors.wrapper import (
    ExecRunnerError,
    run_exec_runner,
)


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
    """Execute opponent programs given their IDs.

    Uses DEFAULT_CACHE (InputHashCache): reruns automatically when the
    opponent ID list changes.  Same IDs → cached results reused (no subprocess
    overhead).  Different IDs → reruns with fresh opponents.

    Fallback: when opponent_ids is empty (cold start), falls back to
    pre-loaded fallback_codes if provided.
    """

    InputsModel = OpponentIdsInput
    OutputModel = Box[Any]
    # DEFAULT_CACHE (InputHashCache) is the class default — reruns when opponent_ids hash changes

    def __init__(
        self,
        opponent_provider: OpponentArchiveProvider,
        n_opponents: int = 5,
        fallback_codes: list[str] | None = None,
        per_opponent_timeout: float = 10.0,
        python_path: list[Path] | None = None,
        max_memory_mb: int | None = None,
        archive_reeval: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._provider = opponent_provider
        self._n = n_opponents
        self._fallback_codes = fallback_codes or []
        self._per_timeout = per_opponent_timeout
        self._python_path = python_path or []
        self._max_memory_mb = max_memory_mb
        self._archive_reeval = archive_reeval

    def get_cache_handler(self):  # type: ignore[override]
        """InputHashCache when archive_reeval=True (re-eval only when IDs change).
        NO_CACHE when archive_reeval=False (always re-evaluate — adversarial-v2 baseline)."""
        return DEFAULT_CACHE if self._archive_reeval else NO_CACHE

    async def compute(self, program: Program) -> Box[Any]:
        params = cast(OpponentIdsInput, self.params)
        ids: list[str] = params.opponent_ids.data
        codes = await self._provider.get_codes_by_ids(ids)

        if not codes and self._fallback_codes:
            codes = self._fallback_codes
            logger.info(
                "[FetchOpponentResults] using {} fallback opponents (archive empty)",
                len(codes),
            )

        if not codes:
            logger.warning("[FetchOpponentResults] no opponents available")
            return Box[Any](data=[])

        tasks = [self._exec_one(code) for code in codes]
        raw = await asyncio.gather(*tasks, return_exceptions=True)

        results = [
            r
            for r in raw
            if not isinstance(r, (Exception, BaseException)) and r is not None
        ]
        logger.debug(
            "[FetchOpponentResults] {}/{} opponents succeeded",
            len(results),
            len(codes),
        )
        return Box[Any](data=results)

    async def _exec_one(self, code: str) -> Any:
        """Execute one opponent's entrypoint() in a subprocess."""
        try:
            value, _, _ = await run_exec_runner(
                code=code,
                function_name="entrypoint",
                python_path=self._python_path,
                timeout=int(self._per_timeout),
                max_memory_mb=self._max_memory_mb,
                max_output_size=64 * 1024 * 1024,
            )
            return value
        except (ExecRunnerError, TimeoutError, asyncio.CancelledError) as e:
            logger.debug("[FetchOpponentResults] opponent exec failed: {}", e)
            return None
