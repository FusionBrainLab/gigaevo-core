"""Adversarial co-evolution pipeline stages.

FetchOpponentResultsStage: reads opponent codes from the archive,
executes each opponent's entrypoint() in parallel subprocesses
(via run_exec_runner), and returns the list of results as context
for CallValidatorFunction.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger

from gigaevo.adversarial.opponent_provider import OpponentArchiveProvider
from gigaevo.programs.core_types import VoidInput
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.cache_handler import NO_CACHE
from gigaevo.programs.stages.common import Box
from gigaevo.programs.stages.python_executors.wrapper import (
    ExecRunnerError,
    run_exec_runner,
)


class FetchOpponentResultsStage(Stage):
    """Fetch and execute opponent programs from the MAP-Elites archive.

    1. Read opponent codes from OpponentArchiveProvider (async Redis)
    2. If empty (cold start), use fallback_codes
    3. Execute each opponent's entrypoint() in parallel subprocesses
       (uses run_exec_runner -- same as CallProgramFunction)
    4. Return list of results as Box[Any] (context for CallValidatorFunction)

    Each opponent has its own subprocess with timeout -- one rogue opponent
    does not block others.
    """

    InputsModel = VoidInput
    OutputModel = Box[Any]
    cache_handler = NO_CACHE  # opponents change between generations

    def __init__(
        self,
        opponent_provider: OpponentArchiveProvider,
        n_opponents: int = 5,
        fallback_codes: list[str] | None = None,
        per_opponent_timeout: float = 10.0,
        python_path: list[Path] | None = None,
        max_memory_mb: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._provider = opponent_provider
        self._n = n_opponents
        self._fallback_codes = fallback_codes or []
        self._per_timeout = per_opponent_timeout
        self._python_path = python_path or []
        self._max_memory_mb = max_memory_mb

    async def compute(self, program: Program) -> Box[Any]:
        opponents = await self._provider.get_opponents(n=self._n)
        codes = [o.code for o in opponents]
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
