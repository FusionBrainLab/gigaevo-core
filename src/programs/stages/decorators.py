from __future__ import annotations

"""Reusable Stage class decorators for retry and concurrency control.

These decorators wrap the *_execute_stage* coroutine rather than *run()* so they
compose cleanly with the shared *stage_guard* logic.
"""

import asyncio
import functools
from typing import TypeVar, Callable, Type

from loguru import logger

from .base import Stage
from src.programs.utils import build_stage_result
from src.programs.stages.state import StageState

T = TypeVar("T", bound=Type[Stage])

__all__ = ["retry", "semaphore"]


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def retry(times: int = 3, backoff: float = 0.2, exceptions: tuple[type[Exception], ...] = (Exception,)) -> Callable[[T], T]:
    """Retry a Stage's *_execute_stage* up to *times* on failure.

    Only retries when the resulting :pyclass:`ProgramStageResult` is FAILED or
    when *_execute_stage* raises one of *exceptions*.
    """

    def decorator(cls: T) -> T:  # type: ignore[misc]
        if not issubclass(cls, Stage):
            raise TypeError("@retry can only decorate Stage subclasses")

        orig_execute = cls._execute_stage  # type: ignore[attr-defined]

        @functools.wraps(orig_execute)
        async def _execute_with_retry(self: Stage, program, started_at):  # type: ignore[override]
            for attempt in range(1, times + 1):
                try:
                    result = await orig_execute(self, program, started_at)
                    if result.is_completed() or attempt == times:
                        return result
                    logger.debug(
                        f"[{self.stage_name}] Retry {attempt}/{times} – result failed"
                    )
                except exceptions as exc:
                    if attempt == times:
                        raise
                    logger.debug(
                        f"[{self.stage_name}] Retry {attempt}/{times} after exception: {exc}"
                    )
                await asyncio.sleep(backoff * (2 ** (attempt - 1)))

            # Should never reach here
            return result  # type: ignore[_unbound]

        cls._execute_stage = _execute_with_retry  # type: ignore[assignment]
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Semaphore decorator
# ---------------------------------------------------------------------------

def semaphore(limit: int) -> Callable[[T], T]:
    """Limit concurrent executions of a Stage via an *asyncio.Semaphore*.
    """

    _sem = asyncio.Semaphore(max(1, limit))

    def decorator(cls: T) -> T:  # type: ignore[misc]
        if not issubclass(cls, Stage):
            raise TypeError("@semaphore can only decorate Stage subclasses")

        orig_execute = cls._execute_stage  # type: ignore[attr-defined]

        @functools.wraps(orig_execute)
        async def _execute_with_sema(self: Stage, program, started_at):  # type: ignore[override]

            semaphore_timeout = self.timeout * 0.9
            
            try:
                await asyncio.wait_for(_sem.acquire(), timeout=semaphore_timeout)
            except asyncio.TimeoutError:
                logger.error(
                    f"[{self.stage_name}] Program {program.id}: "
                    f"Semaphore acquisition timed out after {semaphore_timeout:.1f}s"
                )
                # Return timeout error instead of hanging forever
                return build_stage_result(
                    status=StageState.FAILED,
                    started_at=started_at,
                    error=f"Semaphore acquisition timeout after {semaphore_timeout:.1f}s",
                    stage_name=self.stage_name,
                    context=f"All {limit} semaphore slots may be held by stuck processes",
                )
            
            try:
                return await orig_execute(self, program, started_at)
            finally:
                _sem.release()

        cls._execute_stage = _execute_with_sema  # type: ignore[assignment]
        return cls

    return decorator 