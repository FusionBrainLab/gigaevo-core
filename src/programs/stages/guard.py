from __future__ import annotations

"""Shared execution guard for Stage objects (lean)."""

import asyncio
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from loguru import logger

from src.exceptions import MetaEvolveError, SecurityViolationError
from src.programs.core_types import ProgramStageResult, StageState
from src.programs.utils import build_stage_result

if TYPE_CHECKING:  # pragma: no cover
    from src.programs.program import Program
    from src.programs.stages.base import Stage


async def stage_guard(stage: "Stage", program: "Program") -> ProgramStageResult:
    started_at = datetime.now(timezone.utc)
    t0 = time.monotonic()

    try:
        result = await asyncio.wait_for(
            stage._execute_stage(program, started_at),  # pylint: disable=protected-access
            timeout=stage.timeout,
        )
        logger.debug(
            "[{stage}] Program {pid}: ok in {dur:.2f}s",
            stage=stage.stage_name,
            pid=program.id,
            dur=(time.monotonic() - t0),
        )
        return result

    except asyncio.TimeoutError:
        err = f"Stage timeout after {stage.timeout:.2f}s"
        logger.error(
            "[{stage}] Program {pid}: {err} (ran {dur:.2f}s)",
            stage=stage.stage_name,
            pid=program.id,
            err=err,
            dur=(time.monotonic() - t0),
        )
        return build_stage_result(
            status=StageState.FAILED,
            started_at=started_at,
            error=err,
            stage_name=stage.stage_name,
            context="Timed out while running stage",
        )

    except SecurityViolationError as exc:
        logger.error(
            "[{stage}] Program {pid}: security violation: {exc}",
            stage=stage.stage_name,
            pid=program.id,
            exc=exc,
        )
        return build_stage_result(
            status=StageState.FAILED,
            started_at=started_at,
            error=exc,
            stage_name=stage.stage_name,
            context="Security violation detected",
        )

    except MetaEvolveError as exc:
        logger.error(
            "[{stage}] Program {pid}: {exc} (ran {dur:.2f}s)",
            stage=stage.stage_name,
            pid=program.id,
            exc=exc,
            dur=(time.monotonic() - t0),
        )
        return build_stage_result(
            status=StageState.FAILED,
            started_at=started_at,
            error=exc,
            stage_name=stage.stage_name,
        )

    except Exception as exc:  # pragma: no cover
        logger.exception(
            "[{stage}] Program {pid}: unexpected error after {dur:.2f}s",
            stage=stage.stage_name,
            pid=program.id,
            dur=(time.monotonic() - t0),
        )
        return build_stage_result(
            status=StageState.FAILED,
            started_at=started_at,
            error=exc,
            stage_name=stage.stage_name,
            context="Unexpected exception",
        )