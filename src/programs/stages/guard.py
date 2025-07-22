from __future__ import annotations

"""Central execution guard shared by all Stage objects.

Encapsulates timeout, resource monitoring, error mapping,
metrics recording and Prometheus export in a single place so every Stage
inherits identical robust behaviour.
"""

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from loguru import logger

from src.exceptions import MetaEvolveError, SecurityViolationError
from src.programs.stages.prometheus import StagePrometheusExporter
from src.programs.stages.state import ProgramStageResult, StageState
from src.programs.utils import build_stage_result

if TYPE_CHECKING:  # pragma: no cover
    from src.programs.program import Program
    from src.programs.stages.base import Stage


async def _run_stage_with_monitoring(
    stage: "Stage", program: "Program", started_at: datetime
) -> ProgramStageResult:
    """Execute stage with resource monitoring - extracted to allow timeout to cover everything."""
    async with stage._resource_monitor(
        program
    ):  # pylint: disable=protected-access
        return await stage._execute_stage(
            program, started_at
        )  # pylint: disable=protected-access


async def stage_guard(
    stage: "Stage", program: "Program"
) -> ProgramStageResult:  # noqa: D401
    """Run *stage* on *program* with uniform guarantees."""

    started_at = datetime.now(timezone.utc)
    duration: float = 0.0

    try:
        result = await asyncio.wait_for(
            _run_stage_with_monitoring(stage, program, started_at),
            timeout=stage.timeout,
        )

        duration = (datetime.now(timezone.utc) - started_at).total_seconds()
        stage.metrics.record_execution(duration, True)
        StagePrometheusExporter.record(stage.stage_name, duration, True)
        logger.debug(
            f"[{stage.stage_name}] Program {program.id}: Completed in {duration:.2f}s"
        )
        return result

    except asyncio.TimeoutError:
        duration = (datetime.now(timezone.utc) - started_at).total_seconds()
        stage.metrics.record_execution(duration, False)
        StagePrometheusExporter.record(stage.stage_name, duration, False)
        err = f"Stage timeout after {stage.timeout}s (includes semaphore wait)"
        logger.error(f"[{stage.stage_name}] Program {program.id}: {err}")
        return build_stage_result(
            status=StageState.FAILED,
            started_at=started_at,
            error=err,
            stage_name=stage.stage_name,
            context=f"Timeout after {stage.timeout} seconds - may have been waiting for semaphore",
        )

    except SecurityViolationError as exc:
        stage.metrics.record_execution(0, False)
        StagePrometheusExporter.record(stage.stage_name, 0, False)
        logger.error(
            f"[{stage.stage_name}] Program {program.id}: Security violation - {exc}"
        )
        return build_stage_result(
            status=StageState.FAILED,
            started_at=started_at,
            error=exc,
            stage_name=stage.stage_name,
            context="Security violation detected",
        )

    except (
        MetaEvolveError
    ) as exc:  # includes ValidationError, ResourceError, etc.
        duration = (datetime.now(timezone.utc) - started_at).total_seconds()
        stage.metrics.record_execution(duration, False)
        StagePrometheusExporter.record(stage.stage_name, duration, False)
        logger.error(f"[{stage.stage_name}] Program {program.id}: {exc}")
        return build_stage_result(
            status=StageState.FAILED,
            started_at=started_at,
            error=exc,
            stage_name=stage.stage_name,
        )

    except Exception as exc:  # pragma: no cover – unexpected
        duration = (datetime.now(timezone.utc) - started_at).total_seconds()
        stage.metrics.record_execution(duration, False)
        StagePrometheusExporter.record(stage.stage_name, duration, False)
        logger.error(
            f"[{stage.stage_name}] Program {program.id}: Unexpected error - {exc}"
        )
        return build_stage_result(
            status=StageState.FAILED,
            started_at=started_at,
            error=exc,
            stage_name=stage.stage_name,
            context="Unexpected exception",
        )
