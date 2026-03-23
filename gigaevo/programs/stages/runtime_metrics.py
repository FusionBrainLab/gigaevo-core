from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import cast

from gigaevo.programs.core_types import StageIO
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage
from gigaevo.programs.stages.common import AnyContainer, Box, FloatDictContainer
from gigaevo.programs.stages.python_executors.wrapper import run_exec_runner
from gigaevo.programs.stages.stage_registry import StageRegistry


class RuntimeFitnessInputs(StageIO):
    candidate: Box[dict[str, float]]
    context: AnyContainer


@StageRegistry.register(
    description="Augment validator metrics with execution-time-based fitness."
)
class RuntimeFitnessStage(Stage):
    InputsModel = RuntimeFitnessInputs
    OutputModel = FloatDictContainer

    def __init__(
        self,
        *,
        source_stage_name: str = "CallProgramFunction",
        problem_dir: str | Path,
        timing_repetitions: int = 1,
        warmup_repetitions: int = 0,
        function_name: str = "entrypoint",
        min_runtime_sec: float = 1.0e-9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.source_stage_name = source_stage_name
        self.problem_dir = Path(problem_dir).resolve()
        self.timing_repetitions = max(1, int(timing_repetitions))
        self.warmup_repetitions = max(0, int(warmup_repetitions))
        self.function_name = function_name
        self.min_runtime_sec = float(min_runtime_sec)

    async def compute(self, program: Program) -> FloatDictContainer:
        metrics = dict(cast(RuntimeFitnessInputs, self.params).candidate.data)

        repeated_times: list[float] = []
        context = cast(RuntimeFitnessInputs, self.params).context
        args = [context.data]

        for _ in range(self.warmup_repetitions):
            await run_exec_runner(
                code=program.code,
                function_name=self.function_name,
                args=args,
                python_path=[self.problem_dir],
                timeout=max(1, int(self.timeout)),
            )

        for _ in range(self.timing_repetitions):
            started = time.perf_counter()
            await run_exec_runner(
                code=program.code,
                function_name=self.function_name,
                args=args,
                python_path=[self.problem_dir],
                timeout=max(1, int(self.timeout)),
            )
            repeated_times.append(time.perf_counter() - started)

        runtime_sec = max(statistics.median(repeated_times), self.min_runtime_sec)
        metrics["execution_time_sec"] = float(runtime_sec)
        metrics["fitness"] = float(1.0 / runtime_sec)
        metrics["timing_repetitions"] = float(self.timing_repetitions)
        metrics["warmup_repetitions"] = float(self.warmup_repetitions)

        return FloatDictContainer(data=metrics)
