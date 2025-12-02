from __future__ import annotations

from typing import Any, Dict

from loguru import logger

from gigaevo.llm.agents.base import LangGraphAgent
from gigaevo.programs.core_types import (
    ProgramStageResult,
    StageIO,
    VoidInput,
    VoidOutput,
)
from gigaevo.programs.program import Program
from gigaevo.programs.stages.base import Stage


class LangGraphStage(Stage):
    """
    Generic wrapper for LangGraph/LangChain-like agents with lifecycle hooks.

    Subclasses MUST define:
      - InputsModel (StageIO): strict schema for agent inputs (Optionals mark optional DAG inputs)
      - OutputModel (StageIO): strict output schema

    Execution flow:
      1) Validate DAG inputs -> self.params (InputsModel)
      2) kwargs0 = preprocess(program, self.params)
           - May return Dict[str, Any] (kwargs to pass to agent)
           - Or return ProgramStageResult to short-circuit (e.g., SKIPPED/FAILED)
      3) Inject program under `program_kwarg` (if set) + merge `extra_kwargs`
      4) result = agent(...) via ainvoke/arun/invoke/run/callable
      5) out = postprocess(program, result)
           - May return OutputModel or ProgramStageResult
           - Defaults coerce result to OutputModel (single-field wrap or dict->validate)
    """

    InputsModel = VoidInput
    OutputModel = VoidOutput

    def __init__(
        self,
        *,
        agent: LangGraphAgent,
        program_kwarg: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.agent = agent
        self.program_kwarg = program_kwarg
        logger.info(
            "[{}] Initialized with agent={} program_kwarg={}",
            self.stage_name,
            getattr(agent, "__class__", type(agent)).__name__,
            self.program_kwarg,
        )

    async def preprocess(
        self, program: Program, params: StageIO
    ) -> Dict[str, Any] | ProgramStageResult:
        """
        Build kwargs for the agent call from validated params.
        Default: pass through all fields from InputsModel.
        """
        fields = self.__class__.InputsModel.model_fields  # type: ignore[attr-defined]
        kwargs: Dict[str, Any] = {}
        for name in fields.keys():
            v = getattr(params, name)
            kwargs[name] = v
        return kwargs

    async def postprocess(
        self, program: Program, agent_result: Any
    ) -> StageIO | ProgramStageResult:
        """
        Coerce/validate agent_result to OutputModel (or return a ProgramStageResult).
        Default behavior:
          - if already OutputModel -> return
          - if OutputModel has a single field and the value matches field type -> wrap
          - if dict-like -> model_validate into OutputModel
          - else -> TypeError (handled by base Stage exception policy)
        """
        # Already correct type
        if isinstance(agent_result, self.__class__.OutputModel):
            return agent_result

        out_fields = self.__class__.OutputModel.model_fields  # type: ignore[attr-defined]

        # Try single-field wrapper (let Pydantic validate)
        if len(out_fields) == 1:
            ((field_name, _),) = out_fields.items()
            try:
                return self.__class__.OutputModel(**{field_name: agent_result})
            except Exception:
                # Pydantic validation failed, continue to try other coercion methods
                pass

        # Dict-like -> validate
        if isinstance(agent_result, dict):
            return self.__class__.OutputModel.model_validate(agent_result)

        raise TypeError(
            f"{self.stage_name}: agent returned {type(agent_result).__name__}; "
            f"cannot coerce to {self.__class__.OutputModel.__name__}"
        )

    async def _agent_call(self, kwargs: Dict[str, Any]) -> Any:
        return await self.agent.arun(**kwargs)

    async def compute(self, program: Program) -> StageIO | ProgramStageResult:
        # 1) Preprocess
        prep = await self.preprocess(program, self.params)
        if isinstance(prep, ProgramStageResult):
            return prep
        kwargs = dict(prep)

        # 2) Inject current program if requested
        if self.program_kwarg is not None:
            if self.program_kwarg in kwargs:
                raise ValueError(
                    f"{self.stage_name}: program_kwarg '{self.program_kwarg}' collides with a preprocessed arg."
                )
            kwargs[self.program_kwarg] = program

        # 3) Call agent
        result = await self._agent_call(kwargs)

        # 5) Postprocess
        return await self.postprocess(program, result)
