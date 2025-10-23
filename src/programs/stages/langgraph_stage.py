"""Generic LangGraph stage wrapper.

This stage provides a reusable wrapper around ANY LangGraph agent.
Instead of creating separate stage classes for each agent, just configure this one!

Example:
    >>> from src.llm.agents.factories import create_insights_agent
    >>> 
    >>> agent = create_insights_agent(llm, task_description, metrics_context)
    >>> 
    >>> stage = LangGraphStage(
    ...     agent=agent,
    ...     prepare_inputs=lambda program: {"program": program},
    ... )
"""

import asyncio
import inspect
from datetime import datetime
from typing import Any, Callable, Optional

from loguru import logger

from src.exceptions import StageError
from src.llm.agents.base import LangGraphAgent
from src.programs.program import Program
from src.programs.core_types import ProgramStageResult, StageState
from src.programs.stages.base import Stage
from src.programs.utils import build_stage_result


class LangGraphStage(Stage):
    """Generic wrapper for any LangGraph agent.
    
    This stage eliminates duplication - it works with ANY agent!
    Just provide:
    1. The agent instance
    2. How to prepare inputs from program
    3. (Optional) Skip condition
    
    The stage handles:
    - Calling agent.arun() with prepared inputs
    - Returning results via stage output (flows through DAG)
    - Error handling
    - Logging
    
    Attributes:
        agent: LangGraph agent to execute
        prepare_inputs: Function that extracts agent inputs from program
        should_skip: Optional function to check if stage should be skipped
    """
    
    def __init__(
        self,
        agent: LangGraphAgent,
        prepare_inputs: Callable[[Program], dict[str, Any]],
        should_skip: Optional[Callable[[Program], tuple[bool, str]]] = None,
        **kwargs
    ):
        """Initialize LangGraph stage wrapper.
        
        Args:
            agent: Pre-configured LangGraph agent (from factory)
            prepare_inputs: Function (sync or async) to extract agent inputs from program.
                            Should return dict of kwargs for agent.arun(**kwargs).
                            Can be async for stages that need to fetch data (e.g., from storage).
            should_skip: Optional function returning (should_skip, reason)
            **kwargs: Passed to Stage base class
            
        Example:
            >>> # Insights agent
            >>> stage = LangGraphStage(
            ...     agent=insights_agent,
            ...     prepare_inputs=lambda p: {"program": p},
            ... )
            >>> 
            >>> # Lineage agent (with skip condition)
            >>> stage = LangGraphStage(
            ...     agent=lineage_agent,
            ...     prepare_inputs=lambda p: {"parent": p.parents[0], "child": p},
            ...     should_skip=lambda p: (not p.parents, "No parents"),
            ... )
        """
        super().__init__(**kwargs)
        self.agent = agent
        self.prepare_inputs = prepare_inputs
        self.should_skip = should_skip
        
        logger.info(
            f"[{self.stage_name}] Initialized LangGraphStage "
            f"(agent={agent.__class__.__name__})"
        )
    
    async def _execute_stage(
        self, program: Program, started_at: datetime
    ) -> ProgramStageResult:
        """Execute agent - completely generic!"""
        
        # Check if should skip
        if self.should_skip:
            should_skip, reason = self.should_skip(program)
            if should_skip:
                logger.debug(f"[{self.stage_name}] Skipping: {reason}")
                return build_stage_result(
                    status=StageState.COMPLETED,
                    started_at=started_at,
                    output=None,
                    stage_name=self.stage_name,
                )
        
        if inspect.iscoroutinefunction(self.prepare_inputs):
            agent_inputs = await self.prepare_inputs(program)
        else:
            agent_inputs = self.prepare_inputs(program)
        
        result = await self.agent.arun(**agent_inputs)
        
        if result is None or (isinstance(result, (list, dict)) and not result):
            raise StageError(f"Agent returned empty result")
        
        return build_stage_result(
            status=StageState.COMPLETED,
            started_at=started_at,
            output=result,
            stage_name=self.stage_name,
        )

