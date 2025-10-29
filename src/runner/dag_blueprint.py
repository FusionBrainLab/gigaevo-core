from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, Field

from src.database.state_manager import ProgramStateManager
from src.programs.dag.automata import DataFlowEdge, ExecutionOrderDependency
from src.programs.dag.dag import DAG
from src.programs.stages.base import Stage


class DAGBlueprint(BaseModel):
    """Blueprint used to build fresh `DAG` instances."""

    nodes: dict[str, Callable[[], Stage]] = Field(
        ..., description="Stage factories by name"
    )
    data_flow_edges: list[DataFlowEdge] = Field(
        ..., description="Data flow edges with semantic input names"
    )
    exec_order_deps: dict[str, list[ExecutionOrderDependency]] | None = Field(
        None, description="Execution order dependencies by stage name"
    )
    max_parallel_stages: int = Field(8, description="Maximum parallel stages allowed")
    dag_timeout: float = Field(2400.0, description="Timeout for DAG execution")

    def build(self, state_manager: ProgramStateManager) -> DAG:
        return DAG(
            nodes={name: factory() for name, factory in self.nodes.items()},
            data_flow_edges=self.data_flow_edges,
            state_manager=state_manager,
            execution_order_deps=self.exec_order_deps,
            max_parallel_stages=self.max_parallel_stages,
            dag_timeout=self.dag_timeout,
        )
