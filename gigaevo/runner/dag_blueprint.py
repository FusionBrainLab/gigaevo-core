from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from gigaevo.database.state_manager import ProgramStateManager
from gigaevo.programs.dag.automata import DataFlowEdge, ExecutionOrderDependency
from gigaevo.programs.dag.dag import DAG
from gigaevo.programs.stages.base import Stage


class DAGBlueprint(BaseModel):
    """Blueprint used to build fresh `DAG` instances."""

    nodes: Mapping[str, Callable[[], Stage]] = Field(
        ..., description="Stage factories by name"
    )
    data_flow_edges: Sequence[DataFlowEdge] = Field(
        ..., description="Data flow edges with semantic input names"
    )
    exec_order_deps: Mapping[str, Sequence[ExecutionOrderDependency]] | None = Field(
        None, description="Execution order dependencies by stage name"
    )
    max_parallel_stages: int = Field(8, description="Maximum parallel stages allowed")
    dag_timeout: float = Field(2400.0, description="Timeout for DAG execution")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def build(self, state_manager: ProgramStateManager) -> DAG:
        return DAG(
            nodes={name: factory() for name, factory in self.nodes.items()},
            data_flow_edges=self.data_flow_edges,
            state_manager=state_manager,
            execution_order_deps=self.exec_order_deps,
            max_parallel_stages=self.max_parallel_stages,
            dag_timeout=self.dag_timeout,
        )
