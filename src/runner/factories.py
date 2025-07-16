"""Factory helpers for runner components.

This module provides `DagFactory`, a utility that produces fresh DAG
instances from immutable templates. It guarantees that per-program DAGs do
not share mutable `Stage` objects which could leak state across runs.
"""
from __future__ import annotations


from src.programs.dag import DAG
from src.programs.state_manager import ProgramStateManager
from .dag_spec import DAGSpec

__all__ = ["DagFactory"]


class DagFactory:
    """Create new `DAG` objects from `DAGSpec` templates.

    Parameters
    ----------
    dag_spec
        A `DAGSpec` object containing stage factories and configuration.
    max_parallel_stages
        Bound parallelism inside each DAG (copied to the constructed DAG).
    """

    def __init__(
        self,
        dag_spec: DAGSpec,
    ) -> None:
        self._spec = dag_spec

    def create(self, state_manager: ProgramStateManager) -> DAG:
        """Return a brand-new `DAG` instance whose `Stage`s are independent."""
        new_nodes = {name: factory() for name, factory in self._spec.nodes.items()}

        return DAG(
            nodes=new_nodes,
            edges=self._spec.edges,
            state_manager=state_manager,
            entry_points=self._spec.entry_points or list(new_nodes.keys()),
            execution_order_deps=self._spec.exec_order_deps,
            max_parallel_stages=self._spec.max_parallel_stages,
            dag_timeout=self._spec.dag_timeout,
        ) 