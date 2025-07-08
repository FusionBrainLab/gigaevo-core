from __future__ import annotations

"""Lightweight immutable specification of a DAG.

`DAGSpec` records factory callables that produce fresh `Stage` instances 
every time a DAG is materialized. This avoids costly `deepcopy` operations 
and guarantees no shared mutable state leaks between program runs.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from src.programs.automata import ExecutionOrderDependency
from src.programs.stages.base import Stage


@dataclass(frozen=True, slots=True)
class DAGSpec:
    """An immutable blueprint used by `DagFactory` to build `DAG` instances."""

    nodes: Dict[str, Callable[[], Stage]]
    edges: Dict[str, List[str]]
    entry_points: Optional[List[str]] = None
    exec_order_deps: Optional[Dict[str, List[ExecutionOrderDependency]]] = None
    max_parallel_stages: int = 8

    def __post_init__(self):
        # Basic validation: all referenced stages in edges must exist.
        unknown = {dst for dsts in self.edges.values() for dst in dsts if dst not in self.nodes}
        if unknown:
            raise ValueError(f"Edges reference unknown stage(s): {', '.join(sorted(unknown))}") 