"""Regression tests for DAG construction bugs.

DAG(nodes={}) crash (fixed):
    DAG.__init__ called max() on an empty generator when nodes={}, producing
    "ValueError: max() arg is an empty sequence" with no useful context.
    Fix: explicit guard raises ValueError("DAG requires at least one stage...").
"""

from __future__ import annotations

import pytest

from gigaevo.programs.dag.dag import DAG
from tests.conftest import NullWriter


async def test_dag_constructor_raises_clear_error_on_empty_nodes(state_manager) -> None:
    """DAG(nodes={}) raises ValueError with a descriptive message.

    Previously: max() on empty generator → "max() arg is an empty sequence".
    Fixed: explicit guard → ValueError("DAG requires at least one stage...").
    """
    with pytest.raises(ValueError, match="at least one stage"):
        DAG(
            nodes={},
            data_flow_edges=[],
            execution_order_deps=None,
            state_manager=state_manager,
            writer=NullWriter(),
        )
