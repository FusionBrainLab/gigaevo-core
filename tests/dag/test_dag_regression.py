"""DAG construction guard tests."""

from __future__ import annotations

import pytest

from gigaevo.programs.dag.dag import DAG
from tests.conftest import NullWriter


async def test_dag_constructor_raises_clear_error_on_empty_nodes(state_manager) -> None:
    """DAG(nodes={}) raises ValueError naming the empty-nodes invariant."""
    with pytest.raises(ValueError, match="at least one stage"):
        DAG(
            nodes={},
            data_flow_edges=[],
            execution_order_deps=None,
            state_manager=state_manager,
            writer=NullWriter(),
        )
