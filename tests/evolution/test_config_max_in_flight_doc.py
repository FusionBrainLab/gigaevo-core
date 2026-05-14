"""Doc-string guard: max_in_flight must describe the two-pool semantics.

Operators read this description via Hydra --help and the generated YAML,
so the wording must call out both producer and buffer pools — peak depth
is ~2N, not N.
"""

from __future__ import annotations

from gigaevo.evolution.engine.config import EngineConfig


def test_max_in_flight_description_mentions_two_pools() -> None:
    field = EngineConfig.model_fields["max_in_flight"]
    desc = field.description or ""
    # Must call out BOTH pools so operators know depth is ~2N, not N.
    assert "producer" in desc.lower()
    assert "buffer" in desc.lower()
    assert "2" in desc or "two" in desc.lower()
