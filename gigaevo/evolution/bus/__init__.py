"""Cross-run migration bus for GigaEvo.

Allows parallel evolution runs to share rejected-but-valid programs.
"""

from gigaevo.evolution.bus.engine import BusedEvolutionEngine
from gigaevo.evolution.bus.node import MigrationNode
from gigaevo.evolution.bus.topology import BusTopology, RingTopology, Topology
from gigaevo.evolution.bus.transport import (
    MigrantEnvelope,
    RedisStreamTransport,
    Transport,
)

__all__ = [
    "BusedEvolutionEngine",
    "BusTopology",
    "MigrantEnvelope",
    "MigrationNode",
    "RedisStreamTransport",
    "RingTopology",
    "Topology",
    "Transport",
]
