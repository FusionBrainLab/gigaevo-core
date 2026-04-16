from __future__ import annotations

from gigaevo.evolution.engine.acceptor import StandardEvolutionAcceptor
from gigaevo.evolution.engine.config import EngineConfig, SteadyStateEngineConfig
from gigaevo.evolution.engine.core import EvolutionEngine
from gigaevo.evolution.engine.metrics import EngineMetrics
from gigaevo.evolution.engine.steady_state import SteadyStateEvolutionEngine
from gigaevo.evolution.engine.stopper import (
    CompositeStopper,
    EvolutionStopper,
    FitnessPlateauStopper,
    MaxGenerationsStopper,
    StopContext,
    StopDecision,
    WallClockStopper,
)
