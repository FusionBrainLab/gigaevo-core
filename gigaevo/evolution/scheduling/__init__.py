"""Scheduling abstractions for controlling DAG evaluation order.

Provides pluggable feature extractors, eval-time predictors, and
program prioritizers.  The default (FIFO) preserves existing behavior;
LPT (Longest Processing Time first) scheduling reduces tail idle time
by starting predicted-longest jobs first.

Quick start::

    from gigaevo.evolution.scheduling import (
        LPTPrioritizer,
        SimpleHeuristicPredictor,
        CodeFeatureExtractor,
    )

    predictor = SimpleHeuristicPredictor()
    prioritizer = LPTPrioritizer(predictor)

    # In DagRunner: sorted_programs = prioritizer.prioritize(candidates)
    # After eval:   prioritizer.predictor.update(program, actual_duration)
"""

from gigaevo.evolution.scheduling.feature_extractor import (
    ChainFeatureExtractor,
    CodeFeatureExtractor,
    CompositeFeatureExtractor,
    FeatureExtractor,
)
from gigaevo.evolution.scheduling.predictor import (
    ConstantPredictor,
    EvalTimePredictor,
    RidgePredictor,
    SimpleHeuristicPredictor,
)
from gigaevo.evolution.scheduling.prioritizer import (
    FIFOPrioritizer,
    LPTPrioritizer,
    ProgramPrioritizer,
    SJFPrioritizer,
)

__all__ = [
    "ChainFeatureExtractor",
    "CodeFeatureExtractor",
    "CompositeFeatureExtractor",
    "ConstantPredictor",
    "EvalTimePredictor",
    "FeatureExtractor",
    "FIFOPrioritizer",
    "LPTPrioritizer",
    "ProgramPrioritizer",
    "RidgePredictor",
    "SJFPrioritizer",
    "SimpleHeuristicPredictor",
]
