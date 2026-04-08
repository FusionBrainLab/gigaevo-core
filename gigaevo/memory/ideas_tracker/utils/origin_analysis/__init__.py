"""Origin analysis subpackage. Public API: analyse(), AnalysisResult."""

from gigaevo.memory.ideas_tracker.utils.origin_analysis.pipeline import analyse
from gigaevo.memory.ideas_tracker.utils.origin_analysis.types import AnalysisResult

__all__ = ["analyse", "AnalysisResult"]
