from typing import Any, Callable, Coroutine

from gigaevo.memory.ideas_tracker.components.analyzer import IdeaAnalyzer
from gigaevo.memory.ideas_tracker.components.analyzer_f import IdeaAnalyzerFast
from gigaevo.memory.ideas_tracker.components.data_components import RecordCardExtended
from gigaevo.memory.ideas_tracker.components.postprocessing import (
    enrich_ideas,
    enrich_ideas_async,
)


def create_postprocessing(
    config: dict[str, str],
) -> Callable[
    [list[RecordCardExtended], IdeaAnalyzer | IdeaAnalyzerFast, str],
    list[RecordCardExtended] | Coroutine[Any, Any, list[RecordCardExtended]],
]:
    if config.get("type") == "default":
        return enrich_ideas
    elif config.get("type") == "fast":
        return enrich_ideas_async
    else:
        raise ValueError(f"Invalid postprocessing type: {config.get('type')}")
