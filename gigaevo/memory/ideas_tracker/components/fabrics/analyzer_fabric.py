from __future__ import annotations

from gigaevo.memory.ideas_tracker.components.analyzer import IdeaAnalyzer
from gigaevo.memory.ideas_tracker.components.analyzer_f import IdeaAnalyzerFast


def create_analyzer(config: dict[str, str]) -> IdeaAnalyzer | IdeaAnalyzerFast:
    """Create an analyzer based on the config."""
    if config.get("type") == "fast":
        return IdeaAnalyzerFast(
            model=config.get("model", "deepseek/deepseek-v3.2"),
            base_url=config.get("base_url", "https://openrouter.ai/api/v1"),
            reasoning=config.get("reasoning", {}),
            **config.get("analyzer_fast_settings", {}),
        )
    else:
        return IdeaAnalyzer(
            model=config.get("model", "deepseek/deepseek-v3.2"),
            base_url=config.get("base_url", "https://openrouter.ai/api/v1"),
            reasoning=config.get("reasoning"),
        )
