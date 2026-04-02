from __future__ import annotations

from typing import Any

from gigaevo.memory.ideas_tracker.components.analyzer import IdeaAnalyzer
from gigaevo.memory.ideas_tracker.components.analyzer_f import IdeaAnalyzerFast


def create_analyzer(config: dict[str, Any]) -> IdeaAnalyzer | IdeaAnalyzerFast:
    """Create an analyzer based on the config."""
    if config.get("type") == "fast":
        fast_settings = config.get("analyzer_fast_settings", {})
        if not isinstance(fast_settings, dict):
            fast_settings = {}
        return IdeaAnalyzerFast(
            model=config.get("model", "deepseek/deepseek-v3.2"),
            base_url=config.get("base_url", "https://openrouter.ai/api/v1"),
            reasoning=config.get("reasoning", {}),
            **fast_settings,
        )
    else:
        reasoning = config.get("reasoning")
        return IdeaAnalyzer(
            model=config.get("model", "deepseek/deepseek-v3.2"),
            base_url=config.get("base_url", "https://openrouter.ai/api/v1"),
            reasoning=reasoning if isinstance(reasoning, dict) else None,
        )
