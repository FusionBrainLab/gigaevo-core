import json

from gigaevo.llm.ideas_tracker.components.analyzer import IdeaAnalyzer
from gigaevo.llm.ideas_tracker.components.analyzer_f import IdeaAnalyzerFast


def _summarize_task_description(
    analyzer: IdeaAnalyzer | IdeaAnalyzerFast,
    task_description: str,
    cache: dict[str, str] | None = None,
) -> str:
    task_text = str(task_description or "").strip()
    if not task_text:
        return ""
    if cache is not None and task_text in cache:
        return cache[task_text]

    summary = ""
    try:
        task_sum_response = analyzer.call_llm("task_description_summary", task_text)
        task_sum_parsed = json.loads(task_sum_response)
        summary = str(task_sum_parsed.get("summary", "")).strip()
    except Exception:
        summary = ""
    if not summary:
        summary = task_text[:240].strip()

    if cache is not None:
        cache[task_text] = summary
    return summary
