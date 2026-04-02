from __future__ import annotations

import asyncio
import json

import tqdm

from gigaevo.memory.ideas_tracker.components.analyzer import IdeaAnalyzer
from gigaevo.memory.ideas_tracker.components.analyzer_f import IdeaAnalyzerFast
from gigaevo.memory.ideas_tracker.components.data_components import RecordCardExtended


def enrich_ideas(
    ideas: list[RecordCardExtended],
    analyzer: IdeaAnalyzer | IdeaAnalyzerFast,
    task_description_summary: str,
) -> list[RecordCardExtended]:
    pbar = tqdm.tqdm(total=len(ideas), desc="Enriching ideas", leave=False)
    for idea in ideas:
        keywords: list[str] = []
        try:
            kw_response = analyzer.call_llm("keywords", idea.description)
            kw_parsed = json.loads(kw_response)
            keywords = kw_parsed.get("keywords", [])
        except Exception:
            pass

        summary = ""
        explanations = getattr(idea, "explanation", {}).get("explanations", [])
        valid_explanations = [e for e in explanations if isinstance(e, str)]
        if len(valid_explanations) == 1:
            summary = valid_explanations[0]
        elif len(valid_explanations) > 1:
            explanations_text = "\n".join(f"- {e}" for e in valid_explanations)
            try:
                sum_response = analyzer.call_llm("usage_summary", explanations_text)
                sum_parsed = json.loads(sum_response)
                summary = sum_parsed.get("summary", "")
            except Exception:
                pass

        idea.update_metadata(
            keywords=keywords,
            summary=summary,
            task_description_summary=task_description_summary,
        )

        pbar.update(1)
    pbar.close()
    return ideas


async def _enrich_idea_async_runner(
    idea: RecordCardExtended,
    analyzer: IdeaAnalyzer | IdeaAnalyzerFast,
    task_description_summary: str,
) -> RecordCardExtended:
    keywords: list[str] = []
    try:
        kw_response = await analyzer.call_llm_async("keywords", idea.description)
        kw_parsed = json.loads(kw_response)
        keywords = kw_parsed.get("keywords", [])
    except Exception:
        pass
    summary = ""
    explanations = getattr(idea, "explanation", {}).get("explanations", [])
    valid_explanations = [e for e in explanations if isinstance(e, str)]
    if len(valid_explanations) == 1:
        summary = valid_explanations[0]
    elif len(valid_explanations) > 1:
        explanations_text = "\n".join(f"- {e}" for e in valid_explanations)
        try:
            sum_response = await analyzer.call_llm_async(
                "usage_summary", explanations_text
            )
            sum_parsed = json.loads(sum_response)
            summary = sum_parsed.get("summary", "")
        except Exception:
            pass
    idea.update_metadata(
        keywords=keywords,
        summary=summary,
        task_description_summary=task_description_summary,
    )
    return idea


async def enrich_idea_async_(
    ideas: list[RecordCardExtended],
    analyzer: IdeaAnalyzer | IdeaAnalyzerFast,
    task_description_summary: str,
) -> list[RecordCardExtended]:
    tasks = [
        _enrich_idea_async_runner(idea, analyzer, task_description_summary)
        for idea in ideas
    ]
    results = await asyncio.gather(*tasks)
    return results


def enrich_ideas_async(
    ideas: list[RecordCardExtended],
    analyzer: IdeaAnalyzer | IdeaAnalyzerFast,
    task_description_summary: str,
) -> list[RecordCardExtended]:
    return asyncio.run(enrich_idea_async_(ideas, analyzer, task_description_summary))
