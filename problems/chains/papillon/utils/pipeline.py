"""Judge pipeline for PAPILLON quality and leakage evaluation.

Adapted from prompts/pupa/utils/pipeline.py — uses chains LLMClient
and local prompt templates. Only includes judge functions needed by
validate.py (no rewrite/generate functions, those are handled by chain steps).
"""

import json
import re

import pandas as pd

from problems.chains.papillon.utils.prompts import (
    QUALITY_JUDGE_PROMPT,
    LEAKAGE_JUDGE_PROMPT,
)


def parse_json_response(response: str) -> dict:
    """Extract JSON from LLM response."""
    json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    response_lower = response.lower()
    if "yes" in response_lower:
        return {"judgment": "yes", "num_pii_leaked": 0}
    elif "no" in response_lower:
        return {"judgment": "no", "num_pii_leaked": 0}
    return {}


async def judge_quality(
    judge_client,
    user_query: str,
    system_response: str,
    target_response: str,
) -> float:
    """Judge if system response is at least as good as target.

    Uses bidirectional comparison (like DSPy's process_scores).
    Returns 1.0 if quality passes, 0.0 otherwise.
    """
    prompt_forward = QUALITY_JUDGE_PROMPT.format(
        user_query=user_query,
        response_a=system_response,
        response_b=target_response,
    )
    response_forward = await judge_client(prompt_forward)
    parsed_forward = parse_json_response(response_forward)
    judgment_forward = int(
        parsed_forward.get("judgment", "").lower().startswith("yes")
    )

    prompt_backward = QUALITY_JUDGE_PROMPT.format(
        user_query=user_query,
        response_a=target_response,
        response_b=system_response,
    )
    response_backward = await judge_client(prompt_backward)
    parsed_backward = parse_json_response(response_backward)
    judgment_backward = int(
        parsed_backward.get("judgment", "").lower().startswith("yes")
    )

    if judgment_forward == judgment_backward:
        return 1.0
    return float(judgment_forward)


async def judge_leakage(
    judge_client,
    sanitized_query: str,
    pii_units: str,
) -> float:
    """Calculate fraction of PII units that appear in sanitized query.

    Returns leakage fraction (0 = no leakage, 1 = all PII leaked).
    """
    if not pii_units or pd.isna(pii_units):
        return 0.0

    pii_list = list(set(p.strip() for p in str(pii_units).split("||") if p.strip()))
    if not pii_list:
        return 0.0

    prompt = LEAKAGE_JUDGE_PROMPT.format(
        pii_list=pii_list,
        sanitized_query=sanitized_query,
    )
    response = await judge_client(prompt)
    parsed = parse_json_response(response)

    num_leaked = parsed.get("num_pii_leaked", 0)
    if isinstance(num_leaked, str):
        try:
            num_leaked = int(num_leaked)
        except ValueError:
            num_leaked = 0

    return min(num_leaked / len(pii_list), 1.0)
