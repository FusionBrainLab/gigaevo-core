"""Pipeline functions for PUPA evaluation."""

import asyncio
import json
import re
from statistics import mean
from typing import List

import pandas as pd

from problems.prompts.client import LLMClient
from problems.prompts.pupa.config import LLM_CONFIG, JUDGE_CONFIG
from problems.prompts.pupa.utils.prompts import QUALITY_JUDGE_PROMPT, LEAKAGE_JUDGE_PROMPT


def parse_json_response(response: str) -> dict:
    """Extract JSON from LLM response."""
    json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: check for yes/no in response
    response_lower = response.lower()
    if "yes" in response_lower:
        return {"judgment": "yes", "num_pii_leaked": 0}
    elif "no" in response_lower:
        return {"judgment": "no", "num_pii_leaked": 0}
    return {}


async def rewrite_query(
    client: LLMClient,
    rewriter_prompt_template: str,
    user_query: str,
) -> str:
    """Rewrite user query using evolved prompt template."""
    prompt = rewriter_prompt_template.format(user_query=user_query)
    sanitized_query = await client(prompt)
    return sanitized_query.strip()


async def generate_response(
    client: LLMClient,
    sanitized_query: str,
) -> str:
    """Generate response from sanitized query."""
    response = await client(sanitized_query)
    return response.strip()


async def judge_quality(
    judge_client: LLMClient,
    user_query: str,
    system_response: str,
    target_response: str,
) -> float:
    """Judge if system response is at least as good as target.

    Uses bidirectional comparison like DSPy's process_scores.
    Returns 1 if quality passes, 0 otherwise.
    """
    # Forward comparison: is system response >= target?
    prompt_forward = QUALITY_JUDGE_PROMPT.format(
        user_query=user_query,
        response_a=system_response,
        response_b=target_response,
    )
    response_forward = await judge_client(prompt_forward)
    parsed_forward = parse_json_response(response_forward)
    judgment_forward = int(parsed_forward.get("judgment", "").lower().startswith("yes"))

    # Backward comparison: is target >= system response?
    prompt_backward = QUALITY_JUDGE_PROMPT.format(
        user_query=user_query,
        response_a=target_response,
        response_b=system_response,
    )
    response_backward = await judge_client(prompt_backward)
    parsed_backward = parse_json_response(response_backward)
    judgment_backward = int(parsed_backward.get("judgment", "").lower().startswith("yes"))

    if judgment_forward == judgment_backward:
        return 1.0
    return float(judgment_forward)


async def judge_leakage(
    judge_client: LLMClient,
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

    # Single call with full PII list
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


async def process_sample(
    llm_client: LLMClient,
    judge_client: LLMClient,
    rewriter_prompt_template: str,
    sample: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Process single sample through the full pipeline."""
    async with semaphore:
        user_query = sample["user_query"]
        target_response = sample["target_response"]
        pii_units = sample.get("pii_units", "")

        # Step 1: Rewrite query
        sanitized_query = await rewrite_query(
            llm_client, rewriter_prompt_template, user_query
        )

        # Step 2: Generate response from sanitized query
        system_response = await generate_response(llm_client, sanitized_query)

        # Step 3: Judge quality and leakage
        quality = await judge_quality(
            judge_client, user_query, system_response, target_response
        )
        leakage = await judge_leakage(judge_client, sanitized_query, pii_units)

        return {"quality": quality, "leakage": leakage}


async def run_pipeline_async(
    rewriter_prompt_template: str,
    context: dict,
    dataset_key: str = "train_dataset",
    max_concurrent: int = 32,
) -> List[dict]:
    """Run the full pipeline on dataset."""
    dataset = context[dataset_key]

    if hasattr(dataset, "to_dict"):
        samples = dataset.to_dict("records")
    else:
        samples = list(dataset)

    llm_client = LLMClient(**LLM_CONFIG)
    judge_client = LLMClient(**JUDGE_CONFIG)

    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [
        process_sample(
            llm_client.copy(),
            judge_client.copy(),
            rewriter_prompt_template,
            sample,
            semaphore,
        )
        for sample in samples
    ]

    results = await asyncio.gather(*tasks)

    await llm_client.close()
    await judge_client.close()

    return results


def run_pipeline(
    rewriter_prompt_template: str,
    context: dict,
    dataset_key: str = "train_dataset",
) -> List[dict]:
    """Run the full pipeline synchronously."""
    return asyncio.run(
        run_pipeline_async(rewriter_prompt_template, context, dataset_key)
    )
