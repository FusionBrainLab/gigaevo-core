"""Utilities for prompt evolution problems."""

from __future__ import annotations

import asyncio
from string import Formatter
from typing import Tuple

from gigaevo.database.redis_program_storage import (
    RedisProgramStorage,
    RedisProgramStorageConfig,
)
from problems.prompts.client import LLMClient
from problems.prompts.types import CallLog, OutputDict
from tools.utils import RedisRunConfig

__all__ = [
    "get_best_program",
    "validate_prompt_template",
    "run_prompts",
]


def get_best_program(
    config: RedisRunConfig,
    fitness_col: str = "metric_fitness",
    minimize: bool = False,
) -> dict | None:
    """Extract the best program from Redis by fitness.

    Args:
        config: Redis connection config
        fitness_col: Name of the fitness metric column
        minimize: If True, lower fitness is better

    Returns:
        Dict with program info: {id, code, fitness, metrics, metadata} or None if no programs
    """

    async def _fetch():
        storage = RedisProgramStorage(
            RedisProgramStorageConfig(
                redis_url=config.url(),
                key_prefix=config.redis_prefix,
                max_connections=50,
                connection_pool_timeout=30.0,
                health_check_interval=60,
                read_only=True,
            )
        )
        try:
            return await storage.get_all()
        finally:
            await storage.close()

    programs = asyncio.run(_fetch())

    if not programs:
        return None

    # Filter to programs with valid fitness
    metric_name = fitness_col.replace("metric_", "")
    valid_programs = [
        p
        for p in programs
        if metric_name in p.metrics and p.metrics[metric_name] is not None
    ]

    if not valid_programs:
        return None

    # Find best by fitness
    if minimize:
        best = min(valid_programs, key=lambda p: p.metrics[metric_name])
    else:
        best = max(valid_programs, key=lambda p: p.metrics[metric_name])

    return {
        "id": best.id,
        "code": best.code,
        "fitness": best.metrics[metric_name],
        "metrics": best.metrics,
        "metadata": best.metadata,
    }


def validate_prompt_template(
    prompt_template: str,
    required_placeholders: list[str],
    available_placeholders: list[str],
) -> None:
    """Validate prompt template has required placeholders and no invalid ones.

    Args:
        prompt_template: The prompt template string with {field} placeholders
        required_placeholders: Fields that MUST be in template
        available_placeholders: All fields that CAN be used in template

    Raises:
        ValueError: If required fields are missing or invalid fields are used
    """
    formatter = Formatter()
    found_placeholders: set[str] = set()

    for _, field_name, _, _ in formatter.parse(prompt_template):
        if field_name is not None:  # None represents literal text sections
            # Handle nested field access like {foo.bar}
            base_field = field_name.split(".")[0].split("[")[0]
            if base_field:
                found_placeholders.add(base_field)

    # Check if all required placeholders are present
    missing_required = set(required_placeholders) - found_placeholders
    if missing_required:
        raise ValueError(
            f"Missing required placeholder(s): {', '.join(sorted(missing_required))}. "
            f"Found: {', '.join(sorted(found_placeholders)) if found_placeholders else 'none'}"
        )

    # Check if all found placeholders are in available list
    invalid_placeholders = found_placeholders - set(available_placeholders)
    if invalid_placeholders:
        raise ValueError(
            f"Invalid placeholder(s): {', '.join(sorted(invalid_placeholders))}. "
            f"Available: {', '.join(sorted(available_placeholders))}"
        )


async def _process_sample(
    client: LLMClient,
    prompt_template: str,
    index: int,
    sample: dict,
    semaphore: asyncio.Semaphore,
) -> Tuple[int, str, CallLog]:
    """Process single sample with cost tracking.

    Args:
        client: LLM client (should be a copy for isolation)
        prompt_template: Template to format with sample
        index: Sample index for ordering
        sample: Sample data dict
        semaphore: Concurrency limiter

    Returns:
        Tuple of (index, raw_response, call_log)
    """
    async with semaphore:
        # Clear logs before processing this sample
        client.clear_logs()

        # Format prompt with sample
        prompt = prompt_template.format(**sample)

        # Call LLM (single call per sample for prompts)
        raw_response = await client(prompt)

        # Extract the single call log
        call_log = client.call_logs[0]

        return index, raw_response, call_log


async def _run_prompts_async(
    prompt_template: str,
    client: LLMClient,
    context: dict,
    dataset_key: str = "train_dataset",
    max_concurrent: int = 32,
) -> OutputDict:
    """Run prompt template on dataset asynchronously.

    Args:
        prompt_template: Template string with {field} placeholders
        client: LLM client
        context: Context dict with dataset
        dataset_key: Key for dataset in context
        max_concurrent: Maximum concurrent requests

    Returns:
        OutputDict with raw responses and call_logs
    """
    dataset = context[dataset_key]

    # Convert DataFrame to list of dicts
    if hasattr(dataset, "to_dict"):
        samples = dataset.to_dict("records")
    else:
        samples = list(dataset)

    semaphore = asyncio.Semaphore(max_concurrent)

    # Process all samples concurrently with isolated client copies
    tasks = [
        _process_sample(
            client.copy(),
            prompt_template,
            i,
            sample,
            semaphore,
        )
        for i, sample in enumerate(samples)
    ]

    results = await asyncio.gather(*tasks)

    # Sort by index and extract responses and logs
    results = sorted(results, key=lambda x: x[0])
    responses = [r[1] for r in results]
    call_logs = [r[2] for r in results]

    return OutputDict(predictions=responses, call_logs=call_logs)


def run_prompts(
    prompt_template: str,
    client: LLMClient,
    context: dict,
    dataset_key: str = "train_dataset",
) -> OutputDict:
    """Run prompt template on dataset and return raw responses with cost logs.

    For each sample:
    1. Format template with sample fields
    2. Call LLM
    3. Return raw response (user handles extraction in validate.py)

    Args:
        prompt_template: Template string with {field} placeholders
        client: LLM client (temperature is set in generation_kwargs)
        context: Context dict with dataset
        dataset_key: Key for dataset in context (default: "train_dataset")

    Returns:
        OutputDict with raw responses (in "predictions" key) and call_logs
    """
    return asyncio.run(
        _run_prompts_async(
            prompt_template,
            client,
            context,
            dataset_key,
        )
    )
