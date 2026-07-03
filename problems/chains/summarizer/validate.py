"""Score a CARL chain wire-JSON genome: mean ROUGE-L F1 on the summarizer eval set."""

from __future__ import annotations

from statistics import mean

from mmar_carl.chain import ReasoningChain

from problems.chains.chain_runner import run_chain_on_dataset
from problems.chains.client import LLMClient
from problems.chains.summarizer.rouge import rouge_l_f1
from problems.chains.summarizer.shared_config import (
    get_llm_config,
    load_dataset,
    outer_context_builder,
)
from problems.chains.types import ChainSpec

INVALID_METRICS = {
    "fitness": 0.0,
    "is_valid": 0,
    "n_steps": 0,
    "completion_tokens": 0,
}


class _LogAggregatingClient(LLMClient):
    """The chain runner logs each call into a throwaway ``copy()``; share one log list so usage survives."""

    def copy(self) -> LLMClient:
        clone = super().copy()
        clone._call_logs = self._call_logs
        return clone


def validate(chain_doc: dict):
    try:
        chain = ReasoningChain.from_dict(chain_doc, use_typed_steps=True)
        if not chain.steps:
            raise ValueError("chain has no steps")
    except Exception as e:
        return (dict(INVALID_METRICS), {"error": f"carl_validation_error: {e}"})

    dataset = load_dataset()
    client = _LogAggregatingClient(**get_llm_config())
    spec = ChainSpec(
        system_prompt=chain_doc.get("task_description", ""),
        steps=list(chain.steps),
    )
    results = run_chain_on_dataset(
        spec, client, dataset, outer_context_builder, max_concurrent=8
    )

    scores = [
        rouge_l_f1(result.final_output or "", sample["expected"])
        for sample, result in zip(dataset, results)
    ]
    metrics = {
        "fitness": mean(scores) if scores else 0.0,
        "is_valid": 1,
        "n_steps": len(chain.steps),
        "completion_tokens": sum(log.completion_tokens for log in client.call_logs),
    }
    artifact = {
        "per_sample": [
            {"score": round(score, 4), "output": (result.final_output or "")[:200]}
            for score, result in zip(scores, results)
        ]
    }
    return (metrics, artifact)
